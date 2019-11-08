from argparse import ArgumentParser
from bz2 import open as bz2_open
from json import dump as json_dump, load as json_load
from pathlib import Path
from pickle import load as pickle_load
from typing import Optional

from torch import load as torch_load, no_grad

from mloncode.datasets.codrep_dataset import CodRepDataset
from mloncode.pipelines.codrep.cli_builder import CLIBuilder
from mloncode.pipelines.codrep.parse import parse
from mloncode.pipelines.codrep.tensorize import tensorize
from mloncode.pipelines.codrep.train import build_model
from mloncode.pipelines.pipeline import register_step
from mloncode.utils.config import Config
from mloncode.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_raw_dir()
    cli_builder.add_uasts_dir()
    cli_builder.add_instance_file()
    cli_builder.add_tensors_dir()
    parser.add_argument(
        "--checkpoint-file", required=True, help="Path to the model checkpoint."
    )
    cli_builder.add_configs_dir()
    parser.add_argument(
        "--training-configs-dir",
        required=True,
        help="Path to the configs used for training.",
    )
    parser.add_argument(
        "--prefix", required=True, help="Path prefixing the output paths."
    )
    parser.add_argument("--metadata-dir", help="Path to the metadata output directory.")


@register_step(pipeline_name="codrep", parser_definer=add_arguments_to_parser)
def run(
    *,
    raw_dir: str,
    uasts_dir: str,
    instance_file: str,
    tensors_dir: str,
    checkpoint_file: str,
    configs_dir: str,
    training_configs_dir: str,
    prefix: str,
    metadata_dir: Optional[str],
    log_level: str,
) -> None:
    """Run the model and output CodRep predictions."""
    arguments = locals()
    configs_dir_path = Path(configs_dir).expanduser().resolve()
    configs_dir_path.mkdir(parents=True, exist_ok=True)
    training_configs_dir_path = Path(training_configs_dir).expanduser().resolve()
    tensors_dir_path = Path(tensors_dir).expanduser().resolve()
    Config.from_arguments(
        arguments, ["instance_file", "checkpoint_file"], "configs_dir"
    ).save(configs_dir_path / "train.json")
    logger = setup_logging(__name__, log_level)

    training_configs = {}
    for step in ["parse", "tensorize", "train"]:
        with (training_configs_dir_path / step).with_suffix(".json").open(
            "r", encoding="utf8"
        ) as fh:
            training_configs[step] = json_load(fh)

    parse(
        raw_dir=raw_dir,
        uasts_dir=uasts_dir,
        configs_dir=configs_dir,
        log_level=log_level,
    )

    tensorize(
        uasts_dir=uasts_dir,
        instance_file=instance_file,
        tensors_dir=tensors_dir,
        configs_dir=configs_dir,
        n_workers=training_configs["tensorize"]["options"]["n_workers"],
        pickle_protocol=training_configs["tensorize"]["options"]["pickle_protocol"],
        log_level=log_level,
    )

    dataset = CodRepDataset(input_dir=tensors_dir_path)
    logger.info(f"Dataset of size {len(dataset)}")

    with bz2_open(instance_file, "rb") as fh_instance:
        instance = pickle_load(fh_instance)

    model = build_model(
        instance=instance,
        model_decoder_type=training_configs["train"]["options"]["model_decoder_type"],
        model_encoder_iterations=training_configs["train"]["options"][
            "model_encoder_iterations"
        ],
        model_encoder_output_dim=training_configs["train"]["options"][
            "model_encoder_output_dim"
        ],
        model_encoder_message_dim=training_configs["train"]["options"][
            "model_encoder_message_dim"
        ],
        model_learning_rate=training_configs["train"]["options"]["model_learning_rate"],
        model_batch_size=training_configs["train"]["options"]["model_batch_size"],
        train_dataset=dataset,
        eval_dataset=None,
        test_dataset=None,
    )

    # The model needs a forward to be completely initialized.
    model.training_step(instance.collate([dataset[0]]), 0)
    logger.info(f"Configured model {model}")

    model.load_state_dict(
        torch_load(checkpoint_file, map_location="cpu")["model_state_dict"]
    )
    model.eval()
    logger.info(f"Loaded model parameters from %s", checkpoint_file)

    metadata = None if metadata_dir is None else model.build_metadata()
    metadata_output = (
        None if metadata_dir is None else Path(metadata_dir) / "metadata.json"
    )

    dataloader = model.train_dataloader()

    graph_field = instance.get_field_by_type("graph")
    label_field = instance.get_field_by_type("label")
    indexes_field = instance.get_field_by_type("indexes")
    metadata_field = instance.get_field_by_type("metadata")
    graph_input_fields = instance.get_fields_by_type("input")
    graph_input_dimensions = [48, 48, 32]
    feature_names = [field.name for field in graph_input_fields]
    with no_grad():
        for batch in dataloader:
            graph, etypes = batch[graph_field.name]
            features = [batch[field_name] for field_name in feature_names]
            indexes, offsets = batch[indexes_field.name].indexes
            forward = model.forward(graph, etypes, features, indexes)
            model.decode(
                batched_graph=graph,
                indexes=indexes,
                offsets=offsets,
                forward=forward,
                paths=batch[metadata_field.name],
                prefix=prefix,
                metadata=metadata,
            )

    if metadata_output is not None:
        with metadata_output.open("w", encoding="utf8") as fh:
            json_dump(metadata, fh)
