from argparse import ArgumentParser
from bz2 import open as bz2_open
from enum import Enum
from pathlib import Path
from pickle import load as pickle_load
from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.nn import Linear, LSTM, Module, Sequential
from torch.utils.data import Dataset, random_split

from mloncode.data.instance import Instance
from mloncode.datasets.codrep_dataset import CodRepDataset
from mloncode.models.codrep_model import CodRepModel
from mloncode.modules.graph_encoders.ggnn import GGNN
from mloncode.modules.misc.graph_embedding import GraphEmbedding
from mloncode.modules.misc.item_getter import ItemGetter
from mloncode.modules.misc.squeezer import Squeezer
from mloncode.modules.misc.unsqueezer import Unsqueezer
from mloncode.pipelines.codrep.cli_builder import CLIBuilder
from mloncode.pipelines.pipeline import register_step
from mloncode.utils.config import Config
from mloncode.utils.helpers import setup_logging


class DecoderType(Enum):
    FF = "ff"
    RNN = "rnn"


class Optimizer(Enum):
    Adam = "adam"
    SGD = "sgd"


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_instance_file()
    cli_builder.add_tensors_dir()
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory where the run artifacts will be output.",
    )
    cli_builder.add_configs_dir()
    parser.add_argument(
        "--model-encoder-iterations",
        help="Number of message passing iterations to apply (defaults to %(default)s).",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--model-encoder-output-dim",
        help="Dimensionality of the encoder output (defaults to %(default)s).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model-encoder-message-dim",
        help="Dimensionality of the encoder messages (defaults to %(default)s).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model-decoder-type",
        help="Type of decoder to use (defaults to %(default)s).",
        choices=[t.value for t in DecoderType],
        default=DecoderType.FF.value,
    )
    parser.add_argument(
        "--model-batch-size",
        help="Size of the training batches (defaults to %(default)s).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--model-learning-rate",
        help="Learning rate of the optimizer (defaults to %(default)s).",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--trainer-epochs",
        help="Number of epochs to train for (defaults to %(default)s).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--trainer-eval-every",
        help="Number of iterations before an evaluation epoch (defaults to "
        "%(default)s).",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--trainer-limit-epochs-at",
        help="Number of batches to limit an epoch to.",
        type=int,
    )
    parser.add_argument(
        "--trainer-train-eval-split",
        help="Proportion kept for training (defaults to %(default)s). "
        "Rest goes to evaluation.",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--trainer-selection-metric",
        help="Name of the metric to use for checkpoint selection "
        "(defaults to %(default)s).",
        default="mrr",
    )
    parser.add_argument(
        "--trainer-kept-checkpoints",
        help="Number of best checkpoints to keep (defaults to %(default)s).",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--trainer-cuda", help="CUDA index of the device to use for training.", type=int
    )


@register_step(
    pipeline_name="codrep",
    parser_definer=add_arguments_to_parser,
    graceful_keyboard_interruption=True,
)
def train(
    *,
    instance_file: str,
    tensors_dir: str,
    train_dir: str,
    configs_dir: str,
    model_encoder_iterations: int,
    model_encoder_output_dim: int,
    model_encoder_message_dim: int,
    model_decoder_type: str,
    model_learning_rate: float,
    model_batch_size: int,
    trainer_epochs: int,
    trainer_eval_every: int,
    trainer_limit_epochs_at: Optional[int],
    trainer_train_eval_split: float,
    trainer_selection_metric: str,
    trainer_kept_checkpoints: int,
    trainer_cuda: Optional[int],
    log_level: str,
) -> None:
    """Run the training."""
    Config.from_arguments(
        locals(), ["instance_file", "tensors_dir", "train_dir"], "configs_dir"
    ).save(Path(configs_dir) / "train.json")
    logger = setup_logging(__name__, log_level)

    tensors_dir_path = Path(tensors_dir).expanduser().resolve()
    train_dir_path = Path(train_dir).expanduser().resolve()
    train_dir_path.mkdir(parents=True, exist_ok=True)

    with bz2_open(instance_file, "rb") as fh:
        instance = pickle_load(fh)

    dataset = CodRepDataset(input_dir=tensors_dir_path)
    logger.info("Dataset of size %d", len(dataset))

    train_length = round(0.9 * len(dataset))
    eval_length = round(0.05 * len(dataset))
    test_length = len(dataset) - train_length - eval_length

    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_length, eval_length, test_length]
    )

    if trainer_cuda is not None:
        if not cuda_is_available():
            raise RuntimeError("CUDA is not available on this system.")
        device = torch_device("cuda:%d" % trainer_cuda)
    else:
        device = torch_device("cpu")
    model = build_model(
        instance=instance,
        model_encoder_iterations=model_encoder_iterations,
        model_encoder_output_dim=model_encoder_output_dim,
        model_encoder_message_dim=model_encoder_message_dim,
        model_decoder_type=model_decoder_type,
        model_learning_rate=model_learning_rate,
        model_batch_size=model_batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
    )
    # The model needs a forward to be completely initialized.
    model.training_step(instance.collate([dataset[0]]), 0)
    logger.info("Configured model %s", model)

    checkpoint_callback = ModelCheckpoint(
        filepath=train_dir,
        save_best_only=True,
        verbose=True,
        monitor="eval_mrr",
        mode="max",
        prefix="",
    )

    trainer = Trainer(
        default_save_path=train_dir, checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


def build_model(
    instance: Instance,
    model_encoder_iterations: int,
    model_encoder_output_dim: int,
    model_encoder_message_dim: int,
    model_decoder_type: str,
    model_learning_rate: float,
    model_batch_size: int,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    test_dataset: Optional[Dataset],
) -> CodRepModel:
    graph_field = instance.get_field_by_type("graph")
    label_field = instance.get_field_by_type("label")
    indexes_field = instance.get_field_by_type("indexes")
    graph_input_fields = instance.get_fields_by_type("input")
    graph_input_dimensions = [48, 48, 32]
    feature_names = [field.name for field in graph_input_fields]
    if DecoderType(model_decoder_type) is DecoderType.FF:
        class_projection: Module = Linear(
            in_features=model_encoder_output_dim, out_features=2
        )
    else:
        class_projection = Sequential(
            Unsqueezer(0),
            LSTM(
                input_size=model_encoder_output_dim,
                hidden_size=model_encoder_output_dim // 2,
                batch_first=True,
                bidirectional=True,
            ),
            ItemGetter(0),
            Squeezer(),
            Linear(in_features=model_encoder_output_dim, out_features=2),
        )
    return CodRepModel(
        graph_embedder=GraphEmbedding(
            graph_input_dimensions,
            [field.vocabulary for field in graph_input_fields],  # type: ignore
        ),
        graph_encoder=GGNN(
            in_feats=sum(graph_input_dimensions),
            out_feats=model_encoder_output_dim,
            n_steps=model_encoder_iterations,
            n_etypes=len(graph_field.vocabulary),  # type: ignore
        ),
        class_projection=class_projection,
        graph_field_name=graph_field.name,
        feature_field_names=feature_names,
        indexes_field_name=indexes_field.name,
        label_field_name=label_field.name,
        batch_size=model_batch_size,
        lr=model_learning_rate,
        instance=instance,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
    )
