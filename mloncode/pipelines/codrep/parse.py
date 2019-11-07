from argparse import ArgumentParser
from pathlib import Path
from time import time

from asdf import AsdfFile

from mloncode.data.types.codrep_label import CodRepLabel
from mloncode.parsing.java_parser import JavaParser
from mloncode.parsing.parser import ParsingException
from mloncode.pipelines.codrep.cli_builder import CLIBuilder
from mloncode.pipelines.pipeline import register_step
from mloncode.utils.config import Config
from mloncode.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_raw_dir()
    cli_builder.add_uasts_dir()
    cli_builder.add_configs_dir()


@register_step(pipeline_name="codrep", parser_definer=add_arguments_to_parser)
def parse(*, raw_dir: str, uasts_dir: str, configs_dir: str, log_level: str) -> None:
    """Parse a CodRep 2019 dataset into UASTs."""
    Config.from_arguments(locals(), ["raw_dir", "uasts_dir"], "configs_dir").save(
        Path(configs_dir) / "parse.json"
    )
    logger = setup_logging(__name__, log_level)
    raw_dir_path = Path(raw_dir).expanduser().resolve()
    uasts_dir_path = Path(uasts_dir).expanduser().resolve()
    uasts_dir_path.mkdir(parents=True, exist_ok=True)

    parser = JavaParser(split_formatting=True)
    logger.info("Parsing %s", raw_dir_path)
    labels_file = raw_dir_path / "out.txt"
    extract_labels = labels_file.is_file()
    if extract_labels:
        error_offsets = {}
        for i, line in enumerate(labels_file.open("r", encoding="utf8")):
            error_offsets["%d.txt" % i] = int(line) - 1
    for file_path in raw_dir_path.rglob("*.txt"):
        if file_path.samefile(labels_file):
            continue
        file_path_relative = file_path.relative_to(raw_dir_path)
        start = time()
        logger.debug("Parsing %s", file_path_relative)
        try:
            nodes = parser.parse(raw_dir_path, file_path_relative)
        except ParsingException:
            continue
        logger.debug(
            "Parsed  %s into %d nodes in %.2fms",
            file_path_relative,
            len(nodes.nodes),
            (time() - start) * 1000,
        )
        error_node_index = None
        if extract_labels:
            error_offset = error_offsets[file_path.name]
            for formatting_i, i in enumerate(nodes.formatting_indexes):
                node = nodes.nodes[i]
                if node.start == error_offset:
                    error_node_index = formatting_i
                    break
            else:
                logger.warning(
                    "Could not retrieve a formatting node for the error at offset %d "
                    "of file %s.",
                    error_offset,
                    file_path.with_suffix("").name,
                )
                continue
        codrep_label = CodRepLabel(
            error_index=error_node_index,
            n_formatting_nodes=len(nodes.formatting_indexes),
        )
        output_subdirectory = uasts_dir_path / file_path_relative.parent
        output_subdirectory.mkdir(parents=True, exist_ok=True)
        with (output_subdirectory / file_path.with_suffix(".asdf").name).open(
            "wb"
        ) as fh:
            af = AsdfFile(
                dict(
                    nodes=nodes.to_tree(file_path.read_text(encoding="utf-8")),
                    codrep_label=codrep_label.to_tree(),
                    filepath=str(file_path_relative),
                )
            )
            af.write_to(fh, all_array_compression="bzp2")
