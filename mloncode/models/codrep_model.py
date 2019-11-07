from bisect import bisect_right
from logging import getLogger
from typing import Any, Dict, List, Optional

from dgl import unbatch
from torch.nn import LogSoftmax, Module, NLLLoss

from mloncode.models.model import Model
from mloncode.modules.graph_encoders.graph_encoder import GraphEncoder
from mloncode.modules.misc.graph_embedding import GraphEmbedding
from mloncode.modules.misc.selector import Selector


class CodRepModel(Model):
    """
    CodRep Model.

    Uses a graph encoder and a projection decoder with an optional RNN.
    """

    _logger = getLogger(__name__)

    def __init__(
        self,
        graph_embedder: GraphEmbedding,
        graph_encoder: GraphEncoder,
        class_projection: Module,
        graph_field_name: str,
        feature_field_names: List[str],
        indexes_field_name: str,
        label_field_name: str,
    ) -> None:
        """Construct a complete model."""
        super().__init__()
        self.graph_embedder = graph_embedder
        self.graph_encoder = graph_encoder
        self.selector = Selector()
        self.class_projection = class_projection
        self.graph_field_name = graph_field_name
        self.feature_field_names = feature_field_names
        self.indexes_field_name = indexes_field_name
        self.label_field_name = label_field_name
        self.softmax = LogSoftmax(dim=1)

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """Forward pass of an embedder, encoder and decoder."""
        if "forward" in sample:
            raise RuntimeError("Forward already computed.")
        if "loss" in sample:
            raise RuntimeError("Loss already computed.")
        graph, etypes = sample[self.graph_field_name]
        features = [sample[field_name] for field_name in self.feature_field_names]
        formatting_indexes = sample[self.indexes_field_name].indexes
        graph = self.graph_embedder(graph=graph, features=features)
        encodings = self.graph_encoder(
            graph=graph, feat=graph.ndata["x"], etypes=etypes
        )
        label_encodings = self.selector(tensor=encodings, indexes=formatting_indexes)
        projections = self.class_projection(label_encodings)
        softmaxed = self.softmax(projections)
        labels = sample[self.label_field_name]
        sample["forward"] = softmaxed
        if labels is not None:
            sample["loss"] = NLLLoss(
                weight=softmaxed.new_tensor(
                    [graph.batch_size, formatting_indexes.numel() - graph.batch_size]
                )
            )(softmaxed, labels)
        return sample

    def decode(
        self,
        *,
        sample: Dict[str, Any],
        prefix: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        batched_graph = sample["typed_dgl_graph"].graph
        graphs = unbatch(batched_graph)
        start = 0
        total_number_of_nodes = 0
        bounds = []
        numpy_indexes = sample["indexes"].indexes.cpu().numpy()
        for graph in graphs:
            total_number_of_nodes += graph.number_of_nodes()
            end = bisect_right(numpy_indexes, total_number_of_nodes - 1)
            bounds.append((start, end))
            start = end
        for (start, end), path in zip(bounds, sample["metadata"]):
            path_probas = sample["forward"][start:end, 1]
            path_indexes = sample["indexes"].offsets[start:end]
            predictions = path_indexes[path_probas.argsort(descending=True)]
            if metadata is not None and "metadata" in metadata:
                metadata["metadata"][path] = {
                    index: ["%.8f" % (2 ** proba)]
                    for index, proba in zip(path_indexes.tolist(), path_probas.tolist())
                }
            predictions += 1
            print("%s%s %s" % (prefix, path, " ".join(map(str, predictions.numpy()))))

    def build_metadata(self) -> Dict[str, Any]:
        return dict(columns=["Probability"], metadata={})
