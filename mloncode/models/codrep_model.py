from bisect import bisect_right
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

from dgl import BatchedDGLGraph, unbatch
from pytorch_lightning import data_loader, LightningModule
from torch import stack as torch_stack, Tensor
from torch.nn import LogSoftmax, Module, NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from mloncode.data.instance import Instance
from mloncode.modules.graph_encoders.graph_encoder import GraphEncoder
from mloncode.modules.misc.graph_embedding import GraphEmbedding
from mloncode.modules.misc.selector import Selector
from mloncode.utils.torch_helpers import data_if_packed


class CodRepModel(LightningModule):
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
        instance: Instance,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        batch_size: int,
        lr: float,
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
        self.instance = instance
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

    def forward(
        self,
        graph: BatchedDGLGraph,
        etypes: Tensor,
        features: List[Tensor],
        formatting_indexes: Tensor,
    ) -> Tensor:
        """Forward pass of an embedder, encoder and decoder."""
        graph = self.graph_embedder(graph=graph, features=features)
        encodings = self.graph_encoder(
            graph=graph, feat=graph.ndata["x"], etypes=etypes
        )
        label_encodings = self.selector(tensor=encodings, indexes=formatting_indexes)
        projections = self.class_projection(label_encodings)
        return self.softmax(projections)

    def training_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        graph, etypes = batch[self.graph_field_name]
        features = [batch[field_name] for field_name in self.feature_field_names]
        formatting_indexes = batch[self.indexes_field_name].indexes
        labels = batch[self.label_field_name]
        forward = self.forward(graph, etypes, features, formatting_indexes)
        loss = NLLLoss(
            weight=forward.new_tensor(
                [graph.batch_size, formatting_indexes.numel() - graph.batch_size]
            )
        )(forward, labels)
        loss_item = data_if_packed(loss).item()
        return dict(
            loss=loss,
            log=dict(
                train_cross_entropy=loss_item,
                perplexity=2 ** loss_item,
                mrr=self.mrr(labels, graph, formatting_indexes, forward),
            ),
        )

    def validation_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        graph, etypes = batch[self.graph_field_name]
        features = [batch[field_name] for field_name in self.feature_field_names]
        formatting_indexes = batch[self.indexes_field_name].indexes
        labels = batch[self.label_field_name]
        forward = self.forward(graph, etypes, features, formatting_indexes)
        loss = NLLLoss(
            weight=forward.new_tensor(
                [graph.batch_size, formatting_indexes.numel() - graph.batch_size]
            )
        )(forward, labels)
        return dict(
            eval_loss=loss,
            eval_cross_entropy=self.cross_entropy(loss),
            eval_perplexity=self.perplexity(loss),
            eval_mrr=self.mrr(labels, graph, formatting_indexes, forward),
        )

    def validation_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_loss = torch_stack([x["eval_loss"] for x in outputs]).mean()

        def average(input_list: List[Dict[str, Any]], key: str) -> float:
            return sum(i[key] for i in input_list) / len(input_list)

        return dict(
            avg_val_loss=avg_loss,
            log=dict(
                eval_loss=avg_loss,
                eval_cross_entropy=average(outputs, "eval_cross_entropy"),
                eval_perplexity=average(outputs, "eval_perplexity"),
                eval_mrr=average(outputs, "eval_mrr"),
            ),
        )

    def decode(
        self,
        *,
        batched_graph: BatchedDGLGraph,
        indexes: Tensor,
        offsets: Tensor,
        forward: Tensor,
        paths: List[str],
        prefix: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        graphs = unbatch(batched_graph)
        start = 0
        total_number_of_nodes = 0
        bounds = []
        numpy_indexes = indexes.cpu().numpy()
        for graph in graphs:
            total_number_of_nodes += graph.number_of_nodes()
            end = bisect_right(numpy_indexes, total_number_of_nodes - 1)
            bounds.append((start, end))
            start = end
        for (start, end), path in zip(bounds, paths):
            path_probas = forward[start:end, 1]
            path_indexes = offsets[start:end]
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

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Scheduler, List[Union[Optimizer, Scheduler]]]:
        return Adam(self.parameters(), lr=self.lr)

    @data_loader
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.instance.collate,
            batch_size=self.batch_size,
            num_workers=1,
        )

    @data_loader
    def val_dataloader(self) -> DataLoader:
        return (
            DataLoader(
                self.eval_dataset,
                shuffle=True,
                collate_fn=self.instance.collate,
                batch_size=self.batch_size,
                num_workers=1,
            )
            if self.eval_dataset is not None
            else None
        )

    @data_loader
    def test_dataloader(self) -> DataLoader:
        return (
            DataLoader(
                self.test_dataset,
                shuffle=True,
                collate_fn=self.instance.collate,
                batch_size=self.batch_size,
                num_workers=1,
            )
            if self.test_dataset is not None
            else None
        )

    @staticmethod
    def mrr(
        labels: Tensor, batched_graph: BatchedDGLGraph, indexes: Tensor, forward: Tensor
    ) -> float:
        ground_truth = data_if_packed(labels).argmax(dim=0)
        graphs = unbatch(batched_graph)
        start = 0
        total_number_of_nodes = 0
        bounds = []
        numpy_indexes = indexes.cpu().numpy()
        for graph in graphs:
            total_number_of_nodes += graph.number_of_nodes()
            end = bisect_right(numpy_indexes, total_number_of_nodes - 1)
            bounds.append((start, end))
            start = end
        ranks = []
        for start, end in bounds:
            predictions = data_if_packed(forward)[start:end, 1].argsort(descending=True)
            ground_truth = data_if_packed(labels)[start:end].argmax(dim=0)
            ranks.append((predictions == ground_truth).nonzero().item())
        return sum(1 / (rank + 1) for rank in ranks) / len(ranks)

    @staticmethod
    def cross_entropy(loss: Tensor) -> float:
        return data_if_packed(loss).item()

    @staticmethod
    def perplexity(loss: Tensor) -> float:
        return 2 ** data_if_packed(loss).item()
