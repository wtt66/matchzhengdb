from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

__version__ = "0.0-stub"


class UMAP:
    """Lightweight UMAP compatibility stub.

    BERTopic only requires an object with fit/transform/fit_transform methods.
    This stub uses PCA or TruncatedSVD to avoid heavyweight numba/umap imports
    in constrained Windows environments.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        metric: str = "euclidean",
        random_state: int | None = 42,
        low_memory: bool = False,
        **_: object,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.low_memory = low_memory
        self._model = None

    def _build_model(self, X):
        max_components = min(self.n_components, max(1, min(X.shape[0] - 1, X.shape[1] - 1)))
        if hasattr(X, "tocsr"):
            return TruncatedSVD(n_components=max_components, random_state=self.random_state)
        return PCA(n_components=max_components, random_state=self.random_state)

    def fit(self, X, y=None):
        self._model = self._build_model(X)
        self._model.fit(X)
        return self

    def transform(self, X):
        if self._model is None:
            self.fit(X)
        return self._model.transform(X)

    def fit_transform(self, X, y=None):
        self._model = self._build_model(X)
        return self._model.fit_transform(X)
