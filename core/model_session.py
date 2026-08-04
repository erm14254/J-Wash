from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
import copy
import itertools
import threading
import time
from typing import Any


class OperationType(str, Enum):
    LOAD = "load"
    UNLOAD = "unload"
    GENERATE = "generate"
    LENS_UPDATE = "lens_update"
    INTERVENTION_UPDATE = "intervention_update"
    EXPORT = "export"
    GGUF_BAKE = "gguf_bake"
    MODEL_DELETE = "model_delete"
    MODEL_READ = "model_read"
    FIT = "fit"


@dataclass(frozen=True)
class LoadedModelBundle:
    hf_model: Any
    tokenizer: Any
    jl: Any
    meta: MappingProxyType
    capability_profile: Any

    @classmethod
    def from_parts(cls, hf_model: Any, tokenizer: Any, jl: Any, meta: dict, capability_profile: Any):
        return cls(hf_model, tokenizer, jl, MappingProxyType(copy.deepcopy(meta)), capability_profile)


@dataclass(frozen=True)
class OperationStatus:
    id: int
    type: str
    acquired_session: int
    current_session: int
    start_timestamp: float
    cancellation_requested: bool = False


@dataclass(frozen=True)
class OperationToken:
    id: int
    type: OperationType
    acquired_session: int
    expected_session: int
    requires_loaded: bool | None
    start_timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ModelSessionSnapshot:
    model_session_id: int
    model_state: str
    bundle: LoadedModelBundle | None
    loaded: bool
    lens_binding: Any = None
    interventions: Any = None
    last_generation: Any = None
    operation: OperationStatus | None = None


class OperationConflict(RuntimeError):
    pass


class ModelStateError(RuntimeError):
    pass


class ModelSessionCoordinator:
    def __init__(self):
        self._lock = threading.Lock()
        self._model_session_id = 0
        self._bundle: LoadedModelBundle | None = None
        self._operation: OperationToken | None = None
        self._cancelled: set[int] = set()
        self._op_ids = itertools.count(1)
        self._lens_binding = None
        self._interventions = None
        self._last_generation = None
        self._lens_binding_id = 0
        self._intervention_revision = 0

    def snapshot(self) -> ModelSessionSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> ModelSessionSnapshot:
        op = None
        if self._operation is not None:
            op = OperationStatus(
                id=self._operation.id,
                type=self._operation.type.value,
                acquired_session=self._operation.acquired_session,
                current_session=self._model_session_id,
                start_timestamp=self._operation.start_timestamp,
                cancellation_requested=self._operation.id in self._cancelled,
            )
        return ModelSessionSnapshot(
            model_session_id=self._model_session_id,
            model_state="loaded" if self._bundle is not None else "unloaded",
            bundle=self._bundle,
            loaded=self._bundle is not None,
            lens_binding=copy.deepcopy(self._lens_binding),
            interventions=copy.deepcopy(self._interventions),
            last_generation=copy.deepcopy(self._last_generation),
            operation=op,
        )

    @property
    def busy(self):
        with self._lock:
            return None if self._operation is None else self._operation.type.value

    def acquire(self, op_type: OperationType | str, *, requires_loaded: bool | None = None) -> tuple[OperationToken, ModelSessionSnapshot]:
        op_type = OperationType(op_type)
        with self._lock:
            if self._operation is not None:
                raise OperationConflict(f"model operation already in progress: {self._operation.type.value}")
            loaded = self._bundle is not None
            if requires_loaded is True and not loaded:
                raise ModelStateError("no model loaded")
            if requires_loaded is False and loaded:
                raise ModelStateError("model must be unloaded")
            token = OperationToken(next(self._op_ids), op_type, self._model_session_id, self._model_session_id, requires_loaded)
            self._operation = token
            self._cancelled.discard(token.id)
            return token, self._snapshot_locked()

    def is_current(self, token: OperationToken) -> bool:
        with self._lock:
            return self._operation == token

    def request_cancel(self, token: OperationToken) -> bool:
        with self._lock:
            if self._operation != token:
                return False
            self._cancelled.add(token.id)
            return True

    def is_cancelled(self, token: OperationToken) -> bool:
        with self._lock:
            return token.id in self._cancelled

    def release(self, token: OperationToken) -> bool:
        with self._lock:
            if self._operation != token:
                return False
            self._operation = None
            self._cancelled.discard(token.id)
            return True

    def publish_loaded(self, token: OperationToken, bundle: LoadedModelBundle, *, expected_unloaded_session: int | None = None) -> int:
        with self._lock:
            if self._operation != token:
                raise OperationConflict("stale model operation")
            if self._bundle is not None:
                raise ModelStateError("model already loaded")
            expected = token.expected_session if expected_unloaded_session is None else expected_unloaded_session
            if self._model_session_id != expected:
                raise OperationConflict("stale model session")
            self._model_session_id += 1
            self._bundle = bundle
            return self._model_session_id

    def withdraw_loaded(self, token: OperationToken | None = None) -> tuple[LoadedModelBundle | None, int, bool]:
        with self._lock:
            if token is not None and self._operation != token:
                raise OperationConflict("stale model operation")
            old = self._bundle
            if old is None:
                return None, self._model_session_id, False
            self._model_session_id += 1
            self._bundle = None
            self._lens_binding = None
            self._last_generation = None
            self._interventions = None
            return old, self._model_session_id, True

    def update_lens_binding(self, token: OperationToken, binding: Any) -> int:
        with self._lock:
            if self._operation != token:
                raise OperationConflict("stale lens operation")
            self._lens_binding_id += 1
            self._lens_binding = copy.deepcopy(binding)
            return self._lens_binding_id

    def clear_lens_binding(self):
        with self._lock:
            self._lens_binding = None

    def update_interventions(self, token: OperationToken | None, snapshot: Any) -> int:
        with self._lock:
            if token is not None and self._operation != token:
                raise OperationConflict("stale intervention operation")
            self._intervention_revision += 1
            self._interventions = copy.deepcopy(snapshot)
            return self._intervention_revision

    def publish_last_generation(self, token: OperationToken, session_id: int, data: Any) -> bool:
        with self._lock:
            if self._operation != token or self._model_session_id != session_id:
                return False
            self._last_generation = copy.deepcopy(data)
            return True
