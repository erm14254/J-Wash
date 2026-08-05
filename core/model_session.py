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


@dataclass(frozen=True)
class ModelStatusSnapshot:
    model_session_id: int
    model_state: str
    loaded: bool
    model_meta: MappingProxyType | None
    capability_profile: Any
    lens: Any = None
    interventions: Any = None
    last_generation: Any = None
    operation: OperationStatus | None = None


@dataclass(frozen=True)
class WorkerOutcome:
    value: Any = None
    failure_kind: str | None = None
    failure_message: str | None = None
    http_status: int | None = None

    @property
    def ok(self) -> bool:
        return self.failure_kind is None


class OperationConflict(RuntimeError):
    pass


class ModelStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationContext:
    token: OperationToken
    model_session_id: int
    bundle: LoadedModelBundle
    hf_model: Any
    tokenizer: Any
    jl: Any
    meta: MappingProxyType
    capability_profile: Any
    lens: Any
    intervention_snapshot: Any
    stop_event: Any


class WorkerDispatch:
    """Ownership handoff for async handlers that dispatch synchronous workers."""

    def __init__(self, coordinator: "ModelSessionCoordinator", token: OperationToken, stop_event: Any = None, payload: Any = None):
        self.coordinator = coordinator
        self.token = token
        self.stop_event = stop_event
        self._payload = payload
        self._lock = threading.Lock()
        self._claimed = False
        self._released_before_claim = False

    @property
    def claimed(self):
        with self._lock:
            return self._claimed

    def claim(self) -> bool:
        with self._lock:
            if self._released_before_claim:
                return False
            self._claimed = True
            return True

    def take_payload(self):
        with self._lock:
            payload, self._payload = self._payload, None
            return payload

    def clear_payload(self):
        with self._lock:
            self._payload = None

    def cancel_from_awaiter(self) -> bool:
        with self._lock:
            if not self._claimed:
                self._released_before_claim = True
                self._payload = None
                return self.coordinator.release(self.token)
            self.coordinator.request_cancel(self.token)
            if self.stop_event is not None:
                self.stop_event.set()
            return False

    def release_from_worker(self) -> bool:
        return self.coordinator.release(self.token)


class ModelSessionCoordinator:
    def __init__(self):
        self._lock = threading.Lock()
        self._model_session_id = 0
        self._bundle: LoadedModelBundle | None = None
        self._operation: OperationToken | None = None
        self._cancelled: set[int] = set()
        self._op_ids = itertools.count(1)
        self._lens_binding = None
        self._interventions = self.empty_intervention_record(0)
        self._last_generation = None
        self._lens_binding_id = 0
        self._intervention_revision = 0
        self._copy_hook = None

    @staticmethod
    def empty_intervention_record(session_id: int = 0, *, revision: int = 0, scale: float = 1.0):
        return MappingProxyType({
            "revision": revision,
            "model_session_id": session_id,
            "lens_binding_id": None,
            "scale": scale,
            "mode": "standard",
            "summary": (),
            "active_summary": (),
            "rules": (),
            "active_rules": (),
        })

    def _observe_copy_outside_lock(self, value):
        hook = self._copy_hook
        if hook is not None:
            hook(value)

    def snapshot(self, *, include_bundle: bool = True) -> ModelSessionSnapshot:
        with self._lock:
            raw = self._snapshot_refs_locked(include_bundle=include_bundle)
        self._observe_copy_outside_lock(raw)
        return self._snapshot_from_refs(raw)

    def _copy_small(self, value):
        if self._copy_hook is not None:
            self._copy_hook(value)
        return copy.deepcopy(value)

    @staticmethod
    def _clone_rule_container(rule):
        cloned = dict(rule)
        if "layers" in cloned and cloned["layers"] is not None:
            cloned["layers"] = list(cloned["layers"])
        if "dirs_a" in cloned and cloned["dirs_a"] is not None:
            cloned["dirs_a"] = dict(cloned["dirs_a"])
        if "dirs_b" in cloned and cloned["dirs_b"] is not None:
            cloned["dirs_b"] = dict(cloned["dirs_b"])
        return cloned

    @classmethod
    def _freeze_intervention_record(cls, record):
        if record is None:
            return None
        if not isinstance(record, dict):
            return record
        frozen = {
            "revision": record.get("revision"),
            "model_session_id": record.get("model_session_id"),
            "lens_binding_id": record.get("lens_binding_id"),
            "scale": record.get("scale"),
            "mode": record.get("mode"),
            "summary": copy.deepcopy(record.get("summary") or []),
            "active_summary": copy.deepcopy(record.get("active_summary") or []),
            "rules": [cls._clone_rule_container(r) for r in record.get("rules") or []],
            "active_rules": [cls._clone_rule_container(r) for r in record.get("active_rules") or []],
        }
        return MappingProxyType(frozen)

    @staticmethod
    def _copy_intervention_record(record):
        if record is None:
            return None
        if isinstance(record, MappingProxyType):
            record = dict(record)
        if isinstance(record, dict):
            return {
                "revision": record.get("revision"),
                "model_session_id": record.get("model_session_id"),
                "lens_binding_id": record.get("lens_binding_id"),
                "scale": record.get("scale"),
                "mode": record.get("mode"),
                "summary": list(record.get("summary") or []),
                "active_summary": list(record.get("active_summary") or []),
                "rules": [ModelSessionCoordinator._clone_rule_container(r) for r in record.get("rules") or []],
                "active_rules": [ModelSessionCoordinator._clone_rule_container(r) for r in record.get("active_rules") or []],
            }
        return record

    def _operation_status_locked(self):
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
        return op

    @staticmethod
    def _light_lens(record):
        if record is None:
            return None
        if isinstance(record, MappingProxyType):
            record = dict(record)
        if isinstance(record, dict):
            return MappingProxyType(dict(record))
        meta = getattr(record, "meta", None)
        return MappingProxyType(dict(meta)) if meta is not None else None

    @staticmethod
    def _light_interventions(record):
        if record is None:
            return None
        if isinstance(record, MappingProxyType):
            record = dict(record)
        if isinstance(record, dict):
            return {
                "revision": record.get("revision"),
                "model_session_id": record.get("model_session_id"),
                "lens_binding_id": record.get("lens_binding_id"),
                "scale": record.get("scale"),
                "mode": record.get("mode"),
                "summary": tuple(record.get("summary") or ()),
                "active_summary": tuple(record.get("active_summary") or ()),
            }
        return record

    def _snapshot_refs_locked(self, *, include_bundle: bool = True):
        op = self._operation_status_locked()
        return {
            "model_session_id": self._model_session_id,
            "model_state": "loaded" if self._bundle is not None else "unloaded",
            "bundle": self._bundle if include_bundle else None,
            "loaded": self._bundle is not None,
            "lens_binding": self._lens_binding,
            "interventions": self._interventions,
            "last_generation": self._last_generation,
            "operation": op,
            "include_bundle": include_bundle,
        }

    @classmethod
    def _snapshot_from_refs(cls, raw) -> ModelSessionSnapshot:
        include_bundle = raw["include_bundle"]
        return ModelSessionSnapshot(
            model_session_id=raw["model_session_id"],
            model_state=raw["model_state"],
            bundle=raw["bundle"],
            loaded=raw["loaded"],
            lens_binding=raw["lens_binding"] if include_bundle else cls._light_lens(raw["lens_binding"]),
            interventions=raw["interventions"] if include_bundle else cls._light_interventions(raw["interventions"]),
            last_generation=raw["last_generation"],
            operation=raw["operation"],
        )

    def status_snapshot(self) -> ModelStatusSnapshot:
        with self._lock:
            bundle = self._bundle
            raw = {
                "model_session_id": self._model_session_id,
                "model_state": "loaded" if bundle is not None else "unloaded",
                "loaded": bundle is not None,
                "model_meta": bundle.meta if bundle is not None else None,
                "capability_profile": bundle.capability_profile if bundle is not None else None,
                "lens": self._lens_binding,
                "interventions": self._interventions,
                "last_generation": self._last_generation,
                "operation": self._operation_status_locked(),
            }
        self._observe_copy_outside_lock(raw)
        return ModelStatusSnapshot(
            model_session_id=raw["model_session_id"],
            model_state=raw["model_state"],
            loaded=raw["loaded"],
            model_meta=raw["model_meta"],
            capability_profile=raw["capability_profile"],
            lens=self._light_lens(raw["lens"]),
            interventions=self._light_interventions(raw["interventions"]),
            last_generation=raw["last_generation"],
            operation=raw["operation"],
        )

    @property
    def busy(self):
        with self._lock:
            return None if self._operation is None else self._operation.type.value

    def acquire(self, op_type: OperationType | str, *, requires_loaded: bool | None = None, include_bundle: bool = True) -> tuple[OperationToken, ModelSessionSnapshot]:
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
            raw = self._snapshot_refs_locked(include_bundle=include_bundle)
        try:
            self._observe_copy_outside_lock(raw)
            return token, self._snapshot_from_refs(raw)
        except Exception:
            with self._lock:
                if self._operation == token:
                    self._operation = None
                    self._cancelled.discard(token.id)
            raise

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

    def withdraw_loaded(self, token: OperationToken | None = None, *, intervention_record: Any = None) -> tuple[LoadedModelBundle | None, int, bool]:
        prepared_interventions = (
            self._freeze_intervention_record(intervention_record)
            if intervention_record is not None
            else None
        )
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
            self._interventions = prepared_interventions or self.empty_intervention_record(
                self._model_session_id,
                revision=self._intervention_revision,
            )
            return old, self._model_session_id, True

    def update_lens_binding(self, token: OperationToken, binding: Any) -> int:
        prepared = self._copy_small(binding)
        with self._lock:
            if self._operation != token:
                raise OperationConflict("stale lens operation")
            self._lens_binding_id += 1
            self._lens_binding = prepared
            return self._lens_binding_id

    def clear_lens_binding(self, token: OperationToken | None = None):
        with self._lock:
            if token is not None and self._operation != token:
                raise OperationConflict("stale lens operation")
            self._lens_binding = None

    def update_interventions(self, token: OperationToken, snapshot: Any) -> int:
        prepared = self._freeze_intervention_record(snapshot)
        with self._lock:
            if self._operation != token:
                raise OperationConflict("stale intervention operation")
            self._intervention_revision += 1
            self._interventions = prepared
            return self._intervention_revision

    def bootstrap_interventions_for_test(self, snapshot: Any) -> int:
        """Install synthetic intervention status without operation ownership.

        This is intentionally reserved for tests/bootstrap data where no
        production request is mutating live model-bound state.
        """
        prepared = self._freeze_intervention_record(snapshot)
        with self._lock:
            self._intervention_revision += 1
            self._interventions = prepared
            return self._intervention_revision

    def publish_last_generation(self, token: OperationToken, session_id: int, data: Any) -> bool:
        prepared = self._copy_small(data)
        with self._lock:
            if self._operation != token or self._model_session_id != session_id:
                return False
            self._last_generation = prepared
            return True
