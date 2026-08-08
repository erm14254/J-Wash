import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import json
import logging
import mimetypes
import os
import re
import threading
import time
from threading import Event as ThreadingEvent
import uuid
import stat
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core import editing, registry
from core import capabilities
from core.ablation import Interventions, UnknownInterventionRule
from core.neighbors import TokenNeighbors
from core import fitting
from core.fitting import FitManager
from core.gpus import gpu_stats
from core.lens_manager import LensManager
from core.model_session import GenerationContext, LoadedModelBundle, OperationConflict, OperationType, ModelStateError, WorkerDispatch, WorkerOutcome
from core.model_manager import (
    GenerationAttemptOwner,
    ModelManager,
    _resolve_revision,
    resolve_local_dir,
    resolve_source,
)
from core.store import Store, FrameStorageError, FramesNotAttached, FramePointerTurnover, FrameFileMissing

manager = ModelManager()
lens_manager = LensManager()
store = Store()
fit_manager = FitManager()
interventions = Interventions()
neighbors = TokenNeighbors()
_transition_intervention_rollback = None


def _prepare_model_transition_interventions(target_session_id):
    global _transition_intervention_rollback
    previous = interventions.state_record()
    try:
        interventions.clear_for_model_transition()
        snapshot = interventions.snapshot()
        snapshot["model_session_id"] = target_session_id
        snapshot["lens_binding_id"] = None
        _transition_intervention_rollback = previous
        return snapshot
    except Exception:
        interventions.restore_state_record(previous)
        raise


def _rollback_model_transition_interventions():
    global _transition_intervention_rollback
    previous, _transition_intervention_rollback = _transition_intervention_rollback, None
    if previous is not None:
        interventions.restore_state_record(previous)


def _cleanup_model_bound_state(_session_id=None, token=None):
    global _transition_intervention_rollback
    _transition_intervention_rollback = None
    errors = []
    old_lens = None
    try:
        old_lens = lens_manager.withdraw()
        manager.coordinator.clear_lens_binding(token)
    except Exception as exc:
        errors.append(exc)
    try:
        neighbors.reset()
    except Exception as exc:
        errors.append(exc)
    # Intervention clearing is prepared and installed atomically with the
    # authoritative coordinator withdrawal.  This cleanup callback only owns
    # resource/cache cleanup after that transition has linearized.
    cleanup_error = lens_manager.cleanup_withdrawn(old_lens)
    if cleanup_error is not None:
        errors.append(cleanup_error)
    if errors:
        first = errors[0]
        logging.getLogger(__name__).warning(
            "model-bound cleanup completed with %d non-fatal error(s)", len(errors),
            exc_info=(type(first), first, first.__traceback__),
        )


manager.on_model_withdraw = _cleanup_model_bound_state
manager.on_model_transition_prepare = _prepare_model_transition_interventions
manager.on_model_transition_rollback = _rollback_model_transition_interventions

_ws_locks = {}
_loop_holder = {}
_deferred_worker_tasks = set()




@dataclass(frozen=True)
class RouteOutcome:
    value: object | None = None
    status: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class HandoffSetup:
    task: object | None = None
    dispatch: object | None = None
    gate: object | None = None
    state: object | None = None
    status: int | None = None
    kind: str | None = None
    message: str | None = None


def _bounded_failure(exc, fallback="request failed", limit=512):
    try:
        kind = getattr(type(exc), "__name__", "BaseException")[:128]
    except BaseException:
        kind = "BaseException"
    try:
        message = str(exc)
    except BaseException:
        message = fallback
    if not isinstance(message, str):
        message = fallback
    message = message[:limit]
    try:
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
    except BaseException:
        pass
    return kind, message


def _route_value(value=None):
    return RouteOutcome(value=value)


def _route_http(status, message):
    try:
        bounded = str(message)[:512]
    except BaseException:
        bounded = "request failed"
    return RouteOutcome(status=status, message=bounded)


def _raise_route_outcome(outcome: RouteOutcome):
    if outcome.status is not None:
        raise HTTPException(outcome.status, outcome.message or "request failed") from None
    return outcome.value


class PreparedWorkerPayload:
    def __init__(self, payload):
        self._payload = payload

    def take(self):
        payload, self._payload = self._payload, None
        if payload is None:
            raise RuntimeError("prepared payload already transferred")
        return payload


def _route_exception(status, exc):
    _kind, message = _bounded_failure(exc)
    return _route_http(status, message)

def _worker_success(value=None):
    return WorkerOutcome(value=value)


def _worker_failure(exc, *, default_status=500, value_status=422):
    kind, message = _bounded_failure(exc, "worker failure could not be formatted")
    if isinstance(exc, HTTPException):
        try:
            detail = str(exc.detail)[:512]
        except BaseException:
            detail = "request failed"
        outcome = WorkerOutcome(
            failure_kind="http",
            failure_message=detail,
            http_status=exc.status_code,
        )
    else:
        status = (409 if isinstance(exc, editing.PublicationCancelled) else
                  value_status if isinstance(exc, ValueError) else default_status)
        outcome = WorkerOutcome(failure_kind=kind, failure_message=message, http_status=status)
    try:
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
    except BaseException:
        pass
    return outcome


def _release_worker_total(dispatch):
    """Do not let a transient release failure orphan coordinator ownership."""
    while True:
        try:
            if dispatch.release_from_worker():
                return
            if not dispatch.coordinator.is_current(dispatch.token):
                return
        except BaseException:
            pass
        _retry_thread_yield()


def _abort_unclaimed_total(dispatch):
    while True:
        try:
            if dispatch.abort_unclaimed():
                return
            if dispatch.claimed:
                _release_worker_total(dispatch)
                return
            if not dispatch.coordinator.is_current(dispatch.token):
                return
        except BaseException:
            pass
        _retry_thread_yield()


def _retry_thread_yield():
    """Yield a worker thread without allowing an injected yield fault to escape."""
    while True:
        try:
            time.sleep(0)
            return
        except BaseException:
            continue


def _release_token_total(coordinator, token):
    """Release an acquired token only after exact non-ownership is proven."""
    while True:
        try:
            coordinator.release(token)
        except BaseException:
            pass
        try:
            if not coordinator.is_current(token):
                return
        except BaseException:
            pass
        _retry_thread_yield()


def _remember_task_cancellation():
    current = asyncio.current_task()
    if current is not None and hasattr(current, "uncancel"):
        current.uncancel()


async def _cleanup_yield():
    """Yield during cleanup while retaining and reporting caller cancellation."""
    cancelled = False
    while True:
        try:
            await asyncio.sleep(0)
            return cancelled
        except asyncio.CancelledError:
            cancelled = True
            _remember_task_cancellation()
        except BaseException:
            continue


def _execute_dispatched_worker(dispatch, execute, *, default_status=500, value_status=422,
                               rollback=None, teardown=None):
    """Exception-complete claim, payload transfer, body, rollback and release."""
    claimed = False
    payload = result = failure = cancellation = rollback_failure = teardown_failure = None
    try:
        try:
            claimed = bool(dispatch.claim())
        except BaseException as exc:
            cancellation = isinstance(exc, asyncio.CancelledError)
            failure = None if cancellation else _worker_failure(
                exc, default_status=default_status, value_status=value_status,
            )
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            exc = None
            _abort_unclaimed_total(dispatch)
            if cancellation:
                raise asyncio.CancelledError() from None
            return failure
        if not claimed:
            return _worker_success()
        try:
            payload = dispatch.take_payload()
            result = execute(payload)
            return _worker_success(result)
        except BaseException as exc:
            cancellation = isinstance(exc, asyncio.CancelledError)
            failure = None if cancellation else _worker_failure(
                exc, default_status=default_status, value_status=value_status,
            )
            if rollback is not None:
                try:
                    rollback()
                except BaseException as rollback_exc:
                    rollback_failure = _worker_failure(rollback_exc)
                    try:
                        rollback_exc.__traceback__ = rollback_exc.__context__ = rollback_exc.__cause__ = None
                    except BaseException:
                        pass
                    rollback_exc = None
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            exc = None
        finally:
            if teardown is not None:
                try:
                    teardown()
                except BaseException as teardown_exc:
                    teardown_failure = _worker_failure(teardown_exc)
                    try:
                        teardown_exc.__traceback__ = teardown_exc.__context__ = teardown_exc.__cause__ = None
                    except BaseException:
                        pass
                    teardown_exc = None
            payload = result = execute = rollback = teardown = rollback_failure = teardown_failure = None
            try:
                dispatch.clear_payload()
            except BaseException:
                pass
            _release_worker_total(dispatch)
        if cancellation:
            raise asyncio.CancelledError() from None
        return failure
    finally:
        payload = result = execute = rollback = teardown = failure = rollback_failure = teardown_failure = None


def _raise_worker_outcome(outcome: WorkerOutcome):
    if outcome is None:
        return None
    if not isinstance(outcome, WorkerOutcome):
        return outcome
    if outcome.ok:
        return outcome.value
    raise HTTPException(outcome.http_status or 500, outcome.failure_message or "model worker failed") from None


def _coordinated_interventions(snap):
    if snap.interventions is None:
        raise HTTPException(500, "coordinated intervention state unavailable")
    return snap.interventions


class HandoffState(Enum):
    NEW = "NEW"
    ACQUIRED = "ACQUIRED"
    PREPARING = "PREPARING"
    DISPATCH_READY = "DISPATCH_READY"
    WRAPPER_READY = "WRAPPER_READY"
    TASK_CREATED = "TASK_CREATED"
    TASK_RETAINED = "TASK_RETAINED"
    TRANSFERRED = "TRANSFERRED"
    CLOSED = "CLOSED"


class OperationHandoff:
    """Owns an acquired operation until a retained worker can take over."""

    def __init__(self, coordinator, token=None, stop_event=None):
        if stop_event is None and hasattr(token, "set") and hasattr(token, "is_set"):
            stop_event, token = token, None
        self.coordinator = coordinator
        self.token = token
        self.stop_event = stop_event
        self.dispatch = None
        self.wrapper = None
        self.task = None
        self.heavy = {}
        self.transferred = False
        self.closed = False
        self.state = HandoffState.ACQUIRED if token is not None else HandoffState.NEW
        self.snapshot = None

    @classmethod
    @asynccontextmanager
    async def acquire_scope(cls, coordinator, operation_type, *, stop_event=None, **acquire_kwargs):
        """Install cleanup ownership before acquiring a coordinator lease."""
        handoff = cls(coordinator, stop_event=stop_event)
        cancelled = False
        snapshot = None
        try:
            token, snapshot = handoff.acquire(operation_type, **acquire_kwargs)
            handoff.snapshot = snapshot
            yield handoff
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
        finally:
            snapshot = None
            handoff.snapshot = None
            if not handoff.transferred and not handoff.closed:
                cancelled |= await handoff._close_uninterruptibly()
        if cancelled:
            raise asyncio.CancelledError()

    async def _close_uninterruptibly(self):
        """Resolve ownership without a bounded-retry escape hatch."""
        cancelled = False
        while True:
            try:
                if self.closed or self.transferred:
                    break
                if self.token is None or not self.coordinator.is_current(self.token):
                    self.token = None
                    self.closed = True
                    self.state = HandoffState.CLOSED
                    break
            except BaseException:
                pass
            close_coro = close_task = None
            try:
                close_coro = self.close()
                try:
                    close_task = asyncio.create_task(close_coro)
                    close_coro = None
                except BaseException:
                    if close_coro is not None and hasattr(close_coro, "close"):
                        close_coro.close()
                    close_coro = None
                    try:
                        await self.close()
                    except asyncio.CancelledError:
                        cancelled = True
                    except BaseException:
                        pass
                else:
                    try:
                        await _drain_worker_uninterruptibly(close_task)
                        close_task.result()
                    except asyncio.CancelledError:
                        cancelled = True
                    except BaseException:
                        pass
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                pass
            finally:
                while close_task is not None and not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        cancelled = True
                        current = asyncio.current_task()
                        if current is not None and hasattr(current, "uncancel"):
                            current.uncancel()
                    except BaseException:
                        if close_task.done():
                            break
                        cancelled |= await _cleanup_yield()
                if close_coro is not None and hasattr(close_coro, "close"):
                    try:
                        close_coro.close()
                    except BaseException:
                        pass
                close_coro = close_task = None
            if cancelled:
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()
            cancelled |= await _cleanup_yield()
        return cancelled

    def _set_state(self, state):
        allowed = {
            HandoffState.NEW: {HandoffState.ACQUIRED, HandoffState.CLOSED},
            HandoffState.ACQUIRED: {HandoffState.PREPARING, HandoffState.DISPATCH_READY, HandoffState.CLOSED},
            HandoffState.PREPARING: {HandoffState.PREPARING, HandoffState.DISPATCH_READY, HandoffState.CLOSED},
            HandoffState.DISPATCH_READY: {HandoffState.WRAPPER_READY, HandoffState.CLOSED},
            HandoffState.WRAPPER_READY: {HandoffState.TASK_CREATED, HandoffState.CLOSED},
            HandoffState.TASK_CREATED: {HandoffState.TASK_RETAINED, HandoffState.CLOSED},
            HandoffState.TASK_RETAINED: {HandoffState.TRANSFERRED, HandoffState.CLOSED},
            HandoffState.TRANSFERRED: set(),
            HandoffState.CLOSED: {HandoffState.CLOSED},
        }
        if state not in allowed[self.state]:
            raise RuntimeError(f"invalid operation handoff transition {self.state.value}->{state.value}")
        self.state = state

    def acquire(self, operation_type, **kwargs):
        if self.closed or self.state is not HandoffState.NEW:
            raise RuntimeError("operation handoff acquire is not available")
        token, snap = self.coordinator.acquire(operation_type, **kwargs)
        self.token = token
        failure = self._register_acquired(token, snap)
        if failure is not None:
            snap = None
            self.clear_heavy()
            _release_token_total(self.coordinator, token)
            token = None
            self.token = None
            self.closed = True
            self.state = HandoffState.CLOSED
            raise RuntimeError(f"{failure[0]}: {failure[1]}") from None
        return token, snap

    def _register_acquired(self, token, snap):
        try:
            self._set_state(HandoffState.ACQUIRED)
            self.token = token
            self.set_heavy(snap=snap)
        except BaseException as exc:
            failure = _bounded_failure(exc, "operation acquisition registration failed")
            exc = None
            snap = None
            self.heavy.clear()
            return failure
        return None

    def set_heavy(self, **refs):
        if self.closed or self.transferred:
            raise RuntimeError("handoff is closed")
        if self.state in (HandoffState.ACQUIRED, HandoffState.PREPARING):
            self._set_state(HandoffState.PREPARING)
        self.heavy.update(refs)

    def clear_heavy(self):
        self.heavy.clear()

    def set_dispatch(self, dispatch):
        if self.closed:
            raise RuntimeError("handoff is closed")
        if self.dispatch is not None:
            raise RuntimeError("operation handoff dispatch already set")
        self._set_state(HandoffState.DISPATCH_READY)
        self.dispatch = dispatch

    def create_dispatch(self, payload=None):
        if self.state not in (HandoffState.ACQUIRED, HandoffState.PREPARING):
            raise RuntimeError("operation handoff dispatch is not available")
        self.set_heavy(dispatch_payload=payload)
        dispatch, failure = self._construct_dispatch(payload)
        payload = None
        if failure is not None:
            self.heavy.pop("dispatch_payload", None)
            raise RuntimeError(f"{failure[0]}: {failure[1]}") from None
        try:
            self.set_dispatch(dispatch)
        except BaseException as exc:
            try:
                dispatch.clear_payload()
            except BaseException:
                pass
            try:
                message = str(exc)[:512]
            except BaseException:
                message = "dispatch registration failed"
            failure = (getattr(type(exc), "__name__", "BaseException")[:128], message)
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            exc = dispatch = None
            self.heavy.pop("dispatch_payload", None)
            raise RuntimeError(f"{failure[0]}: {failure[1]}") from None
        self.heavy.pop("dispatch_payload", None)
        return dispatch

    def _construct_dispatch(self, payload):
        dispatch = failure = None
        try:
            dispatch = WorkerDispatch(self.coordinator, self.token, self.stop_event, payload=payload)
        except BaseException as exc:
            if dispatch is not None:
                try:
                    dispatch.clear_payload()
                except BaseException:
                    pass
            try:
                message = str(exc)[:512]
            except BaseException:
                message = "dispatch construction failed"
            failure = (getattr(type(exc), "__name__", "BaseException")[:128], message)
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            exc = None
        finally:
            payload = None
        return dispatch, failure

    @asynccontextmanager
    async def awaiter_scope(self):
        """Guarantee awaiter ownership is resolved until retained transfer."""
        if self.state not in (HandoffState.ACQUIRED, HandoffState.PREPARING):
            raise RuntimeError("operation handoff scope requires successful acquisition")
        cancelled = False
        try:
            yield self
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
            raise
        finally:
            if not self.transferred and not self.closed:
                cancelled |= await self._close_uninterruptibly()
            if cancelled:
                raise asyncio.CancelledError()

    async def close(self):
        """Resolve awaiter-side ownership; repeated closes are safe."""
        if self.closed or self.transferred:
            return
        self.clear_heavy()
        cancelled = False
        resolved = self.dispatch is None and self.token is None
        try:
            if self.dispatch is not None:
                try:
                    resolved = bool(self.dispatch.cancel_from_awaiter())
                except BaseException:
                    logging.getLogger(__name__).exception("operation dispatch cancellation failed")
                    try:
                        if not self.dispatch.claimed:
                            self.dispatch.clear_payload()
                            resolved = bool(self.coordinator.release(self.token))
                    except BaseException:
                        logging.getLogger(__name__).exception("operation dispatch emergency release failed")
                if self.task is not None:
                    try:
                        await _drain_worker_uninterruptibly(self.task)
                    except asyncio.CancelledError:
                        cancelled = True
                    except BaseException:
                        logging.getLogger(__name__).exception("operation worker drain failed")
                try:
                    resolved = resolved or not self.coordinator.is_current(self.token)
                except BaseException:
                    pass
            elif self.token is not None:
                resolved = bool(self.coordinator.release(self.token))
        finally:
            if resolved:
                self.task = None
                self.wrapper = None
                self.dispatch = None
                self.token = None
                self.closed = True
                self._set_state(HandoffState.CLOSED)
        if cancelled:
            raise asyncio.CancelledError()
        if not resolved:
            raise RuntimeError("operation handoff ownership could not be resolved")

    def set_wrapper(self, wrapper):
        self._set_state(HandoffState.WRAPPER_READY)
        self.wrapper = wrapper

    def set_task(self, task):
        if self.dispatch is None:
            raise RuntimeError("operation handoff task requires dispatch")
        self._set_state(HandoffState.TASK_CREATED)
        self.task = task

    async def create_thread_task(self, worker_factory, *, factory_args=(), factory_kwargs=None):
        primary_error = None
        worker = wrapper = task = None
        retained = False
        try:
            factory = worker_factory
            args = tuple(factory_args)
            kwargs = dict(factory_kwargs or {})
            self.set_heavy(factory=factory, factory_args=args, factory_kwargs=kwargs)
            worker = factory(*args, **kwargs)
            self.set_heavy(worker=worker)
            wrapper = asyncio.to_thread(worker)
            self.set_wrapper(wrapper)
            task = asyncio.create_task(wrapper)
            wrapper = None
            self.wrapper = None
            _deferred_worker_tasks.add(task)
            retained = True
            self.set_task(task)
            _retain_worker_task(task)
            self._set_state(HandoffState.TASK_RETAINED)
            self._set_state(HandoffState.TRANSFERRED)
            self.transferred = True
            self.clear_heavy()
            factory = args = kwargs = worker = wrapper = None
            return task
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            try:
                message = str(exc)[:512]
            except BaseException:
                message = "worker task construction failed"
            failure = (getattr(type(exc), "__name__", "BaseException")[:128], message)
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            primary_error = exc = None
            self.wrapper = None
            wrapper_to_close = wrapper if task is None else None
            if wrapper_to_close is not None and hasattr(wrapper_to_close, "close"):
                try:
                    wrapper_to_close.close()
                except BaseException:
                    logging.getLogger(__name__).exception("failed to close unsubmitted worker wrapper")
            self.clear_heavy()
            if retained or task is not None:
                _deferred_worker_tasks.discard(task)
            task_to_drain = task or self.task
            if task_to_drain is not None:
                try:
                    if not task_to_drain.done():
                        task_to_drain.cancel()
                except BaseException:
                    pass
                while not task_to_drain.done():
                    try:
                        await _drain_worker_uninterruptibly(task_to_drain)
                    except asyncio.CancelledError:
                        cancelled = True
                        current = asyncio.current_task()
                        if current is not None and hasattr(current, "uncancel"):
                            current.uncancel()
                    except BaseException:
                        cancelled |= await _cleanup_yield()
            factory = args = kwargs = worker = wrapper = task = task_to_drain = None
            if not self.closed and not self.transferred:
                cancelled |= await self._close_uninterruptibly()
            if cancelled:
                raise asyncio.CancelledError() from None
            raise RuntimeError(f"{failure[0]}: {failure[1]}") from None


def _return_worker(worker):
    return worker


def _retain_worker_task(task):
    def _done(done_task):
        try:
            if not done_task.cancelled():
                done_task.exception()
        except Exception:
            logging.getLogger(__name__).exception("background model worker failed after awaiter cancellation")
        finally:
            _deferred_worker_tasks.discard(done_task)

    added = False
    try:
        _deferred_worker_tasks.add(task)
        added = True
        task.add_done_callback(_done)
    except BaseException:
        if added:
            _deferred_worker_tasks.discard(task)
        raise




async def _drain_worker_uninterruptibly(task):
    async def _drain():
        return await asyncio.gather(asyncio.shield(task), return_exceptions=True)
    drain_coro = drain = None
    cancelled = False
    try:
        try:
            drain_coro = _drain()
            drain = asyncio.create_task(drain_coro)
            drain_coro = None
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                cancelled = True
                _remember_task_cancellation()
            if drain_coro is not None and hasattr(drain_coro, "close"):
                drain_coro.close()
            drain_coro = None
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.done():
                        cancelled = True
                        _remember_task_cancellation()
                except BaseException:
                    if not task.done():
                        cancelled |= await _cleanup_yield()
            try:
                result = [task.result()]
            except BaseException as task_exc:
                result = [task_exc]
            if cancelled:
                raise asyncio.CancelledError() from None
            return result
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                if not drain.done():
                    cancelled = True
                    _remember_task_cancellation()
            except BaseException:
                if not drain.done():
                    cancelled |= await _cleanup_yield()
        result = drain.result()
    finally:
        if drain_coro is not None and hasattr(drain_coro, "close"):
            drain_coro.close()
        if drain is not None and not drain.done():
            drain.cancel()
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    if not drain.done():
                        cancelled = True
                        _remember_task_cancellation()
                except BaseException:
                    if not drain.done():
                        cancelled |= await _cleanup_yield()
    if cancelled:
        raise asyncio.CancelledError() from None
    return result


async def _await_transferred_worker(task, dispatch, *, on_cancel=None):
    """Await a retained worker and resolve exact ownership before cancellation."""
    cancelled = False
    result = None
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled = True
        _remember_task_cancellation()
    if not cancelled:
        return result
    result = None
    if on_cancel is not None:
        try:
            on_cancel()
        except BaseException as exc:
            _bounded_failure(exc, "publication cancellation failed")
    try:
        dispatch.cancel_from_awaiter()
    except BaseException as exc:
        _bounded_failure(exc, "dispatch cancellation failed")
    while not task.done():
        try:
            await _drain_worker_uninterruptibly(task)
        except asyncio.CancelledError:
            _remember_task_cancellation()
        except BaseException:
            await _cleanup_yield()
    while True:
        try:
            current = dispatch.coordinator.is_current(dispatch.token)
        except BaseException:
            current = True
        if not current:
            break
        if dispatch.claimed:
            _release_worker_total(dispatch)
        else:
            _abort_unclaimed_total(dispatch)
        await _cleanup_yield()
    try:
        if not task.cancelled():
            task.exception()
    except BaseException as exc:
        _bounded_failure(exc, "worker completion failed")
    task = dispatch = on_cancel = None
    raise asyncio.CancelledError() from None


def _valid_devices():
    """Accepted devices = "auto" + one cuda:N per GPU actually present.
    Adaptive: no longer assumes the personal 2-GPU (cuda:0/cuda:1) setup."""
    try:
        n = len(gpu_stats())
    except Exception:
        n = 0
    return {"auto"} | {f"cuda:{i}" for i in range(n)}


class _QuietPolling(logging.Filter):
    """Drops the access lines from the UI polling (GET /api/status every 2 s)."""

    def filter(self, record):
        return "GET /api/status " not in record.getMessage()


@asynccontextmanager
async def lifespan(_app):
    loop = asyncio.get_running_loop()
    access_logger = logging.getLogger("uvicorn.access")
    quiet_filter = _QuietPolling()
    _loop_holder["loop"] = loop
    access_logger.addFilter(quiet_filter)
    try:
        try:
            inspection = await asyncio.to_thread(editing.inspect_abandoned_export_temps)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception(
                "abandoned export temporary inspection failed; startup will continue"
            )
        else:
            logging.getLogger(__name__).info(
                "export temporary inspection: abandoned=%d active=%d recent=%d completed=%d unsafe=%d errors=%d",
                len(inspection["abandoned"]), len(inspection["active"]),
                len(inspection["recent"]), len(inspection["completed"]),
                len(inspection["changed_or_unsafe"]), len(inspection["errors"]),
            )
            for failure in inspection["errors"]:
                logging.getLogger(__name__).warning(
                    "could not safely inspect export temporary artifact %s: %s",
                    failure["path"], failure["error"],
                )
        yield
    finally:
        access_logger.removeFilter(quiet_filter)
        if _loop_holder.get("loop") is loop:
            _loop_holder.pop("loop", None)


app = FastAPI(title="J-Wash", lifespan=lifespan)


async def _ws_send(ws, text):
    lock = _ws_locks.get(ws)
    if lock is None:
        return
    async with lock:
        await ws.send_text(text)


def _broadcast_fit(state):
    loop = _loop_holder.get("loop")
    if loop is None:
        return
    payload = json.dumps({"type": "fit_progress", "fit": state})
    for ws in list(_ws_locks):
        asyncio.run_coroutine_threadsafe(_ws_send(ws, payload), loop)


fit_manager.on_progress = _broadcast_fit


def _broadcast_capabilities_changed(model_session_id):
    """Best-effort invalidation only; status remains the authoritative payload."""
    loop = _loop_holder.get("loop")
    if loop is None:
        return
    payload = json.dumps({"type": "capabilities_changed", "model_session_id": model_session_id})
    for ws in list(_ws_locks):
        notification = _ws_send(ws, payload)
        try:
            future = asyncio.run_coroutine_threadsafe(notification, loop)
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        except Exception:
            notification.close()
            logging.getLogger(__name__).debug("capability notification failed", exc_info=True)

# concurrent HF downloads: one state per repo_id
_downloads = {}
_downloads_lock = threading.Lock()
# last synchronous /api/generate exchange (CLI/MCP): surfaced in /api/status so
# the UI can show what an external client generates, without persisting it
_generation_counter = 0


class LoadRequest(BaseModel):
    model_id: str
    dtype: str = config.DEFAULT_DTYPE
    quant: str | None = None
    device: str = config.DEFAULT_DEVICE


class DownloadRequest(BaseModel):
    repo_id: str


class DeleteModelRequest(BaseModel):
    model_id: str


class LensLoadRequest(BaseModel):
    repo_id: str | None = None
    filename: str = "lens.pt"
    revision: str | None = None
    path: str | None = None
    layers: list[int] | None = None
    k: int = 8


class LensLayersRequest(BaseModel):
    layers: list[int]
    k: int | None = None


class PinRequest(BaseModel):
    gen_id: int
    generation_run_id: str | None = None
    token_ids: list[int]


class ConversationPatch(BaseModel):
    title: str | None = None
    tags: list[str] | None = None


class InterventionRequest(BaseModel):
    token_id: int
    mode: str = "scale"
    factor: float = 0.0
    replacement_id: int | None = None
    layers: list[int] | None = None


_NAME_RE = re.compile(r"[\w][\w.\- ]*", re.UNICODE)


def _safe_name(name):
    """Validate a user-supplied file/folder name (presets, exports): plain
    names only — no separators, no traversal."""
    name = (name or "").strip()
    if not name or ".." in name or not _NAME_RE.fullmatch(name):
        raise HTTPException(
            422, f"invalid name {name!r}: letters, digits, spaces, . - _ only"
        )
    return name


class InterventionPatch(BaseModel):
    factor: float | None = None
    layers: list[int] | None = None
    enabled: bool | None = None
    token_id: int | None = None
    replacement_id: int | None = None
    mode: str | None = None  # scale | replace


class InterventionsScale(BaseModel):
    scale: float | None = None
    mode: str | None = None


class ExportRequest(BaseModel):
    format: str = "layers"
    name: str


class FitRequest(BaseModel):
    model_id: str
    dtype: str = "bf16"
    quant: str | None = None
    n_prompts: int = 100
    datasets: list[str] = [fitting.DATASET_WIKITEXT]  # any HF ids; several = equal-parts mix
    devices: list[str] = ["cuda:0"]
    name: str | None = None
    dim_batch: int | None = None
    max_seq_len: int = 128
    source_layers: list[int] | None = None
    continue_from: str | None = None


@app.get("/api/models")
def api_models():
    return {"models": manager.list_models()}


@app.get("/api/status")
def api_status():
    coord = manager.coordinator.status_snapshot()
    iv = coord.interventions or {}
    scale = iv.get("scale", 1.0)
    mode = iv.get("mode", "standard")
    loaded = coord.loaded
    profile = coord.capability_profile
    meta = dict(coord.model_meta or {})
    capability_snapshot = capabilities.snapshot(profile, mode, loaded=loaded)
    loaded_meta = ({**meta, **capabilities.legacy(profile, loaded=True)} if loaded else None)
    return {
        "loaded": loaded_meta,
        "capabilities": capability_snapshot,
        "busy": None if coord.operation is None else coord.operation.type,
        "model_session_id": coord.model_session_id,
        "model_state": coord.model_state,
        "operation": None if coord.operation is None else {
            "id": coord.operation.id,
            "type": coord.operation.type,
            "acquired_session": coord.operation.acquired_session,
            "current_session": coord.operation.current_session,
            "start_timestamp": coord.operation.start_timestamp,
            "cancellation_requested": coord.operation.cancellation_requested,
        },
        "lens": dict(coord.lens) if hasattr(coord.lens, "items") else coord.lens,
        "gpus": gpu_stats(),
        "downloads": list(_downloads.values()),
        "convert": _convert_state,
        "fit": fit_manager.state,
        "gguf": _gguf_state_snapshot(),
        "interventions": iv.get("summary", []),
        "interventions_scale": scale,
        "interventions_mode": mode,
        "last_generation": coord.last_generation,
    }


def _conflict(exc):
    return HTTPException(409, str(exc))


def _publish_intervention_snapshot(token, snap):
    data = interventions.snapshot()
    data["model_session_id"] = snap.model_session_id
    lens = manager.coordinator.status_snapshot().lens
    data["lens_binding_id"] = (lens or {}).get("lens_binding_id") if hasattr(lens or {}, "get") else None
    manager.coordinator.update_interventions(token, data)
    return data


def _run_intervention_transaction(token, snap, mutate):
    previous = interventions.state_record()
    try:
        result = mutate()
        published = _publish_intervention_snapshot(token, snap)
        return result, published
    except Exception:
        interventions.restore_state_record(previous)
        raise

@app.post("/api/load")
async def api_load(req: LoadRequest):
    if req.dtype not in config.DTYPES:
        raise HTTPException(422, f"invalid dtype: {req.dtype}")
    if req.quant is not None and req.quant not in config.QUANTS:
        raise HTTPException(422, f"invalid quant: {req.quant}")
    if req.device not in _valid_devices():
        raise HTTPException(422, f"invalid device: {req.device}")
    before_session = manager.coordinator.status_snapshot().model_session_id
    try:
        result = await asyncio.to_thread(
            manager.load, req.model_id, req.dtype, req.quant, req.device
        )
    except OperationConflict as exc:
        raise _conflict(exc)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        after_session = manager.coordinator.status_snapshot().model_session_id
        if after_session != before_session:
            _broadcast_capabilities_changed(after_session)
    return result


def _make_delete_model_worker(dispatch, model_id, delete_fn):
    def worker():
        return _execute_dispatched_worker(
            dispatch, lambda _payload: delete_fn(model_id),
            default_status=500, value_status=400,
        )
    return worker


async def _prepare_delete_model_handoff(handoff, model_id, stop_event, delete_fn):
    snap = status = payload = dispatch = task = None
    try:
        snap = handoff.snapshot
        status = manager.coordinator.status_snapshot()
        loaded_model_id = (status.model_meta or {}).get("model_id") if status.model_meta is not None else None
        if loaded_model_id == model_id:
            return HandoffSetup(status=409, kind="ModelStateError", message="unload this model before deleting it")
        dispatch = handoff.create_dispatch()
        snap = status = payload = None
        task = await handoff.create_thread_task(
            _make_delete_model_worker, factory_args=(dispatch, str(model_id), delete_fn),
        )
        return HandoffSetup(task=task, dispatch=dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "model deletion setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        snap = status = payload = dispatch = task = None


@app.post("/api/models/delete")
async def api_delete_model(req: DeleteModelRequest):
    from core.model_manager import delete_model
    stop_event = ThreadingEvent()
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.MODEL_DELETE,
            stop_event=stop_event, include_bundle=False,
        ) as handoff:
            setup = await _prepare_delete_model_handoff(
                handoff, str(req.model_id), stop_event, delete_model,
            )
    except OperationConflict as exc:
        raise _conflict(exc)
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "model deletion setup failed") from None
    task, dispatch = setup.task, setup.dispatch
    return _raise_worker_outcome(await _await_transferred_worker(task, dispatch))



# --- user settings (Options tab) --------------------------------------------
SETTINGS_PATH = config.DATA_DIR / "settings.json"
SETTINGS_DEFAULTS = {
    "default_quant": "",       # '', 'int8' or 'nf4' — preselected in the Model tab
    "auto_layer_radius": 2,    # editor: peak ± radius when auto-selecting layers
    "chat_markdown": True,     # render assistant replies as markdown
    "hf_cache": "",            # HF cache dir — applied at startup (--hf-cache wins)
    "llamacpp_dir": "",        # llama.cpp folder → enables the direct GGUF export
}


def read_settings():
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return {**SETTINGS_DEFAULTS, **{k: v for k, v in data.items() if k in SETTINGS_DEFAULTS}}


class SettingsPatch(BaseModel):
    default_quant: str | None = None
    auto_layer_radius: int | None = None
    chat_markdown: bool | None = None
    hf_cache: str | None = None
    llamacpp_dir: str | None = None


@app.get("/api/settings")
def api_settings():
    return read_settings()


@app.patch("/api/settings")
def api_settings_patch(req: SettingsPatch):
    if req.default_quant is not None and req.default_quant not in ("", "int8", "nf4"):
        raise HTTPException(422, f"invalid quant: {req.default_quant}")
    current = read_settings()
    for key, value in req.model_dump(exclude_none=True).items():
        if key == "auto_layer_radius":
            value = max(0, min(8, int(value)))
        current[key] = value
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return current


class RegisterModelRequest(BaseModel):
    path: str


@app.post("/api/models/register")
def api_models_register(req: RegisterModelRequest):
    """Add a model folder to the available list (no copy — just remembered)."""
    from core.model_manager import register_model_dir
    try:
        return register_model_dir(req.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/models/unregister")
def api_models_unregister(req: RegisterModelRequest):
    """Forget a registered entry; the model files are left untouched."""
    from core.model_manager import unregister_model_dir
    try:
        return unregister_model_dir(req.path)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/unload")
async def api_unload():
    before_session = manager.coordinator.status_snapshot().model_session_id
    try:
        result = await asyncio.to_thread(manager.unload)
    except OperationConflict as exc:
        raise _conflict(exc)
    after_session = manager.coordinator.status_snapshot().model_session_id
    if after_session != before_session:
        _broadcast_capabilities_changed(after_session)
    return result


def _make_lens_load_worker(dispatch, token):
    def worker():
        old_state = None
        def execute(payload):
            nonlocal old_state
            old_state = lens_manager.state_record()
            result = lens_manager.load(
                manager, repo_id=payload["repo_id"], filename=payload["filename"],
                revision=payload["revision"], path=payload["path"],
                layers=payload["layers"], k=payload["k"],
            )
            manager.coordinator.update_lens_binding(token, result)
            return result
        def rollback():
            if old_state is not None:
                lens_manager.restore_state_record(old_state)
        def teardown():
            nonlocal old_state
            old_state = None
        try:
            return _execute_dispatched_worker(
                dispatch, execute, rollback=rollback, teardown=teardown,
            )
        finally:
            old_state = execute = rollback = teardown = None
    return worker


async def _prepare_lens_load_handoff(handoff, request_values):
    snap = payload = dispatch = task = None
    try:
        snap = handoff.snapshot
        payload = dict(request_values)
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = None
        snap = None
        task = await handoff.create_thread_task(
            _make_lens_load_worker, factory_args=(dispatch, handoff.token),
        )
        return HandoffSetup(task=task, dispatch=dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "lens setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        snap = payload = dispatch = task = None


@app.post("/api/lens/load")
async def api_lens_load(req: LensLoadRequest):
    if not req.repo_id and not req.path:
        raise HTTPException(422, "repo_id or path required")
    stop_event = ThreadingEvent()
    request_values = {
        "repo_id": req.repo_id, "filename": req.filename, "revision": req.revision,
        "path": req.path, "layers": req.layers, "k": req.k,
    }
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.LENS_UPDATE, stop_event=stop_event,
            requires_loaded=True, include_bundle=False,
        ) as handoff:
            setup = await _prepare_lens_load_handoff(handoff, request_values)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "lens setup failed") from None
    task, dispatch = setup.task, setup.dispatch
    return _raise_worker_outcome(await _await_transferred_worker(task, dispatch))



@app.post("/api/lens/unload")
def api_lens_unload():
    token = None
    old = None
    try:
        token, _snap = manager.coordinator.acquire(OperationType.LENS_UPDATE, include_bundle=False)
        old = lens_manager.withdraw()
        try:
            manager.coordinator.clear_lens_binding(token)
        except Exception:
            lens_manager.restore_withdrawn(old)
            raise
        cleanup_error = lens_manager.cleanup_withdrawn(old)
        old = None
        if cleanup_error is not None:
            logging.getLogger(__name__).warning(
                "lens cleanup failed after authoritative unload",
                exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
            )
        return {"unloaded": True}
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        if token is not None:
            manager.coordinator.release(token)


@app.post("/api/lens/layers")
def api_lens_layers(req: LensLayersRequest):
    token = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.LENS_UPDATE, requires_loaded=True, include_bundle=False)
        outcome = _lens_layers_resource(token, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _lens_layers_resource(token, req):
    old_state = result = None
    try:
        old_state = lens_manager.state_record()
        result = lens_manager.set_layers(manager, req.layers, k=req.k)
        try:
            manager.coordinator.update_lens_binding(token, result)
        except Exception:
            lens_manager.restore_state_record(old_state)
            raise
        return _route_value(dict(result))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return _route_http(exc.status_code, exc.detail)
        return _route_exception(500, exc)
    finally:
        old_state = result = None


@app.get("/api/interventions")
def api_interventions():
    snap = manager.coordinator.status_snapshot()
    iv = snap.interventions or {}
    return {"rules": list(iv.get("summary") or [])}


@app.post("/api/interventions")
def api_interventions_add(req: InterventionRequest):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
        outcome = _interventions_add_resource(token, snap, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_add_resource(token, snap, req):
    bundle = rules = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        if lens_manager.lens is None:
            return _route_http(422, "model and lens required")
        def mutate():
            return interventions.add(
                lens_manager,
                bundle.jl,
                token_id=req.token_id,
                mode=req.mode,
                factor=req.factor,
                replacement_id=req.replacement_id,
                layers=req.layers,
            )
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = rules = mutate = None


@app.patch("/api/interventions/{rule_id}")
def api_interventions_patch(rule_id: int, req: InterventionPatch):
    needs_dirs = any(
        x is not None for x in (req.layers, req.token_id, req.replacement_id, req.mode)
    )
    effectful_patch = needs_dirs or req.factor is not None or req.enabled is True
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(
            OperationType.INTERVENTION_UPDATE, requires_loaded=True if effectful_patch else None
        )
        outcome = _interventions_patch_resource(
            token, snap, rule_id, req, needs_dirs, effectful_patch
        )
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_patch_resource(token, snap, rule_id, req, needs_dirs, effectful_patch=None):
    bundle = rules = None
    if effectful_patch is None:
        effectful_patch = needs_dirs
    if effectful_patch:
        bundle = _bundle_from_snapshot_or_legacy(snap)
    try:
        def mutate():
            return interventions.update(
                rule_id,
                factor=req.factor,
                layers=req.layers,
                enabled=req.enabled,
                token_id=req.token_id,
                replacement_id=req.replacement_id,
                mode=req.mode,
                lens_manager=lens_manager if needs_dirs else None,
                jl=bundle.jl if needs_dirs else None,
            )
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except UnknownInterventionRule as exc:
        return _route_http(404, str(exc))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = rules = mutate = None


@app.patch("/api/interventions")
def api_interventions_scale(req: InterventionsScale):
    token = snap = None
    prior_mode = "standard"
    resulting_session_id = None
    try:
        token, snap = manager.coordinator.acquire(
            OperationType.INTERVENTION_UPDATE,
            requires_loaded=True if req.mode is not None else None,
        )
        prior_mode = (_coordinated_interventions(snap) or {}).get("mode", "standard")
        resulting_session_id = snap.model_session_id
        outcome = _interventions_scale_resource(token, snap, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    result = _raise_route_outcome(outcome)
    if req.mode is not None and result.get("mode") != prior_mode:
        _broadcast_capabilities_changed(resulting_session_id)
    return result


def _interventions_scale_resource(token, snap, req):
    try:
        def mutate():
            return interventions.set_scale_and_mode(scale=req.scale, mode=req.mode)
        (scale, mode), _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"scale": scale, "mode": mode})
    except ValueError as exc:
        return _route_http(422, str(exc))
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        mutate = None


@app.delete("/api/interventions/{rule_id}")
def api_interventions_remove(rule_id: int):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
        outcome = _interventions_remove_resource(token, snap, rule_id)
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_remove_resource(token, snap, rule_id):
    rules = None
    try:
        def mutate():
            return interventions.remove(rule_id)
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        rules = mutate = None


@app.delete("/api/interventions")
def api_interventions_clear():
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
        outcome = _interventions_clear_resource(token, snap)
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_clear_resource(token, snap):
    rules = None
    try:
        def mutate():
            return interventions.remove()
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        rules = mutate = None


def _stable_lens_descriptor(lens_binding):
    if lens_binding is None:
        return {
            "schema_version": 1,
            "source_kind": "none",
            "repo_id": None,
            "path": None,
            "filename": None,
            "revision": None,
            "model_id": None,
            "model_revision": None,
            "fitted_layers": [],
            "tapped_layers": [],
            "k": None,
            "runtime_lens_binding_id": None,
            "runtime_model_session_id": None,
        }
    lens = dict(lens_binding)
    path = lens.get("path")
    repo_id = lens.get("repo_id")
    source_kind = "local" if path else ("hub" if repo_id else "none")

    def int_list(value):
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return []
        try:
            return [int(v) for v in value]
        except TypeError:
            return []

    fitted = int_list(lens.get("fitted_layers_all"))
    if not fitted:
        fitted = int_list(lens.get("fitted_layers"))
    return {
        "schema_version": 1,
        "source_kind": source_kind,
        "repo_id": repo_id,
        "path": os.path.normpath(str(path)) if path else None,
        "filename": lens.get("filename") or ("lens.pt" if repo_id else None),
        "revision": lens.get("revision"),
        "model_id": lens.get("model_id"),
        "model_revision": lens.get("model_revision"),
        "fitted_layers": fitted,
        "tapped_layers": int_list(lens.get("tapped_layers")),
        "k": int(lens["k"]) if lens.get("k") is not None else None,
        "runtime_lens_binding_id": lens.get("lens_binding_id"),
        "runtime_model_session_id": lens.get("model_session_id"),
    }


def _lens_descriptor_mismatch(saved, current):
    if saved is None:
        return False
    current = current or _stable_lens_descriptor(None)
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        return True
    if not isinstance(current, dict) or current.get("schema_version") != 1:
        return True
    stable_keys = (
        "source_kind", "repo_id", "path", "filename", "revision",
        "model_id", "model_revision", "fitted_layers", "tapped_layers", "k",
    )
    return any(saved.get(key) != current.get(key) for key in stable_keys)


def _lens_descriptor_is_legacy(value):
    return value is not None and (not isinstance(value, dict) or value.get("schema_version") != 1)


@app.get("/api/presets")
def api_presets():
    return {"presets": editing.list_presets()}


@app.post("/api/presets/{name}")
def api_presets_save(name: str):
    name = _safe_name(name)
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True, include_bundle=True)
        bundle = snap.bundle
        iv = _coordinated_interventions(snap)
        rules = [dict(rule) for rule in (iv.get("summary") or [])]
        if not rules:
            raise HTTPException(422, "no active intervention to save")
        model_id = bundle.meta.get("model_id") if bundle is not None else None
        scale = iv.get("scale")
        lens_descriptor = _stable_lens_descriptor(snap.lens_binding)
        result = editing.save_preset(
            name, rules, model_id, scale=scale,
            model_revision=bundle.meta.get("revision") if bundle is not None else None,
            intervention_mode=iv.get("mode"),
            intervention_scale=scale,
            intervention_revision=iv.get("revision"),
            model_session_id=iv.get("model_session_id"),
            lens_binding_id=iv.get("lens_binding_id"),
            lens_descriptor=lens_descriptor,
        )
        return result
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = bundle = iv = rules = result = None
        if token is not None:
            manager.coordinator.release(token)


@app.post("/api/presets/{name}/apply")
def api_presets_apply(name: str):
    name = _safe_name(name)
    try:
        preset = editing.load_preset(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
        outcome = _presets_apply_resource(token, snap, preset)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _presets_apply_resource(token, snap, preset):
    bundle = current_iv = rules = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        if lens_manager.lens is None:
            return _route_http(422, "model and lens required")
        warnings = []
        if preset.get("model_id") and preset["model_id"] != bundle.meta["model_id"]:
            warnings.append(
                f"preset saved for {preset['model_id']}, loaded model: {bundle.meta['model_id']}"
            )
        if preset.get("model_revision") and preset.get("model_revision") != bundle.meta.get("revision"):
            warnings.append(
                f"preset saved for revision {preset['model_revision']}, loaded revision: {bundle.meta.get('revision')}"
            )
        current_iv = _coordinated_interventions(snap)
        if preset.get("intervention_mode") and preset.get("intervention_mode") != current_iv.get("mode"):
            warnings.append(
                f"preset saved under intervention mode {preset['intervention_mode']}; current mode: {current_iv.get('mode')}"
            )
        current_lens_descriptor = _stable_lens_descriptor(snap.lens_binding)
        if preset.get("lens_descriptor") is not None:
            if _lens_descriptor_is_legacy(preset.get("lens_descriptor")):
                warnings.append("preset has sparse lens provenance; stable lens compatibility cannot be verified")
            elif _lens_descriptor_mismatch(preset.get("lens_descriptor"), current_lens_descriptor):
                warnings.append("preset lens descriptor differs from the current coordinated lens")
        elif preset.get("lens_binding_id") is not None:
            warnings.append("preset has only process-local lens provenance; current stable lens identity is unknown")
        if preset.get("schema_version") is None:
            warnings.append("legacy preset: provenance is unknown")
        def mutate():
            rules = None
            for rule in preset.get("rules", []):
                try:
                    rules = interventions.add(
                        lens_manager,
                        bundle.jl,
                        token_id=rule["token_id"],
                        mode=rule["mode"],
                        factor=rule["factor"],
                        replacement_id=rule.get("replacement_id"),
                        layers=rule.get("layers"),
                        enabled=rule.get("enabled", True),
                    )
                except ValueError as exc:
                    warnings.append(f"rule {rule.get('token')!r} skipped: {exc}")
            if preset.get("intervention_scale", preset.get("scale")) is not None:
                interventions.set_scale(preset.get("intervention_scale", preset.get("scale")))
            return rules
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({
            "rules": rules or interventions.summary(),
            "scale": interventions.global_scale,
            "warnings": warnings,
        })
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = current_iv = rules = mutate = warnings = None


@app.delete("/api/presets/{name}")
def api_presets_delete(name: str):
    editing.delete_preset(_safe_name(name))
    return {"ok": True}


def _make_export_worker(dispatch):
    def worker():
        def execute(payload):
            call_kwargs = dict(payload["kwargs"])
            call_kwargs["publication_guard"] = payload["publication_gate"]
            return payload["export_fn"](
                payload["rules"], payload["bundle"].jl, payload["meta"],
                fmt=payload["format"], name=payload["name"],
                source_dir=payload["source_dir"], scale=payload["scale"], **call_kwargs,
            )
        return _execute_dispatched_worker(dispatch, execute)
    return worker


async def _prepare_export_handoff(handoff, req):
    token = snap = outcome = prepared = payload_seed = publication_gate = payload = dispatch = task = None
    try:
        token, snap = handoff.token, handoff.snapshot
        outcome = _export_preflight_resource(snap, req)
        snap = None
        if outcome.status is not None:
            return HandoffSetup(status=outcome.status, kind="ExportPreflight", message=outcome.message)
        prepared = outcome.value
        payload_seed = prepared.take()
        prepared = outcome = None
        publication_gate = editing.PublicationGate(
            lambda coordinator=handoff.coordinator, owner=token: coordinator.can_publish(owner)
        )
        payload = dict(payload_seed)
        payload.update(publication_gate=publication_gate, format=req.format, name=req.name)
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = payload_seed = None
        token = prepared = outcome = None
        task = await handoff.create_thread_task(_make_export_worker, factory_args=(dispatch,))
        return HandoffSetup(task=task, dispatch=dispatch, gate=publication_gate)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "export setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        token = snap = outcome = prepared = payload_seed = payload = dispatch = task = publication_gate = None


@app.post("/api/edit/export")
async def api_edit_export(req: ExportRequest):
    req.name = _safe_name(req.name)
    if req.format not in ("layers", "lora", "full"):
        raise HTTPException(422, f"unknown format: {req.format}")
    stop_event = ThreadingEvent()
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.EXPORT,
            stop_event=stop_event, requires_loaded=True,
        ) as handoff:
            setup = await _prepare_export_handoff(handoff, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "export setup failed") from None
    task, dispatch, publication_gate = setup.task, setup.dispatch, setup.gate
    return _raise_worker_outcome(await _await_transferred_worker(
        task, dispatch, on_cancel=publication_gate.cancel,
    ))



def _export_preflight_resource(snap, req):
    bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None
    try:
        bundle = snap.bundle
        meta = dict(bundle.meta)
        intervention_snapshot = _coordinated_interventions(snap)
        rules = intervention_snapshot["active_rules"]
        if not rules:
            return _route_http(422, "no active intervention to export (rules disabled or without layers?)")
        source_dir = resolve_local_dir(meta["model_id"], revision=meta.get("revision"))
        scale, mode = intervention_snapshot["scale"], intervention_snapshot["mode"]
        kwargs = {}
        if mode in ("readthrough", "exact"):
            export_fn = editing.export_rebase
            kwargs["exact"] = mode == "exact"
        elif mode == "abliteration":
            export_fn = editing.export_abliteration
        else:
            return _route_http(
                422,
                "export requires a pure-weights mode: switch to \"read projection\" "
                "(or \"global projection\" on write-norm architectures) — per-layer "
                "steering does not bake faithfully",
            )
        return _route_value(PreparedWorkerPayload({
            "bundle": bundle, "meta": meta, "rules": rules, "source_dir": source_dir,
            "scale": scale, "kwargs": kwargs, "export_fn": export_fn,
        }))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None




def _gguf_preflight_resource(snap):
    bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        meta = dict(bundle.meta)
        intervention_snapshot = _coordinated_interventions(snap)
        rules = intervention_snapshot["active_rules"]
        if not rules:
            return _route_http(422, "no active intervention to export")
        scale, mode = intervention_snapshot["scale"], intervention_snapshot["mode"]
        if mode in ("readthrough", "exact"):
            export_fn, kwargs = editing.export_rebase, {"exact": mode == "exact"}
        elif mode == "abliteration":
            export_fn, kwargs = editing.export_abliteration, {}
        else:
            return _route_http(422, "export requires a pure-weights mode")
        source_dir = resolve_local_dir(meta["model_id"], revision=meta.get("revision"))
        return _route_value(PreparedWorkerPayload({
            "bundle": bundle, "meta": meta, "rules": rules, "source_dir": source_dir,
            "scale": scale, "kwargs": kwargs, "export_fn": export_fn,
        }))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None


# --- direct GGUF export (via a user-provided llama.cpp folder) ---------------
# Two stages: (1) bake the full HF checkpoint into data/edits/<name>/hf — kept
# as a CACHE so several GGUF types can be exported without re-baking — then
# (2) convert_hf_to_gguf.py (+ llama-quantize for quantized types) in a
# background thread, progress polled through /api/status.
_gguf_state = {"state": "idle", "name": None, "step": None, "error": None, "result": None}
_gguf_lock = threading.Lock()

@dataclass(frozen=True)
class _GGUFJobToken:
    id: str
    name: str

_gguf_owner = None
_gguf_delete_claim = None

def _gguf_state_snapshot():
    with _gguf_lock:
        return dict(_gguf_state)

def _reserve_gguf_job(name):
    global _gguf_owner
    token = _GGUFJobToken(uuid.uuid4().hex, name)
    running_state = {
        "state": "running", "name": name, "step": "checking checkpoint",
        "error": None, "result": None,
    }
    with _gguf_lock:
        if _gguf_owner is not None or _gguf_delete_claim is not None:
            raise OperationConflict("a GGUF export or cache deletion is already in progress")
        _gguf_owner = token
        _gguf_state.clear()
        _gguf_state.update(running_state)
    return token

def _gguf_job_is_current(token):
    with _gguf_lock:
        return _gguf_owner == token

def _gguf_job_progress(token, step):
    with _gguf_lock:
        if _gguf_owner != token:
            return False
        _gguf_state["step"] = step
        return True

def _finish_gguf_job(token, *, result=None, error=None):
    global _gguf_owner
    with _gguf_lock:
        if _gguf_owner != token:
            return False
        _gguf_state.update(state="done" if error is None else "error", step=None,
                           error=error, result=result if error is None else None)
        _gguf_owner = None
        return True

def _validate_gguf_artifact(path):
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"GGUF producer succeeded without producing an output: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"GGUF artifact is inaccessible: {path}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeError(f"GGUF artifact must be a regular nonempty file: {path}")
    if details.st_size <= 0:
        raise RuntimeError(f"GGUF producer produced an empty output: {path}")
    return details

def _publish_gguf_temp(token, temp_path, final_path):
    _validate_gguf_artifact(temp_path)
    with _gguf_lock:
        if _gguf_owner != token:
            return False
        temp_path.replace(final_path)
        return True

GGUF_BASE_TYPES = ("bf16", "f16")
GGUF_QUANT_TYPES = ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m")


class GGUFExportRequest(BaseModel):
    name: str
    gguf_type: str = "q4_k_m"


def _llamacpp_paths():
    """(convert_py, quantize_exe, gguf_py) from the configured llama.cpp dir."""
    root = read_settings().get("llamacpp_dir") or ""
    root = Path(root).expanduser() if root else None
    if not root or not root.is_dir():
        raise ValueError(
            "llama.cpp folder not set — configure it in the Options tab to "
            "enable the direct GGUF export"
        )
    convert = root / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise ValueError(f"convert_hf_to_gguf.py not found in {root}")
    quantize = None
    for cand in ("llama-quantize", "llama-quantize.exe"):
        for sub in (".", "bin", "build/bin"):
            p = root / sub / cand
            if p.exists():
                quantize = p
                break
        if quantize:
            break
    gguf_py = root / "gguf-py"
    return convert, quantize, (gguf_py if gguf_py.is_dir() else None)


def _gguf_worker(token, job_name, job_dir, filename_stem, gguf_type,
                 hf_dir, convert, quantize, gguf_py):
    import subprocess
    import sys
    if not _gguf_job_is_current(token):
        return
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        if not _gguf_job_is_current(token):
            return
        base_type = gguf_type if gguf_type in GGUF_BASE_TYPES else "bf16"
        base_gguf = job_dir / f"{filename_stem}-{base_type}.gguf"
        env = dict(os.environ)
        if gguf_py is not None:  # vendored gguf package inside the llama.cpp repo
            env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
        if base_gguf.exists() or base_gguf.is_symlink():
            if not _gguf_job_is_current(token):
                return
            _validate_gguf_artifact(base_gguf)
        else:
            if not _gguf_job_progress(token, f"converting to {base_type}"):
                return
            temp_gguf = base_gguf.with_name(
                f".{base_gguf.stem}.tmp-{uuid.uuid4().hex}.gguf"
            )
            temp_gguf.unlink(missing_ok=True)
            with editing.artifact_lease(temp_gguf):
                try:
                    proc = subprocess.run(
                        [sys.executable, "-X", "utf8", str(convert), str(hf_dir),
                         "--outfile", str(temp_gguf), "--outtype", base_type],
                        capture_output=True, text=True, env=env,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"convert_hf_to_gguf failed: {proc.stderr[-2000:]}"
                        )
                    if not _publish_gguf_temp(token, temp_gguf, base_gguf):
                        return
                finally:
                    temp_gguf.unlink(missing_ok=True)
        result_path = base_gguf
        if gguf_type not in GGUF_BASE_TYPES:
            if not _gguf_job_is_current(token):
                return
            if quantize is None:
                raise RuntimeError(
                    "llama-quantize not found in the llama.cpp folder — only "
                    "bf16/f16 exports are possible"
                )
            if not _gguf_job_progress(token, f"quantizing to {gguf_type}"):
                return
            result_path = job_dir / f"{filename_stem}-{gguf_type}.gguf"
            temp_quantized = result_path.with_name(
                f".{result_path.stem}.tmp-{uuid.uuid4().hex}.gguf"
            )
            temp_quantized.unlink(missing_ok=True)
            with editing.artifact_lease(temp_quantized):
                try:
                    proc = subprocess.run(
                        [str(quantize), str(base_gguf), str(temp_quantized), gguf_type],
                        capture_output=True, text=True,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(f"llama-quantize failed: {proc.stderr[-2000:]}")
                    if not _publish_gguf_temp(token, temp_quantized, result_path):
                        return
                finally:
                    temp_quantized.unlink(missing_ok=True)
        if not _gguf_job_is_current(token):
            return
        details = _validate_gguf_artifact(result_path)
        _finish_gguf_job(token, result={
                "gguf": str(result_path),
                "size_bytes": details.st_size,
                "hf_cache": str(hf_dir),
            })
    except Exception as exc:
        _finish_gguf_job(token, error=str(exc))


def _make_gguf_bake_worker(dispatch):
    def worker():
        def execute(payload):
            call_kwargs = dict(payload["kwargs"])
            call_kwargs["publication_guard"] = payload["publication_gate"]
            return payload["export_fn"](
                payload["rules"], payload["bundle"].jl, payload["meta"],
                fmt="full", name=payload["name"], source_dir=payload["source_dir"],
                scale=payload["scale"], **call_kwargs,
            )
        return _execute_dispatched_worker(dispatch, execute)
    return worker


async def _prepare_gguf_bake_handoff(handoff, export_name):
    token = snap = outcome = prepared = payload_seed = publication_gate = payload = dispatch = task = None
    try:
        token, snap = handoff.token, handoff.snapshot
        outcome = _gguf_preflight_resource(snap)
        snap = None
        if outcome.status is not None:
            return HandoffSetup(status=outcome.status, kind="GGUFPreflight", message=outcome.message)
        prepared = outcome.value
        payload_seed = prepared.take()
        prepared = outcome = None
        publication_gate = editing.PublicationGate(
            lambda coordinator=handoff.coordinator, owner=token: coordinator.can_publish(owner)
        )
        payload = dict(payload_seed)
        payload.update(name=export_name, publication_gate=publication_gate)
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = payload_seed = None
        token = prepared = outcome = None
        task = await handoff.create_thread_task(
            _make_gguf_bake_worker, factory_args=(dispatch,),
        )
        return HandoffSetup(task=task, dispatch=dispatch, gate=publication_gate)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "GGUF bake setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        token = snap = outcome = prepared = payload_seed = publication_gate = payload = dispatch = task = None


@app.post("/api/edit/export-gguf")
async def api_edit_export_gguf(req: GGUFExportRequest):
    try:
        name_parts = editing.validate_export_name(req.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    validated_name = "/".join(name_parts)
    if req.gguf_type not in GGUF_BASE_TYPES + GGUF_QUANT_TYPES:
        raise HTTPException(422, f"unknown GGUF type: {req.gguf_type}")
    try:
        convert, quantize, gguf_py = _llamacpp_paths()
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if req.gguf_type not in GGUF_BASE_TYPES and quantize is None:
        raise HTTPException(422, "llama-quantize not found — pick bf16 or f16")
    job_dir = editing.EDITS_DIR.joinpath(*name_parts)
    filename_stem = name_parts[-1]
    hf_dir = job_dir / "hf"
    try:
        job_token = _reserve_gguf_job(validated_name)
    except OperationConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    try:
        baked = "reused"
        if not (hf_dir / "config.json").exists():
            _gguf_job_progress(job_token, "baking checkpoint")
            stop_event = ThreadingEvent()
            try:
                async with OperationHandoff.acquire_scope(
                    manager.coordinator, OperationType.GGUF_BAKE,
                    stop_event=stop_event, requires_loaded=True,
                ) as handoff:
                    setup = await _prepare_gguf_bake_handoff(
                        handoff, "/".join((*name_parts, "hf")),
                    )
            except OperationConflict as exc:
                raise _conflict(exc)
            except ModelStateError as exc:
                raise HTTPException(
                    422, "no model loaded (and no cached checkpoint for this name)"
                ) from None
            if setup.status is not None:
                raise HTTPException(
                    setup.status, setup.message or "GGUF bake setup failed"
                )
            task, dispatch, publication_gate = setup.task, setup.dispatch, setup.gate
            _raise_worker_outcome(await _await_transferred_worker(
                task, dispatch, on_cancel=publication_gate.cancel,
            ))
            baked = "baked"
        _gguf_job_progress(job_token, "starting")
        worker = threading.Thread(
            target=_gguf_worker,
            args=(job_token, validated_name, job_dir, filename_stem, req.gguf_type,
                  hf_dir, convert, quantize, gguf_py),
            daemon=True,
        )
        worker.start()
    except asyncio.CancelledError:
        _finish_gguf_job(job_token, error="GGUF export request cancelled")
        raise
    except HTTPException as exc:
        _finish_gguf_job(job_token, error=str(exc.detail))
        raise
    except BaseException as exc:
        _finish_gguf_job(job_token, error=str(exc))
        raise HTTPException(500, str(exc)) from exc
    return {"started": True, "checkpoint": baked, "state": _gguf_state_snapshot()}


class GGUFCacheRequest(BaseModel):
    name: str


@app.post("/api/edit/gguf-cache/delete")
def api_gguf_cache_delete(req: GGUFCacheRequest):
    """Drop the cached HF checkpoint of a GGUF export (the .gguf files stay)."""
    import shutil
    try:
        name_parts = editing.validate_export_name(req.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    hf_dir = editing.EDITS_DIR.joinpath(*name_parts) / "hf"
    global _gguf_delete_claim
    claim = uuid.uuid4().hex
    with _gguf_lock:
        if _gguf_owner is not None or _gguf_delete_claim is not None:
            raise HTTPException(409, "a GGUF export or cache deletion is in progress")
        _gguf_delete_claim = claim
    try:
        if not hf_dir.is_dir():
            raise HTTPException(404, f"no cached checkpoint for {req.name}")
        descendants = list(hf_dir.rglob("*"))
        if any((p.suffix.lower() == ".gguf" and p.is_file()) or
               (p.name == "config.json" and p.parent != hf_dir) for p in descendants):
            raise HTTPException(409, "cache contains nested export artifacts")
        freed = sum(f.stat().st_size for f in descendants if f.is_file())
        shutil.rmtree(hf_dir)
        return {"deleted": str(hf_dir), "freed_bytes": freed}
    finally:
        with _gguf_lock:
            if _gguf_delete_claim == claim:
                _gguf_delete_claim = None


class GenerateSyncRequest(BaseModel):
    messages: list[dict]
    sampling: dict = {}


def _bundle_from_snapshot_or_legacy(snap):
    if snap.bundle is None:
        raise ModelStateError("no model loaded")
    return snap.bundle


def _captured_generation_context(token, snap, stop_event):
    bundle = _bundle_from_snapshot_or_legacy(snap)
    return GenerationContext(
        token=token,
        model_session_id=snap.model_session_id,
        bundle=bundle,
        hf_model=bundle.hf_model,
        tokenizer=bundle.tokenizer,
        jl=bundle.jl,
        meta=bundle.meta,
        capability_profile=bundle.capability_profile,
        lens=lens_manager.snapshot_for_generation(),
        intervention_snapshot=_coordinated_interventions(snap),
        stop_event=stop_event,
    )


def _make_http_generation_worker(dispatch, stop_event, done, token):
    def emit(frame):
        if frame["type"] == "done":
            done.update(frame)
        elif frame["type"] == "error":
            done["error"] = frame.get("message")

    def worker():
        global _generation_counter
        def execute(payload):
            global _generation_counter
            context = payload["context"]
            manager.generate(
                payload["messages"], payload["sampling"], stop_event, emit,
                lens=context, ablator=interventions if context.intervention_snapshot.get("rules") else None,
            )
            if done.get("error"):
                raise RuntimeError(done["error"])
            _generation_counter += 1
            last = {
                "n": _generation_counter,
                "prompt": next((m.get("content", "") for m in reversed(payload["messages"]) if m.get("role") == "user"), ""),
                "text": done.get("text", ""), "stats": done.get("stats"), "meta": done.get("meta"),
            }
            manager.coordinator.publish_last_generation(token, context.model_session_id, last)
            done["last_generation"] = last
            return {"text": done.get("text", ""), "stats": done.get("stats"), "meta": done.get("meta"), "last_generation": last}
        return _execute_dispatched_worker(dispatch, execute)
    return worker


async def _prepare_http_generation_handoff(handoff, messages, sampling, stop_event):
    token = snap = context = payload = dispatch = task = done = None
    try:
        token, snap = handoff.token, handoff.snapshot
        context = _captured_generation_context(token, snap, stop_event)
        payload = {"context": context, "messages": list(messages), "sampling": dict(sampling)}
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = None
        snap = context = None
        done = {}
        task = await handoff.create_thread_task(
            _make_http_generation_worker,
            factory_args=(dispatch, stop_event, done, token),
        )
        return HandoffSetup(task=task, dispatch=dispatch, state=done)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "generation setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        token = snap = context = payload = dispatch = task = done = None


@app.post("/api/generate")
async def api_generate_sync(req: GenerateSyncRequest):
    stop_event = ThreadingEvent()
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.GENERATE,
            stop_event=stop_event, requires_loaded=True,
        ) as handoff:
            setup = await _prepare_http_generation_handoff(
                handoff, req.messages, req.sampling, stop_event,
            )
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "generation setup failed") from None
    task, dispatch = setup.task, setup.dispatch
    outcome_value = _raise_worker_outcome(await _await_transferred_worker(task, dispatch))
    for ws in list(_ws_locks):
        try:
            await _ws_send(ws, json.dumps({"type": "api_generation"}))
        except Exception:
            pass
    return {
        "text": (outcome_value or {}).get("text", ""),
        "stats": (outcome_value or {}).get("stats"),
        "meta": (outcome_value or {}).get("meta"),
        "last_generation": (outcome_value or {}).get("last_generation"),
    }



class NeighborsRequest(BaseModel):
    token_ids: list[int]
    k: int = 3


def _make_token_neighbors_worker(dispatch):
    def worker():
        return _execute_dispatched_worker(
            dispatch,
            lambda payload: neighbors.lookup(
                payload["bundle"].jl, payload["bundle"].tokenizer,
                payload["key"], payload["token_ids"], payload["k"],
            ),
        )
    return worker


async def _prepare_token_neighbors_handoff(handoff, token_ids, k):
    snap = bundle = jl = tokenizer = meta_copy = key = payload = dispatch = task = None
    try:
        snap = handoff.snapshot
        bundle = snap.bundle
        jl, tokenizer = bundle.jl, bundle.tokenizer
        meta_copy = dict(bundle.meta)
        key = (snap.model_session_id, meta_copy.get("model_id"), meta_copy.get("revision"))
        payload = {
            "bundle": bundle, "key": key, "token_ids": list(token_ids[:64]), "k": k,
        }
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = None
        snap = bundle = jl = tokenizer = meta_copy = key = None
        task = await handoff.create_thread_task(
            _make_token_neighbors_worker, factory_args=(dispatch,),
        )
        return HandoffSetup(task=task, dispatch=dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "token-neighbor setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        snap = bundle = jl = tokenizer = meta_copy = key = payload = dispatch = task = None


@app.post("/api/token-neighbors")
async def api_token_neighbors(req: NeighborsRequest):
    stop_event = ThreadingEvent()
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.MODEL_READ,
            stop_event=stop_event, requires_loaded=True,
        ) as handoff:
            setup = await _prepare_token_neighbors_handoff(
                handoff, req.token_ids, req.k,
            )
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    except asyncio.CancelledError:
        raise
    except HTTPException:
        raise
    except BaseException as exc:
        _kind, detail = _bounded_failure(exc, "token-neighbor setup failed")
        raise HTTPException(500, detail) from None
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "token-neighbor setup failed") from None
    task, dispatch = setup.task, setup.dispatch
    result = _raise_worker_outcome(await _await_transferred_worker(task, dispatch))
    return {"neighbors": {str(tid): entries for tid, entries in result.items()}}



@app.get("/api/token-lookup")
def api_token_lookup(q: str):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True)
        outcome = _token_lookup_resource(snap, q)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _token_lookup_resource(snap, q: str):
    tokenizer = candidates = None
    try:
        tokenizer = snap.bundle.tokenizer
        candidates = {}
        for variant in (q, " " + q, q.lower(), " " + q.lower(),
                        q.capitalize(), " " + q.capitalize(), q.upper(), " " + q.upper()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in candidates:
                candidates[ids[0]] = tokenizer.decode([ids[0]])
        return _route_value({"candidates": [{"id": tid, "str": s} for tid, s in candidates.items()]})
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return _route_http(exc.status_code, exc.detail)
        return _route_exception(500, exc)
    finally:
        tokenizer = candidates = None


@app.get("/api/registry/local")
def api_registry_local():
    return {"lenses": registry.local_lenses()}


@app.get("/api/registry/for-model")
async def api_registry_for_model(model_id: str, revision: str | None = None):
    return await asyncio.to_thread(registry.lenses_for_model, model_id, revision)


@app.get("/api/registry/resolve")
def api_registry_resolve(path: str | None = None, repo_id: str | None = None, filename: str | None = None):
    return registry.resolve_lens(path=path, repo_id=repo_id, filename=filename)


@app.post("/api/fit")
def api_fit(req: FitRequest):
    token = None
    try:
        token, _ = manager.coordinator.acquire(OperationType.FIT, requires_loaded=False)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError:
        raise HTTPException(409, "unload the model first: fitting needs all the VRAM")
    valid = _valid_devices()
    bad = [d for d in req.devices if d not in valid]
    if bad:
        manager.coordinator.release(token)
        raise HTTPException(422, f"invalid device(s): {', '.join(bad)}")
    source = resolve_source(req.model_id)
    transferred = False

    def release_fit():
        manager.coordinator.release(token)

    try:
        result = fit_manager.start(
            model_id=req.model_id,
            source=source,
            model_revision=_resolve_revision(source),
            n_prompts=req.n_prompts,
            dtype=req.dtype,
            quant=req.quant,
            datasets=req.datasets,
            devices=req.devices,
            name=req.name,
            dim_batch=req.dim_batch,
            max_seq_len=req.max_seq_len,
            source_layers=req.source_layers,
            continue_from=req.continue_from,
            reservation_release=release_fit,
        )
        transferred = True
        return result
    except ValueError as exc:
        if not transferred:
            manager.coordinator.release(token)
        raise HTTPException(409, str(exc))
    except Exception:
        if not transferred:
            manager.coordinator.release(token)
        raise


@app.get("/api/fit/status")
def api_fit_status():
    return fit_manager.state


@app.post("/api/fit/stop")
def api_fit_stop():
    return fit_manager.stop()


def _make_lens_pin_worker(dispatch):
    def worker():
        return _execute_dispatched_worker(
            dispatch,
            lambda payload: lens_manager.pin_ranks(
                payload["gen_id"], payload["token_ids"], payload["jl"],
                generation_run_id=payload["generation_run_id"],
            ),
        )
    return worker


async def _prepare_lens_pin_handoff(handoff, gen_id, generation_run_id, token_ids):
    snap = bundle = jl = payload = dispatch = task = None
    try:
        snap = handoff.snapshot
        bundle = snap.bundle
        jl = bundle.jl
        payload = {
            "jl": jl, "gen_id": gen_id, "generation_run_id": generation_run_id,
            "token_ids": list(token_ids),
        }
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = None
        snap = bundle = jl = None
        task = await handoff.create_thread_task(
            _make_lens_pin_worker, factory_args=(dispatch,),
        )
        return HandoffSetup(task=task, dispatch=dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "lens pin setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        snap = bundle = jl = payload = dispatch = task = None


@app.post("/api/lens/pin")
async def api_lens_pin(req: PinRequest):
    run_id = req.generation_run_id
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(ch not in "0123456789abcdef" for ch in run_id)
    ):
        raise HTTPException(409, "generation run identity is required; residual capture may have expired")
    stop_event = ThreadingEvent()
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.MODEL_READ,
            stop_event=stop_event, requires_loaded=True,
        ) as handoff:
            setup = await _prepare_lens_pin_handoff(
                handoff, req.gen_id, req.generation_run_id, req.token_ids,
            )
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    if setup.status is not None:
        raise HTTPException(setup.status, setup.message or "lens pin setup failed") from None
    task, dispatch = setup.task, setup.dispatch
    return _raise_worker_outcome(await _await_transferred_worker(task, dispatch))



# Alternative/duplicate weight folders, never needed for transformers inference
# (GPT-OSS-20B ships original/ + metal/ = 2 × ~14 GB of waste).
DOWNLOAD_IGNORE_DIRS = [
    "original/*", "metal/*", "onnx/*", "openvino/*", "coreml/*", "gguf/*",
]
# Auxiliary files that are always useful (configs, tokenizer, custom code) — light.
DOWNLOAD_EXTRAS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py", "*.md"]
# "large" fp32 model: past 1 GB of weights, convert to bf16 automatically
AUTO_BF16_MIN_BYTES = 1_000_000_000


def _plan_download(api, repo_id, token):
    """Pick the strict minimum: ONE weight set (the lightest if several variants)
    + the auxiliary files. Returns (allow_patterns, ignore_patterns, plan) —
    allow_patterns None = "take everything" fallback."""
    import re

    files = {}  # path -> size
    for entry in api.list_repo_tree(repo_id, recursive=True, token=token or None):
        size = getattr(entry, "size", None)
        if size is not None:
            files[entry.path] = size

    def in_ignored_dir(path):
        return any(path.startswith(d.split("/*")[0] + "/") for d in DOWNLOAD_IGNORE_DIRS)

    # root-level safetensors sets, grouped by variant:
    # "model(-00001-of-00002)?.safetensors" -> group "model";
    # "model.fp32(-...)?.safetensors" -> group "model.fp32", etc.
    st_groups = {}
    for path, size in files.items():
        if "/" in path or not path.endswith(".safetensors"):
            continue
        stem = re.sub(r"-\d{5}-of-\d{5}", "", path.removesuffix(".safetensors"))
        st_groups.setdefault(stem, []).append(path)

    if st_groups:
        stem, chosen = min(
            st_groups.items(), key=lambda kv: sum(files[p] for p in kv[1])
        )
        index = f"{stem}.safetensors.index.json"
        patterns = sorted(chosen) + ([index] if index in files else []) + DOWNLOAD_EXTRAS
        return patterns, None, {
            "kind": "safetensors",
            "variant": stem,
            "size_bytes": sum(files[p] for p in chosen),
        }

    # no safetensors (legacy .bin/.h5 repos, or GGUF-only ones we can't load):
    # take everything except the alternative folders and obvious format
    # duplicates. GGUF weights are ignored — J-Wash only loads transformers
    # (safetensors) models.
    ignore = DOWNLOAD_IGNORE_DIRS + ["*.gguf", "*.msgpack", "*.h5", "*.tflite", "*.onnx"]
    return None, ignore, {
        "kind": "fallback",
        "size_bytes": sum(
            s for p, s in files.items()
            if not in_ignored_dir(p) and not p.endswith(".gguf")
        ),
    }


def _maybe_autoconvert_bf16(repo_id, state):
    """After download: if the safetensors weights are float32 and heavy, convert
    to bf16 automatically into a local folder (halves the space in use; the HF
    cache source stays intact)."""
    from pathlib import Path

    from safetensors import safe_open

    from core.model_manager import convert_to_bf16, resolve_local_dir

    src = resolve_local_dir(repo_id)
    if not src:
        return
    src = Path(src)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        return
    total = sum(s.stat().st_size for s in shards)
    if total < AUTO_BF16_MIN_BYTES:
        return
    import math

    with safe_open(str(shards[0]), framework="pt") as f:
        keys = list(f.keys())
        if not keys:
            return
        # the shard's biggest tensor is representative of the "large layers"
        biggest = max(keys, key=lambda k: math.prod(f.get_slice(k).get_shape()))
        dtype = str(f.get_slice(biggest).get_dtype())
    if dtype not in ("F32", "F64"):
        return
    state.update(state="converting")
    base = repo_id.split("/")[-1]
    result = convert_to_bf16(str(src), out_dir=str(config.LOCAL_MODELS_ROOT / f"{base}-bf16"))
    state.update(converted=result["id"])


def _download_worker(repo_id):
    import os

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    state = _downloads[repo_id]
    try:
        allow, ignore, plan = None, None, None
        try:
            allow, ignore, plan = _plan_download(HfApi(), repo_id, token)
            state.update(plan=plan)
        except Exception:
            # planning failed (network, permissions): cautious fallback
            ignore = DOWNLOAD_IGNORE_DIRS + ["*.gguf", "*.msgpack", "*.h5", "*.tflite", "*.onnx"]
        # progress: poll the cache size (robust — the tqdm hook misses some files
        # depending on the download mechanism). We sum the repo's blobs (including
        # .incomplete files) and compare to the planned total.
        total_bytes = (plan or {}).get("size_bytes") or 0
        _blobs = config.HF_CACHE / "hub" / f"models--{repo_id.replace('/', '--')}" / "blobs"
        stop_poll = ThreadingEvent()

        def _poll_progress():
            while not stop_poll.is_set():
                done = 0
                if _blobs.exists():
                    for f in _blobs.iterdir():
                        try:
                            done += f.stat().st_size
                        except OSError:
                            pass
                if total_bytes:
                    state["progress"] = {"done": min(done, total_bytes), "total": total_bytes}
                stop_poll.wait(1.0)

        poller = threading.Thread(target=_poll_progress, daemon=True)
        poller.start()
        try:
            snapshot_download(
                repo_id, token=token or None,
                allow_patterns=allow, ignore_patterns=ignore,
            )
        finally:
            stop_poll.set()
            state.pop("progress", None)
        if plan and plan["kind"] == "safetensors":
            _maybe_autoconvert_bf16(repo_id, state)
        state.update(state="done", error=None)
    except GatedRepoError:
        msg = (
            f'gated repo "{repo_id}": accept the terms on huggingface.co and make '
            "sure a valid HF_TOKEN is set in the environment."
            + ("" if token else " (no HF_TOKEN detected)")
        )
        state.update(state="error", error=msg)
    except RepositoryNotFoundError:
        state.update(
            state="error",
            error=f'repo "{repo_id}" not found (or private without access using the current token)',
        )
    except Exception as exc:
        state.update(state="error", error=str(exc))


@app.post("/api/download")
def api_download(req: DownloadRequest):
    repo_id = req.repo_id.strip()
    with _downloads_lock:
        current = _downloads.get(repo_id)
        if current and current["state"] == "running":
            raise HTTPException(409, f"download already in progress: {repo_id}")
        # several downloads in parallel: one state per repo
        _downloads[repo_id] = {"repo_id": repo_id, "state": "running", "error": None}
    threading.Thread(target=_download_worker, args=(repo_id,), daemon=True).start()
    return _downloads[repo_id]


@app.delete("/api/download/{repo_id:path}")
def api_download_dismiss(repo_id: str):
    """Remove a finished (done/error) entry from the displayed list."""
    with _downloads_lock:
        state = _downloads.get(repo_id)
        if state and state["state"] != "running":
            del _downloads[repo_id]
    return {"downloads": list(_downloads.values())}


@app.get("/api/browse")
def api_browse(path: str | None = None):
    from core.model_manager import browse_dir

    try:
        return browse_dir(path)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


class PickPathRequest(BaseModel):
    kind: str = "dir"  # kept for API compatibility; only directory picking is used


_pick_lock = threading.Lock()


@app.post("/api/pick-path")
async def api_pick_path(req: PickPathRequest):
    """Open Windows' NATIVE file picker (the server runs on the user's own
    machine) and return the chosen path — this notably lets you paste a path,
    which the built-in browser cannot do."""

    def pick():
        if not _pick_lock.acquire(blocking=False):
            raise ValueError("a file picker is already open")
        try:
            # tkinter ships with CPython on every platform, but headless
            # Linux installs may lack it (or a display): fail with a hint
            # instead of a stack trace — the built-in Browse still works.
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                raise ValueError(
                    "no native folder picker available (tkinter missing) — "
                    "use the built-in Browse, or paste the path directly"
                )

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                path = filedialog.askdirectory(
                    parent=root, title="Choose a model folder (HF)"
                )
            finally:
                root.destroy()
            return {"path": path or None}
        finally:
            _pick_lock.release()

    try:
        return await asyncio.to_thread(pick)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


class ConvertRequest(BaseModel):
    path: str


_convert_state = {"path": None, "state": "idle", "error": None, "result": None}


def _convert_worker(path):
    from core.model_manager import convert_to_bf16

    try:
        result = convert_to_bf16(path)
        _convert_state.update(state="done", error=None, result=result)
    except Exception as exc:
        _convert_state.update(state="error", error=str(exc))


@app.post("/api/convert-bf16")
def api_convert_bf16(req: ConvertRequest):
    if _convert_state["state"] == "running":
        raise HTTPException(409, "a conversion is already in progress")
    _convert_state.update(path=req.path, state="running", error=None, result=None)
    threading.Thread(target=_convert_worker, args=(req.path,), daemon=True).start()
    return _convert_state


@app.get("/api/conversations")
def api_conversations(query: str | None = None):
    return {"conversations": store.list_conversations(query)}


@app.get("/api/conversations/{cid}")
def api_conversation(cid: int):
    try:
        return store.get_conversation(cid)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.patch("/api/conversations/{cid}")
def api_conversation_patch(cid: int, req: ConversationPatch):
    outcome = store.update_conversation(cid, title=req.title, tags=req.tags)
    if outcome.state in {"stale", "superseded"}:
        raise HTTPException(409, "conversation changed before update completed")
    if outcome.state == "ambiguous":
        raise HTTPException(500, f"conversation {outcome.entity_id} update outcome is ambiguous")
    if outcome.state != "committed":
        raise HTTPException(500, "conversation update did not commit")
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
def api_conversation_delete(cid: int):
    outcome = store.delete_conversation(cid)
    if outcome.state in {"stale", "superseded"}:
        raise HTTPException(409, "conversation changed before deletion completed")
    if outcome.state == "ambiguous":
        raise HTTPException(500, f"conversation {outcome.entity_id} deletion outcome is ambiguous")
    if outcome.state != "committed":
        raise HTTPException(500, "conversation deletion did not commit")
    return {"ok": True}


@app.get("/api/messages/{mid}/frames")
def api_message_frames(mid: int):
    try:
        archive = store.load_frames(mid)
        if archive.get("pin_publication_state") == "published" and not lens_manager.has_published_gen(
            archive.get("pin_gen_id"), archive.get("pin_generation_run_id"),
        ):
            archive = dict(archive)
            archive["pin_publication_state"] = "unavailable"
            archive["pin_gen_id"] = None
            archive["pin_generation_run_id"] = None
            archive["frames"] = [
                dict(frame, gen=None, generation_run_id=None)
                for frame in archive.get("frames", [])
            ]
        return archive
    except FramesNotAttached as exc:
        raise HTTPException(404, str(exc)) from None
    except FramePointerTurnover as exc:
        raise HTTPException(409, str(exc)) from None
    except FrameFileMissing as exc:
        raise HTTPException(422, str(exc)) from None
    except FrameStorageError as exc:
        raise HTTPException(422, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


class MessagePatch(BaseModel):
    content: str


@app.patch("/api/messages/{mid}")
def api_message_patch(mid: int, req: MessagePatch):
    """Edit a message's content (e.g. rewrite an assistant reply). Later turns
    are generated from the stored path, so the edit takes effect immediately."""
    try:
        current = store.get_message(mid)
        outcome = store.update_message(
            mid,
            req.content,
            meta={
                "provenance_invalidated": True,
                "edited": True,
                "provenance": "user_edited_unknown",
                "continuations": [],
            },
            clear_frames=True,
            expected_version=current.get("version", 0),
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    if outcome.state in {"stale", "superseded"}:
        raise HTTPException(409, f"message {mid} changed before edit completed")
    if outcome.state == "ambiguous":
        raise HTTPException(500, f"message {outcome.entity_id} edit outcome is ambiguous")
    if outcome.state != "committed":
        raise HTTPException(500, f"message {mid} edit did not commit")
    return {"ok": True, "id": mid}


@app.get("/api/conversations/{cid}/export")
def api_conversation_export(cid: int, format: str = "json", frames: int = 0):
    try:
        body, media_type = store.export(cid, fmt=format, include_frames=bool(frames))
    except FramesNotAttached as exc:
        raise HTTPException(404, str(exc)) from None
    except FramePointerTurnover as exc:
        raise HTTPException(409, str(exc)) from None
    except (FrameFileMissing, FrameStorageError) as exc:
        raise HTTPException(422, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
    ext = "json" if format == "json" else "md"
    return Response(
        content=body,
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="conversation-{cid}.{ext}"'},
    )


def _generate_safely(messages, sampling, stop_event, emit, context):
    try:
        manager.generate(
            messages, sampling, stop_event, emit, lens=context,
            ablator=interventions if context.intervention_snapshot.get("rules") else None,
        )
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _persisted_mutation_value(outcome, label, emit):
    if outcome.state == "committed":
        return outcome.value
    if outcome.state in {"stale", "superseded"}:
        message = f"{label} was superseded by newer durable state"
    elif outcome.state == "ambiguous":
        message = f"{label} outcome is ambiguous for ID {outcome.entity_id}; operator recovery is required"
    else:
        message = f"{label} did not commit; storage recovery is required"
    emit({"type": "error", "message": message})
    return None


def _persisted_generate(req, stop_event, emit, gen_context):
    try:
        continue_id = req.get("continue_message_id")
        if continue_id is not None:
            _persisted_continue(req, continue_id, stop_event, emit, gen_context)
            return
        conversation_id = req.get("conversation_id")
        parent_id = req.get("parent_id")
        system = (req.get("system") or "").strip()
        content = req.get("content")
        if conversation_id is None and not content:
            emit({"type": "error", "message": "content required for a new conversation"})
            return
        if conversation_id is None:
            conversation_id = _persisted_mutation_value(
                store.create_conversation(content[:60]), "conversation creation", emit
            )
            if conversation_id is None:
                return
            if system:
                inserted = _persisted_mutation_value(
                    store.add_message(conversation_id, None, "system", system),
                    "system message insertion", emit,
                )
                if inserted is None:
                    return
                parent_id = inserted[0]
        if content:
            inserted = _persisted_mutation_value(
                store.add_message(conversation_id, parent_id, "user", content),
                "user message insertion", emit,
            )
            if inserted is None:
                return
            parent_id = inserted[0]
        if parent_id is None:
            emit({"type": "error", "message": "parent_id or content required"})
            return
        emit({
            "type": "persisted",
            "conversation_id": conversation_id,
            "user_message_id": parent_id,
        })
        messages = store.path_to_root(parent_id)
        lens = gen_context.lens
        layers_used = list(lens.layers) if lens is not None else []
        k_used = lens.k if lens is not None else 0
        frames_acc = []
        done_holder = {}

        def emit_inner(frame):
            if frame["type"] == "done":
                done_holder.update(frame)
            else:
                if frame["type"] == "frame":
                    frames_acc.append(frame)
                emit(frame)

        manager.generate(
            messages=messages, sampling=req.get("sampling", {}), stop_event=stop_event, emit=emit_inner, lens=gen_context,
            ablator=interventions if gen_context.intervention_snapshot.get("rules") else None,
        )
        durable_end = done_holder.get("generated_token_end_pos")
        if isinstance(durable_end, int) and not isinstance(durable_end, bool):
            frames_acc[:] = [
                frame for frame in frames_acc
                if frame.get("phase") in ("reading", "prompt") or frame.get("pos", durable_end) < durable_end
            ]
        meta = dict(
            done_holder.get("meta") or {},
            stats=done_holder.get("stats"),
            stopped=done_holder.get("stopped"),
        )
        insert_meta = dict(meta)
        if frames_acc:
            insert_meta.update(publication_state="frames_pending", frames_expected=True)
        else:
            insert_meta.update(publication_state="complete", frames_expected=False)
        inserted = _persisted_mutation_value(store.add_message(
            conversation_id, parent_id, "assistant", done_holder.get("text", ""), meta=insert_meta,
            return_version=True,
        ), "assistant message insertion", emit)
        if inserted is None:
            return
        message_id, inserted_version = inserted
        if frames_acc:
            complete_meta = dict(
                meta,
                publication_state="complete",
                frames_expected=True,
                **_frames_lens_signature(lens, layers_used, k_used),
            )
            attached = store.save_frames(
                message_id, frames_acc, layers_used, k_used,
                expected_version=inserted_version, complete_meta=complete_meta,
                frame_descriptor=_frame_descriptor(lens, layers_used, k_used),
            )
            if attached.state in {"stale", "superseded"}:
                emit({"type": "error", "message": "message changed before frames were attached"})
                return
            if attached.state == "ambiguous":
                candidate = f", candidate {attached.value}" if isinstance(attached.value, str) else ""
                versions = f", expected version {attached.expected_version}, observed version {attached.observed_version}"
                emit({
                    "type": "error",
                    "message": f"frame attachment durability is unknown for message {attached.entity_id}{candidate}{versions}; operator recovery is required",
                })
                return
            if attached.state == "not_committed":
                failure_meta = dict(
                    meta, publication_state="frames_publication_failed", frames_expected=True,
                    frames_error_kind=attached.error_kind,
                    frames_error=(attached.error_message or "frame attachment did not commit")[:512],
                )
                marked = store.mark_frame_publication_failed(
                    message_id, expected_version=inserted_version, failure_meta=failure_meta
                )
                if marked.state == "ambiguous":
                    emit({"type": "error", "message": f"frame failure outcome is ambiguous for message {marked.entity_id}; operator recovery is required"})
                    return
                if marked.state in {"stale", "superseded"}:
                    emit({"type": "error", "message": "message changed before frame failure could be recorded"})
                    return
                if marked.state == "not_committed":
                    emit({"type": "error", "message": "frame failure did not commit; storage recovery is required"})
                    return
                emit({"type": "error", "message": "frames could not be published", "message_id": message_id})
                return
            if attached.state != "committed":
                emit({"type": "error", "message": "invalid frame attachment outcome"})
                return
        emit(dict(done_holder, conversation_id=conversation_id, message_id=message_id))
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _continuation_metadata(previous_meta, start_offset, end_offset, done_meta, stats, stopped):
    previous_meta = dict(previous_meta or {})
    latest_meta = dict(done_meta or {})
    latest_meta.update(stats=stats, stopped=stopped, continued=True)
    continuations = list(previous_meta.get("continuations") or [])
    if not continuations and start_offset > 0:
        base_meta = dict(previous_meta)
        base_meta.pop("continuations", None)
        base_meta.pop("continuation_attempts", None)
        if previous_meta.get("continued"):
            continuations.append({
                "start_offset": 0,
                "end_offset": start_offset,
                "provenance": "legacy_mixed_unknown",
                "meta": None,
            })
        else:
            continuations.append({"start_offset": 0, "end_offset": start_offset, "meta": base_meta})
    merged = dict(previous_meta)
    if end_offset > start_offset:
        continuations.append({
            "start_offset": start_offset,
            "end_offset": end_offset,
            "meta": dict(latest_meta),
        })
        merged.update(latest_meta)
        merged["continuations"] = continuations
    else:
        attempts = list(previous_meta.get("continuation_attempts") or [])
        attempts.append({"offset": start_offset, "meta": latest_meta})
        if continuations:
            merged["continuations"] = continuations
        merged["continuation_attempts"] = attempts
    return merged


def _frame_descriptor(lens, layers, k):
    if lens is None or not getattr(lens, "meta", None):
        return None
    descriptor = _stable_lens_descriptor(lens.meta)
    return {
        key: value for key, value in descriptor.items()
        if not key.startswith("runtime_")
    } | {"archive_layers": [int(layer) for layer in layers], "k": int(k)}


def _frames_lens_signature(lens, layers, k):
    descriptor = _frame_descriptor(lens, layers, k)
    if lens is None:
        return {"frames_lens_binding_id": None, "frames_layers": [], "frames_k": 0, "frame_descriptor": None}
    return {
        "frames_lens_binding_id": getattr(lens, "binding_id", None),
        "frames_layers": [int(l) for l in layers],
        "frames_k": int(k),
        "frame_descriptor": descriptor,
    }


def _validate_frame_descriptor(existing_archive, lens, layers, k):
    if existing_archive["layers"] != [int(l) for l in layers] or existing_archive["k"] != int(k):
        raise ValueError("continuation lens frame configuration changed")
    saved = existing_archive.get("descriptor")
    current = _frame_descriptor(lens, layers, k)
    if saved is None:
        raise ValueError("existing frame archive has unknown lens/model provenance")
    if _lens_descriptor_mismatch(saved, current) or saved.get("archive_layers") != current.get("archive_layers"):
        raise ValueError("continuation lens/model frame provenance changed")


def _finalize_continuation_frames(new_frames, suffix_start, suffix_end, generation_run_id):
    if isinstance(suffix_start, bool) or not isinstance(suffix_start, int) or suffix_start < 0:
        raise ValueError("continuation did not provide an exact suffix boundary")
    if isinstance(suffix_end, bool) or not isinstance(suffix_end, int) or suffix_end < suffix_start:
        raise ValueError("continuation provided an invalid suffix end boundary")
    positions = [frame.get("pos") for frame in new_frames]
    if any(isinstance(pos, bool) or not isinstance(pos, int) or pos < 0 for pos in positions):
        raise ValueError("continuation emitted an invalid frame position")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError("continuation emitted duplicate or decreasing frame positions")
    finalized = [
        frame for frame in new_frames
        if (
            frame.get("phase") in ("reading", "prompt") and frame["pos"] < suffix_start
        ) or (
            frame.get("phase") in ("thinking", "gen") and suffix_start <= frame["pos"] < suffix_end
        )
    ]
    if any(frame.get("generation_run_id") != generation_run_id for frame in finalized):
        raise ValueError("continuation frames do not belong to the current generation run")
    if any(finalized[i].get("pos") >= finalized[i + 1].get("pos") for i in range(len(finalized) - 1)):
        raise ValueError("continuation frames are not ordered after finalization")
    return finalized


def _resolve_continuation_pin_unavailable_total(
    message_id, *, expected_version, generation_run_id, gen_id,
):
    """Resolve one exact durable pin attempt to unavailable or superseded."""
    while True:
        try:
            outcome = store.finalize_continuation_pin_publication(
                message_id,
                expected_version=expected_version,
                generation_run_id=generation_run_id,
                gen_id=gen_id,
                state="unavailable",
            )
            state = outcome.state
        except BaseException:
            _retry_thread_yield()
            continue
        if state in {"committed", "superseded"}:
            return state
        _retry_thread_yield()


def _persisted_continue(req, message_id, stop_event, emit, gen_context):
    """Persist one continuation while owning its provisional residual capture."""
    msg = store.get_message(message_id)
    expected_content = msg["content"]
    expected_meta = msg.get("meta")
    expected_version = msg.get("version", 0)
    if msg["role"] != "assistant":
        emit({"type": "error", "message": "only an assistant reply can be continued"})
        return
    messages = store.path_to_root(message_id)
    lens = gen_context.lens
    layers_used = list(lens.layers) if lens is not None else []
    k_used = lens.k if lens is not None else 0
    frames_acc = []
    done_holder = {}
    existing_archive = None

    if msg.get("frames_file"):
        try:
            existing_archive = store.load_frames(message_id)
            if lens is not None:
                _validate_frame_descriptor(existing_archive, lens, layers_used, k_used)
        except Exception as exc:
            emit({"type": "error", "message": f"could not preflight existing frames; continuation not started: {exc}"})
            return

    def emit_inner(frame):
        if frame["type"] == "done":
            done_holder.update(frame)
        elif frame["type"] == "frame":
            frames_acc.append(frame)
        else:
            emit(frame)

    attempt_owner = GenerationAttemptOwner()
    generation_returned = False
    try:
        manager.generate(
            messages=messages, sampling=req.get("sampling", {}), stop_event=stop_event, emit=emit_inner, lens=gen_context,
            ablator=interventions if gen_context.intervention_snapshot.get("rules") else None,
            continue_final=True,
            capture_full_input_frames=lens is not None,
            defer_lens_publication=lens is not None,
            attempt_owner=attempt_owner,
        )
        generation_returned = True
    finally:
        if not generation_returned and lens is not None and attempt_owner.gen_id is not None:
            try:
                lens.unpublish_gen(
                    attempt_owner.gen_id,
                    generation_run_id=attempt_owner.generation_run_id,
                    model_session_id=attempt_owner.model_session_id,
                    lens_binding_id=attempt_owner.lens_binding_id,
                )
            except BaseException:
                pass
            try:
                lens.discard_gen(attempt_owner.gen_id)
            except BaseException:
                pass
    attempt_gen_id = (
        attempt_owner.gen_id if lens is not None and attempt_owner.gen_id is not None
        else done_holder.get("gen_id") if lens is not None else None
    )
    attempt_run_id = attempt_owner.generation_run_id or done_holder.get("generation_run_id")
    attempt_resolved = attempt_gen_id is None
    durable_committed = False
    durable_version = expected_version + 1

    def discard_attempt():
        nonlocal attempt_resolved
        if not attempt_resolved and lens is not None:
            lens.discard_gen(attempt_gen_id)
        attempt_resolved = True

    def resolve_durable_state(state):
        return store.finalize_continuation_pin_publication(
            message_id,
            expected_version=durable_version,
            generation_run_id=attempt_run_id,
            gen_id=attempt_gen_id,
            state=state,
        )

    try:
        old_content = expected_content
        appended = done_holder.get("text", "")
        if not appended:
            frames_acc.clear()
            discard_attempt()
            emit(dict(
                type="done", conversation_id=msg["conversation_id"], message_id=message_id,
                text="", content=old_content, continued=False, continuation_noop=True,
            ))
            return

        new_content = old_content + appended
        previous_meta = json.loads(expected_meta) if expected_meta else {}
        done_meta = dict(done_holder.get("meta") or {})
        # Pin identity has its own durable state; model provenance metadata must
        # not independently resurrect an unpublished run after reload.
        done_meta.pop("generation_run_id", None)
        meta = _continuation_metadata(
            previous_meta, len(old_content), len(new_content),
            done_meta, done_holder.get("stats"), done_holder.get("stopped"),
        )
        merged = None
        clear_frames = False
        if frames_acc:
            for stale_key in (
                "frames_invalidated", "frames_invalidation_reason", "frames_error_kind",
                "frames_error", "frames_pending",
            ):
                meta.pop(stale_key, None)
            initial_pin_state = "pending" if attempt_gen_id is not None else "unavailable"
            meta.update(
                publication_state="complete", frames_expected=True,
                pin_publication_state=initial_pin_state,
            )
            meta.pop("pin_gen_id", None)
            meta.pop("pin_generation_run_id", None)
            if meta.get("continuations"):
                meta["continuations"][-1]["pin_publication_state"] = initial_pin_state
            meta.update(_frames_lens_signature(lens, layers_used, k_used))
            merged = _finalize_continuation_frames(
                frames_acc,
                done_holder.get("continuation_suffix_start_pos"),
                done_holder.get("continuation_suffix_end_pos"),
                attempt_run_id,
            )
        elif msg.get("frames_file") and lens is None:
            clear_frames = True
            meta.update(
                frames_expected=False,
                frames_invalidated=True,
                frames_invalidation_reason="lens_disabled_continuation",
            )
        outcome = store.update_message_and_frames_if_unchanged(
            message_id, expected_version, new_content, meta,
            frames=merged, layers=layers_used, k=k_used, clear_frames=clear_frames,
            frame_descriptor=_frame_descriptor(lens, layers_used, k_used) if merged is not None else None,
            pin_publication_state=(
                "pending" if merged is not None and attempt_gen_id is not None
                else "unavailable" if merged is not None else None
            ),
        )
        if outcome.state in {"stale", "superseded"}:
            discard_attempt()
            emit({"type": "error", "message": "message changed during continuation"})
            return
        if outcome.state == "ambiguous":
            discard_attempt()
            candidate = f", candidate {outcome.value}" if isinstance(outcome.value, str) else ""
            versions = f", expected version {outcome.expected_version}, observed version {outcome.observed_version}"
            emit({"type": "error", "message": f"continuation durability is unknown for message {outcome.entity_id}{candidate}{versions}; operator recovery is required"})
            return
        if outcome.state == "not_committed":
            discard_attempt()
            emit({"type": "error", "message": "continuation did not commit; storage recovery is required"})
            return
        if outcome.state != "committed":
            discard_attempt()
            emit({"type": "error", "message": "invalid continuation storage outcome"})
            return
        durable_committed = True
        terminal = dict(done_holder)

        if attempt_gen_id is not None:
            publication_ok = False
            publication_receipt = None
            try:
                publication_receipt = lens.begin_gen_publication(attempt_gen_id)
                durable = resolve_durable_state("published")
                if durable.state != "committed":
                    raise RuntimeError(f"durable pin publication state was {durable.state}")
                publication = lens.commit_gen_publication(publication_receipt)
                publication_ok = (
                    publication.published
                    and publication.gen_id == attempt_gen_id
                    and publication.generation_run_id == attempt_run_id
                )
                if not publication_ok:
                    raise RuntimeError("residual publication identity mismatch")
                attempt_resolved = True
                terminal["pin_publication_state"] = "published"
            except BaseException as exc:
                logging.getLogger(__name__).warning(
                    "durable continuation committed but residual publication failed: %s",
                    _bounded_failure(exc, "residual publication failed")[1],
                )
                if publication_receipt is not None:
                    while True:
                        try:
                            lens.rollback_gen_publication(publication_receipt)
                            break
                        except ValueError:
                            # A successful commit consumes the receipt.  Exact
                            # unpublish below handles the post-commit case.
                            break
                        except BaseException:
                            _retry_thread_yield()
                lens.unpublish_gen(
                    attempt_gen_id,
                    generation_run_id=attempt_run_id,
                    model_session_id=lens.model_session_id,
                    lens_binding_id=lens.binding_id,
                )
                discard_attempt()
                unavailable_state = _resolve_continuation_pin_unavailable_total(
                    message_id,
                    expected_version=durable_version,
                    generation_run_id=attempt_run_id,
                    gen_id=attempt_gen_id,
                )
                if unavailable_state == "superseded":
                    frames_acc.clear()
                    emit({
                        "type": "error",
                        "message": "message changed while continuation pin publication was recovering",
                    })
                    return
                terminal.pop("gen_id", None)
                terminal.pop("generation_run_id", None)
                terminal["pin_publication_state"] = "unavailable"
                publication_ok = False
            if publication_ok:
                for frame in frames_acc:
                    emit(frame)
        else:
            if merged is not None:
                terminal["pin_publication_state"] = "unavailable"
                terminal.pop("gen_id", None)
                terminal.pop("generation_run_id", None)
                for frame in frames_acc:
                    emit(dict(frame, gen=None, generation_run_id=None))
            else:
                for frame in frames_acc:
                    emit(frame)
        frames_acc.clear()
        emit(dict(
            terminal,
            conversation_id=msg["conversation_id"], message_id=message_id,
            text=appended, content=new_content, continued=True, continuation_noop=False,
        ))
    finally:
        frames_acc.clear()
        if not attempt_resolved:
            # Any unexpected post-generation failure leaves durable content true
            # but makes the process-local pin attempt definitively unavailable.
            try:
                lens.unpublish_gen(
                    attempt_gen_id,
                    generation_run_id=attempt_run_id,
                    model_session_id=lens.model_session_id,
                    lens_binding_id=lens.binding_id,
                )
            except BaseException:
                pass
            try:
                discard_attempt()
            except BaseException:
                attempt_resolved = True
            if durable_committed:
                _resolve_continuation_pin_unavailable_total(
                    message_id,
                    expected_version=durable_version,
                    generation_run_id=attempt_run_id,
                    gen_id=attempt_gen_id,
                )


async def _watch_stop(ws, stop_event):
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "stop":
                stop_event.set()
    finally:
        stop_event.set()


def _make_ws_generation_worker(dispatch, stop_event, loop, queue):
    def emit(frame):
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    def worker():
        def execute(payload):
            request = payload["request"]
            context = payload["context"]
            if "messages" in request:
                _generate_safely(request["messages"], request.get("sampling", {}), stop_event, emit, context)
            else:
                _persisted_generate(request, stop_event, emit, context)
        return _execute_dispatched_worker(dispatch, execute)
    return worker


async def _prepare_ws_worker_handoff(handoff, req, loop, queue, stop_event):
    token = snap = context = payload = dispatch = task = None
    try:
        token, snap = handoff.token, handoff.snapshot
        context = _captured_generation_context(token, snap, stop_event)
        if not req.get("lens"):
            context = GenerationContext(
                token=context.token, model_session_id=context.model_session_id, bundle=context.bundle,
                hf_model=context.hf_model, tokenizer=context.tokenizer, jl=context.jl, meta=context.meta,
                capability_profile=context.capability_profile, lens=None,
                intervention_snapshot=context.intervention_snapshot, stop_event=context.stop_event,
            )
        payload = {"context": context, "request": dict(req)}
        try:
            dispatch = handoff.create_dispatch(payload)
        finally:
            payload = None
        snap = context = None
        task = await handoff.create_thread_task(
            _make_ws_generation_worker,
            factory_args=(dispatch, stop_event, loop, queue),
        )
        return HandoffSetup(task=task, dispatch=dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        kind, message = _bounded_failure(exc, "websocket generation setup failed")
        return HandoffSetup(status=500, kind=kind, message=message)
    finally:
        token = snap = context = payload = dispatch = task = None


async def _setup_ws_worker(req):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stop_event = ThreadingEvent()
    setup = None
    try:
        async with OperationHandoff.acquire_scope(
            manager.coordinator, OperationType.GENERATE, stop_event=stop_event, requires_loaded=True,
        ) as active_handoff:
            setup = await _prepare_ws_worker_handoff(
                active_handoff, req, loop, queue, stop_event,
            )
        if setup.status is not None:
            return ("error", setup.status, setup.kind, setup.message)
        return ("ok", setup.task, queue, stop_event, setup.dispatch)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        status = 409 if isinstance(exc, OperationConflict) else 422 if isinstance(exc, ModelStateError) else 500
        kind, message = _bounded_failure(exc, "websocket generation setup failed")
        return ("error", status, kind, message)


async def _cancel_task_uninterruptibly(task):
    if task is None:
        return False
    cancelled = False
    cancel_requested = False
    while not task.done() and not cancel_requested:
        try:
            cancel_requested = bool(task.cancel())
        except asyncio.CancelledError:
            cancelled = True
            _remember_task_cancellation()
        except BaseException:
            logging.getLogger(__name__).exception(
                "task cancellation failed during websocket cleanup"
            )
        if not task.done() and not cancel_requested:
            cancelled |= await _cleanup_yield()
    while not task.done():
        try:
            await _drain_worker_uninterruptibly(task)
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
        except BaseException:
            # The task remains strongly owned here.  A broken drain helper is
            # not evidence that receiver/waiter cleanup completed.
            logging.getLogger(__name__).exception("task drain failed during websocket cleanup")
            cancelled |= await _cleanup_yield()
    return cancelled


async def _drain_retained_worker_total(worker, dispatch):
    """Drain a transferred worker and prove its coordinator token is resolved."""
    cancelled = False
    first_drain = True
    while first_drain or not worker.done():
        first_drain = False
        try:
            await _drain_worker_uninterruptibly(worker)
        except asyncio.CancelledError:
            cancelled = True
            _remember_task_cancellation()
        except BaseException:
            logging.getLogger(__name__).exception("websocket worker drain helper failed")
            cancelled |= await _cleanup_yield()
    coordinator = getattr(dispatch, "coordinator", None)
    token = getattr(dispatch, "token", None)
    if coordinator is None or token is None:
        return cancelled
    while True:
        try:
            current = coordinator.is_current(token)
        except BaseException:
            current = True
        if not current:
            break
        if dispatch.claimed:
            _release_worker_total(dispatch)
        else:
            _abort_unclaimed_total(dispatch)
        cancelled |= await _cleanup_yield()
    return cancelled


async def _finalize_ws_runtime(ws, worker, receiver, waiter, stop_event, dispatch, terminal_seen):
    cancelled = False
    outcome = None
    failure_message = None
    try:
        stop_event.set()
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            cancelled = True
            _remember_task_cancellation()
        logging.getLogger(__name__).exception("failed to set websocket generation stop event")
    try:
        cancelled |= await _cancel_task_uninterruptibly(waiter)
    except BaseException:
        logging.getLogger(__name__).exception("websocket waiter cleanup failed")
    try:
        cancelled |= await _cancel_task_uninterruptibly(receiver)
    except BaseException:
        logging.getLogger(__name__).exception("websocket receiver cleanup failed")
    try:
        dispatch.cancel_from_awaiter()
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            cancelled = True
            _remember_task_cancellation()
        logging.getLogger(__name__).exception("websocket dispatch cancellation failed")
    cancelled |= await _drain_retained_worker_total(worker, dispatch)
    raw_failure = None
    if worker.cancelled():
        raw_failure = "generation worker was cancelled without a terminal frame"
    elif worker.done():
        try:
            outcome = worker.result()
        except BaseException as exc:
            kind, message = _bounded_failure(exc, "generation worker failed", 384)
            raw_failure = f"{kind}: {message}"[:512]
    if not terminal_seen:
        if isinstance(outcome, WorkerOutcome) and not outcome.ok:
            failure_message = outcome.failure_message or "generation worker failed"
        elif isinstance(outcome, WorkerOutcome) and outcome.ok:
            failure_message = "generation worker completed without a terminal frame"
        elif raw_failure is not None:
            failure_message = raw_failure
        else:
            failure_message = "generation worker returned an invalid result without a terminal frame"
    else:
        failure_message = None
    worker = receiver = waiter = dispatch = outcome = None
    if failure_message is not None:
        try:
            await _ws_send(ws, json.dumps({"type": "error", "message": failure_message[:512]}))
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
        except BaseException as exc:
            try:
                exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            except BaseException:
                pass
            exc = None
    failure_message = None
    if cancelled:
        raise asyncio.CancelledError()


async def _run_chat(ws, req):
    setup = await _setup_ws_worker(req)
    if setup[0] == "error":
        await _ws_send(ws, json.dumps({"type": "error", "message": setup[3]}))
        return
    _, worker, queue, stop_event, dispatch = setup
    setup = None
    receiver_coro = receiver = waiter_coro = waiter = None
    terminal_seen = False
    try:
        receiver_coro = _watch_stop(ws, stop_event)
        receiver = asyncio.create_task(receiver_coro)
        receiver_coro = None
        while True:
            waiter_coro = queue.get()
            waiter = asyncio.create_task(waiter_coro)
            waiter_coro = None
            done, _pending = await asyncio.wait(
                {waiter, receiver, worker}, return_when=asyncio.FIRST_COMPLETED,
            )
            if waiter in done:
                frame = waiter.result()
                waiter = None
                await _ws_send(ws, json.dumps(frame))
                if frame["type"] in ("done", "error"):
                    terminal_seen = True
                    break
            elif receiver in done:
                break
            elif worker in done:
                break
    finally:
        if waiter_coro is not None and hasattr(waiter_coro, "close"):
            try:
                waiter_coro.close()
            except BaseException:
                logging.getLogger(__name__).exception("failed to close unsubmitted websocket waiter")
        if receiver_coro is not None and hasattr(receiver_coro, "close"):
            try:
                receiver_coro.close()
            except BaseException:
                logging.getLogger(__name__).exception("failed to close unsubmitted websocket receiver")
        try:
            await _finalize_ws_runtime(
                ws, worker, receiver, waiter, stop_event, dispatch, terminal_seen,
            )
        finally:
            worker = receiver_coro = receiver = waiter_coro = waiter = queue = dispatch = None



@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_locks[ws] = asyncio.Lock()
    try:
        while True:
            req = json.loads(await ws.receive_text())
            if req.get("type") != "chat":
                continue
            await _run_chat(ws, req)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _ws_locks.pop(ws, None)


if config.UI_DIST.exists():
    app.mount("/", StaticFiles(directory=config.UI_DIST, html=True), name="ui")
