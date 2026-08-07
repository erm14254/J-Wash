import json
import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.ablation import HookAttachment
from core.editing import PublicationCancelled, PublicationGate
from core.model_session import (
    LoadedModelBundle,
    ModelSessionCoordinator,
    OperationConflict,
    OperationType,
)


class Handle:
    def __init__(self, seen, name, fail=False):
        self.seen = seen
        self.name = name
        self.fail = fail
        self.removed = 0

    def remove(self):
        self.removed += 1
        self.seen.append(self.name)
        if self.fail:
            raise RuntimeError(self.name)


def test_generation_versus_unload_conflict_event_driven():
    c = ModelSessionCoordinator()
    load, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(load, __import__("core.model_session", fromlist=["LoadedModelBundle"]).LoadedModelBundle.from_parts(object(), object(), object(), {}, None), expected_unloaded_session=snap.model_session_id)
    c.release(load)
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def worker():
        token, _ = c.acquire(OperationType.GENERATE, requires_loaded=True)
        acquired.set()
        assert release.wait(2), "generation release phase blocked"
        c.release(token)

    thread = threading.Thread(target=worker)
    thread.start()
    assert acquired.wait(2), "generation acquisition phase blocked"
    try:
        c.acquire(OperationType.UNLOAD)
    except OperationConflict as exc:
        errors.append(str(exc))
    release.set()
    thread.join(2)
    assert not thread.is_alive(), "generation worker did not terminate after release phase"
    assert errors and "generate" in errors[0]


def test_cancellation_before_and_after_claim():
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    assert c.release(token)
    assert not c.request_cancel(token)

    token, _ = c.acquire(OperationType.GENERATE)
    assert c.request_cancel(token)
    assert c.snapshot().operation.cancellation_requested is True
    assert c.release(token)


def test_hook_attachment_owns_only_its_handles_and_is_idempotent():
    seen = []
    a1 = Handle(seen, "a1")
    a2 = Handle(seen, "a2")
    b = Handle(seen, "b")
    attach_a = HookAttachment([a1, a2])
    attach_b = HookAttachment([b])
    attach_a.close()
    attach_a.close()
    assert seen == ["a1", "a2"]
    assert b.removed == 0
    attach_b.close()
    assert seen == ["a1", "a2", "b"]


def test_hook_attachment_attempts_every_removal():
    seen = []
    attachment = HookAttachment([Handle(seen, "first", fail=True), Handle(seen, "second")])
    try:
        attachment.close()
    except RuntimeError as exc:
        assert str(exc) == "first"
    assert seen == ["first", "second"]


def test_generation_cleanup_preserves_coordinator_and_successor_owner():
    c = ModelSessionCoordinator()
    before = c
    token, _ = c.acquire(OperationType.GENERATE)
    assert c.release(token)
    successor, _ = c.acquire(OperationType.UNLOAD)
    # A stale generation finalizer must not clear a successor or replace the coordinator.
    assert c.release(token) is False
    assert before is c
    assert c.snapshot().operation.id == successor.id
    assert c.release(successor)


def test_dispatch_cancellation_before_claim_releases_without_worker_ownership():
    from core.model_session import WorkerDispatch
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    dispatch = WorkerDispatch(c, token)
    assert dispatch.cancel_from_awaiter() is True
    assert dispatch.claim() is False
    assert c.snapshot().operation is None


def test_dispatch_cancellation_after_claim_leaves_release_to_worker():
    from core.model_session import WorkerDispatch
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    dispatch = WorkerDispatch(c, token)
    assert dispatch.claim() is True
    assert dispatch.cancel_from_awaiter() is False
    assert c.snapshot().operation.id == token.id
    assert c.snapshot().operation.cancellation_requested is True
    assert dispatch.release_from_worker() is True


def test_operation_handoff_factory_failure_releases_unclaimed_dispatch():
    from api import app
    from core.model_session import WorkerDispatch

    async def scenario():
        c = ModelSessionCoordinator()
        token, _ = c.acquire(OperationType.GENERATE)
        stop_event = threading.Event()
        dispatch = WorkerDispatch(c, token, stop_event)
        handoff = app.OperationHandoff(c, token, stop_event)
        handoff.set_dispatch(dispatch)
        with pytest.raises(RuntimeError):
            await handoff.create_thread_task(lambda: (_ for _ in ()).throw(RuntimeError("factory")))
        assert c.snapshot().operation is None
        successor, _ = c.acquire(OperationType.UNLOAD)
        assert c.release(successor)

    asyncio.run(scenario())


def test_operation_handoff_closes_unsubmitted_to_thread_wrapper(monkeypatch):
    from api import app
    from core.model_session import WorkerDispatch

    async def scenario():
        c = ModelSessionCoordinator()
        token, _ = c.acquire(OperationType.GENERATE)
        stop_event = threading.Event()
        dispatch = WorkerDispatch(c, token, stop_event)
        handoff = app.OperationHandoff(c, token, stop_event)
        handoff.set_dispatch(dispatch)
        wrapper_closed = {"value": False}

        class Wrapper:
            def close(self):
                wrapper_closed["value"] = True

        monkeypatch.setattr(app.asyncio, "to_thread", lambda worker: Wrapper())
        def fail_create_task(_wrapper):
            raise RuntimeError("task creation failed")
        monkeypatch.setattr(app.asyncio, "create_task", fail_create_task)

        with pytest.raises(RuntimeError):
            await handoff.create_thread_task(lambda: (lambda: None))
        assert wrapper_closed["value"]
        assert c.snapshot().operation is None

    asyncio.run(scenario())


def test_handoff_awaiter_scope_closes_on_base_exception():
    from api import app

    async def scenario():
        coordinator = ModelSessionCoordinator()
        handoff = app.OperationHandoff(coordinator, stop_event=threading.Event())
        handoff.acquire(OperationType.MODEL_DELETE, include_bundle=False)
        with pytest.raises(KeyboardInterrupt):
            async with handoff.awaiter_scope():
                raise KeyboardInterrupt("setup failed")
        assert handoff.closed
        assert coordinator.snapshot().operation is None
        successor, _ = coordinator.acquire(OperationType.MODEL_DELETE, include_bundle=False)
        coordinator.release(successor)

    asyncio.run(scenario())


def test_handoff_acquire_scope_owns_cleanup_before_snapshot_publication():
    from api import app

    async def scenario():
        coordinator = ModelSessionCoordinator()
        with pytest.raises(RuntimeError):
            async with app.OperationHandoff.acquire_scope(
                coordinator, OperationType.MODEL_DELETE, include_bundle=False,
            ) as lease:
                assert lease.snapshot is not None
                raise RuntimeError("setup")
        assert coordinator.snapshot().operation is None

    asyncio.run(scenario())


def test_handoff_cleanup_task_creation_failure_falls_back_inline(monkeypatch):
    from api import app

    async def scenario():
        coordinator = ModelSessionCoordinator()
        real_create_task = app.asyncio.create_task

        def fail_once(coro):
            monkeypatch.setattr(app.asyncio, "create_task", real_create_task)
            raise RuntimeError("task allocation")

        with pytest.raises(RuntimeError):
            async with app.OperationHandoff.acquire_scope(
                coordinator, OperationType.MODEL_DELETE, include_bundle=False,
            ):
                monkeypatch.setattr(app.asyncio, "create_task", fail_once)
                raise RuntimeError("primary")
        assert coordinator.snapshot().operation is None

    asyncio.run(scenario())


def test_ws_setup_conflict_before_scope_target_assignment_is_model_free(monkeypatch):
    from contextlib import asynccontextmanager
    from api import app

    @asynccontextmanager
    async def fail_before_yield(*_args, **_kwargs):
        raise OperationConflict("busy")
        yield

    monkeypatch.setattr(app.OperationHandoff, "acquire_scope", fail_before_yield)
    result = asyncio.run(app._setup_ws_worker({"messages": []}))
    assert result == ("error", 409, "OperationConflict", "busy")


def test_ws_setup_cancellation_before_scope_target_assignment_is_reraised(monkeypatch):
    from contextlib import asynccontextmanager
    from api import app

    @asynccontextmanager
    async def cancel_before_yield(*_args, **_kwargs):
        raise asyncio.CancelledError()
        yield

    monkeypatch.setattr(app.OperationHandoff, "acquire_scope", cancel_before_yield)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app._setup_ws_worker({"messages": []}))


def test_worker_drain_closes_unsubmitted_drain_coroutine(monkeypatch):
    from api import app

    async def scenario():
        worker = asyncio.create_task(asyncio.sleep(0))
        captured = {}

        def fail_create_task(coro):
            captured["coro"] = coro
            raise MemoryError("drain task allocation failed")

        monkeypatch.setattr(app.asyncio, "create_task", fail_create_task)
        result = await app._drain_worker_uninterruptibly(worker)
        assert result == [None]
        assert captured["coro"].cr_frame is None

    asyncio.run(scenario())


def test_ws_finalizer_continues_after_early_cleanup_base_exception(monkeypatch):
    from api import app

    async def scenario():
        worker = asyncio.create_task(asyncio.sleep(0, result=app.WorkerOutcome()))
        await worker
        phases = []

        async def fail_task_cleanup(_task):
            phases.append("task_cleanup")
            raise KeyboardInterrupt("cleanup injection")

        async def record_drain(task):
            phases.append("worker_drain")
            return await asyncio.gather(task, return_exceptions=True)

        class Dispatch:
            def cancel_from_awaiter(self):
                phases.append("dispatch_cancel")

        monkeypatch.setattr(app, "_cancel_task_uninterruptibly", fail_task_cleanup)
        monkeypatch.setattr(app, "_drain_worker_uninterruptibly", record_drain)
        await app._finalize_ws_runtime(
            object(), worker, object(), object(), threading.Event(), Dispatch(), True,
        )
        assert phases.count("task_cleanup") == 2
        assert phases[-2:] == ["dispatch_cancel", "worker_drain"]

    asyncio.run(scenario())


def test_dispatched_routes_use_handoff_scope_and_factory_descriptors():
    import inspect
    from api import app

    routes = (
        app.api_delete_model, app.api_lens_load, app.api_edit_export,
        app.api_edit_export_gguf, app.api_generate_sync,
        app.api_token_neighbors, app.api_lens_pin, app._setup_ws_worker,
    )
    for route in routes:
        source = inspect.getsource(route)
        assert "OperationHandoff.acquire_scope(" in source
        assert "handoff.acquire(" not in source
        assert "async with handoff.awaiter_scope()" not in source
        assert "manager.coordinator.acquire(" not in source
        assert "manager.coordinator.release(" not in source
        assert "create_thread_task(lambda" not in source


def test_pin_missing_run_identity_is_rejected_before_handoff(monkeypatch):
    from api import app

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("coordinator acquisition must not occur")

    monkeypatch.setattr(app.manager.coordinator, "acquire", forbidden)
    req = app.PinRequest(gen_id=1, token_ids=[1], generation_run_id=None)
    with pytest.raises(app.HTTPException) as raised:
        asyncio.run(app.api_lens_pin(req))
    assert raised.value.status_code == 409
    assert not called


@pytest.mark.parametrize("run_id", [None, "", "A" * 32, "0" * 31, "g" * 32, 123])
def test_pin_malformed_run_identity_is_rejected_before_handoff(monkeypatch, run_id):
    from api import app

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("coordinator acquisition must not occur")

    monkeypatch.setattr(app.manager.coordinator, "acquire", forbidden)
    req = app.PinRequest.model_construct(gen_id=1, token_ids=[1], generation_run_id=run_id)
    with pytest.raises(app.HTTPException) as raised:
        asyncio.run(app.api_lens_pin(req))
    assert raised.value.status_code == 409
    assert not called


def test_publication_gate_cancellation_wins_before_rename():
    published = []
    gate = PublicationGate(lambda: True)
    assert gate.cancel() is True
    try:
        gate.publish(lambda: published.append("renamed"))
    except PublicationCancelled:
        pass
    assert published == []
    assert gate.state == "cancelled-before-publication"


def test_publication_gate_rename_wins_before_cancellation():
    published = []
    gate = PublicationGate(lambda: True)
    gate.publish(lambda: published.append("renamed"))
    assert gate.cancel() is False
    assert published == ["renamed"]
    assert gate.state == "published"


def test_dispatch_payload_cleared_on_cancellation_before_claim():
    import weakref
    from core.model_session import WorkerDispatch

    class Payload:
        pass

    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    payload = Payload()
    ref = weakref.ref(payload)
    dispatch = WorkerDispatch(c, token, payload=payload)
    del payload
    assert dispatch.cancel_from_awaiter() is True
    import gc
    gc.collect()
    assert ref() is None
    assert c.snapshot().operation is None


def test_intervention_update_invalid_replace_is_transactional():
    import pytest
    from core.ablation import Interventions

    iv = Interventions()
    iv._rules = [{
        "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
        "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
        "dirs_a": {0: object()}, "dirs_b": None,
    }]
    iv._revision = 1
    before = iv.state_record()
    with pytest.raises(ValueError, match="replacement_id required"):
        iv.update(1, mode="replace", lens_manager=type("LM", (), {"lens": object()})(), jl=type("JL", (), {"tokenizer": object(), "layers": [object()], "_lm_head": None})())
    assert iv.state_record() == before


def test_lens_cleanup_clears_real_withdrawn_container_before_cuda_cleanup(monkeypatch):
    import weakref
    import gc
    import core.lens_manager as lens_module
    from core.lens_manager import LensManager

    class TensorLike:
        pass

    lens = LensManager()
    live = TensorLike()
    lens.lens = TensorLike()
    lens.mask = live
    lens._J = TensorLike()
    old = lens.withdraw()
    mask_ref = weakref.ref(live)
    seen = {}

    def fake_empty_cache():
        seen["during_cuda_keys"] = list(old.keys())

    monkeypatch.setattr(lens_module.torch.cuda, "empty_cache", fake_empty_cache)
    err = lens.cleanup_withdrawn(old)
    del live
    gc.collect()
    assert err is None
    assert old == {}
    assert seen["during_cuda_keys"] == []
    assert mask_ref() is None


def test_model_transition_cleanup_does_not_restore_rules_after_publication_failure(monkeypatch):
    from api import app

    previous = app.interventions.state_record()
    try:
        app.interventions._rules = [{
            "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
            "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
            "dirs_a": {0: object()}, "dirs_b": None,
        }]
        app.interventions._revision = 1
        app.interventions._scale = 2.5
        app.interventions._mode = "readthrough"

        prepared = app._prepare_model_transition_interventions(42)
        assert prepared["rules"] == []
        app._cleanup_model_bound_state(42, token=object())
        snap = app.interventions.snapshot()
        assert snap["rules"] == []
        assert snap["active_rules"] == []
        assert snap["scale"] == 2.5
        assert snap["mode"] == "standard"
    finally:
        app.interventions.restore_state_record(previous)


def test_http_generation_with_coordinated_mappingproxy_intervention(monkeypatch):
    import asyncio
    from types import MappingProxyType, SimpleNamespace
    from api import app

    class Layer:
        def __init__(self):
            self.handles = []

        def register_forward_hook(self, hook):
            handle = SimpleNamespace(remove=lambda: self.handles.remove(handle))
            self.handles.append(handle)
            return handle

    class FakeJL:
        def __init__(self):
            self.layers = [Layer()]
            self._jwash_declared_quant = None

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    jl = FakeJL()
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(object(), object(), jl, {"model_id": "m", "quant": None}, {"ok": True}),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    direction = object()
    rule = {
        "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
        "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
        "dirs_a": {0: direction}, "dirs_b": None,
    }
    coordinator.bootstrap_interventions_for_test({
        "revision": 7,
        "model_session_id": coordinator.status_snapshot().model_session_id,
        "lens_binding_id": None,
        "scale": 1.0,
        "mode": "standard",
        "rules": [rule],
        "active_rules": [rule],
        "summary": [rule],
        "active_summary": [rule],
    })

    captured = {}

    def fake_generate(_messages, _sampling, _stop_event, emit, lens=None, ablator=None):
        attachment = ablator.attach(lens.jl, snapshot=lens.intervention_snapshot)
        captured["snapshot_type"] = type(lens.intervention_snapshot)
        captured["direction_identity"] = lens.intervention_snapshot["rules"][0]["dirs_a"][0]
        attachment.close()
        emit({"type": "done", "text": "ok", "stats": {"tokens": 0}})

    monkeypatch.setattr(app.manager, "generate", fake_generate)
    result = asyncio.run(app.api_generate_sync(SimpleNamespace(messages=[{"role": "user", "content": "hi"}], sampling={})))
    assert result["text"] == "ok"
    assert captured["snapshot_type"] is MappingProxyType
    assert captured["direction_identity"] is direction
    assert jl.layers[0].handles == []


def test_http_generation_metadata_preserves_captured_intervention_provenance(monkeypatch):
    import asyncio
    import torch
    from types import SimpleNamespace
    from api import app
    from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType

    class FakeTokenizer:
        chat_template = ""
        unk_token_id = -1

        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1]])

        def decode(self, ids, skip_special_tokens=True):
            return "ok" if ids else ""

        def encode(self, *args, **kwargs):
            return [1]

        def convert_tokens_to_ids(self, _token):
            return -1

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=2)

        def __init__(self):
            self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=1, hidden_size=4))
            self.emb = SimpleNamespace(weight=torch.zeros(4, 4))

        def get_input_embeddings(self):
            return self.emb

        def __call__(self, **kwargs):
            logits = torch.zeros(1, 1, 5)
            logits[0, 0, 3] = 10
            return SimpleNamespace(logits=logits, past_key_values=None)

    class Layer:
        def register_forward_hook(self, hook):
            return SimpleNamespace(remove=lambda: None)

    class FakeJL:
        _jwash_declared_quant = None
        layers = [Layer()]

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    monkeypatch.setattr(app.lens_manager, "snapshot_for_generation", lambda: None)
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(
            FakeModel(), FakeTokenizer(), FakeJL(), {"model_id": "fake", "chat_template_fallback": False}, {"ok": True}
        ),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    session_id = coordinator.status_snapshot().model_session_id
    coordinator.bootstrap_interventions_for_test({
        "revision": 9,
        "model_session_id": session_id,
        "lens_binding_id": 3,
        "scale": 2.0,
        "mode": "readthrough",
        "rules": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0, "layers": [], "enabled": False}],
        "active_rules": [],
        "summary": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0, "layers": [], "enabled": False}],
        "active_summary": [],
    })

    result = asyncio.run(app.api_generate_sync(SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        sampling={"max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 0, "seed": -1},
    )))
    provenance = result["meta"]["intervention_provenance"]
    assert provenance["revision"] == 9
    assert provenance["mode"] == "readthrough"
    assert provenance["scale"] == 2.0
    assert provenance["model_session_id"] == session_id
    assert provenance["lens_binding_id"] == 3
    assert provenance["summary"][0]["id"] == 1
    assert provenance["active_summary"] == []
    assert result["last_generation"]["meta"]["intervention_provenance"] == provenance


def test_fallback_generation_emits_authoritative_replacement_character(monkeypatch):
    import threading
    import torch
    from types import SimpleNamespace
    from api import app

    class Tokenizer:
        chat_template = ""
        unk_token_id = -1
        all_special_ids = ()

        def apply_chat_template(self, *_args, **_kwargs):
            return torch.tensor([[1]])

        def decode(self, ids, **_kwargs):
            return "\ufffd" if list(ids) else ""

        def encode(self, *_args, **_kwargs):
            return [1]

        def convert_tokens_to_ids(self, _token):
            return -1

    class Model:
        generation_config = SimpleNamespace(eos_token_id=2)

        def __init__(self):
            self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=1, hidden_size=4))
            self.emb = SimpleNamespace(weight=torch.zeros(4, 4))

        def get_input_embeddings(self):
            return self.emb

        def __call__(self, **_kwargs):
            logits = torch.zeros(1, 1, 5)
            logits[0, 0, 3] = 10
            return SimpleNamespace(logits=logits, past_key_values=None)

    monkeypatch.setattr(app.manager, "hf_model", Model())
    monkeypatch.setattr(app.manager, "tokenizer", Tokenizer())
    monkeypatch.setattr(app.manager, "jl", SimpleNamespace(layers=[]))
    monkeypatch.setattr(
        app.manager, "meta", {"model_id": "fallback", "chat_template_fallback": True}
    )
    emitted = []
    app.manager.generate(
        [{"role": "user", "content": "hi"}],
        {"max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 0, "seed": 1},
        threading.Event(), emitted.append,
    )

    tokens = [event["text"] for event in emitted if event["type"] == "token"]
    done = next(event for event in emitted if event["type"] == "done")
    assert tokens == ["\ufffd"]
    assert "".join(tokens) == done["text"] == "\ufffd"
    assert done["durable_reply_token_ids"] == [3]
    assert done["durable_generated_token_count"] == 1


def test_persisted_continue_records_segment_provenance(monkeypatch):
    import json
    import threading

    from types import SimpleNamespace
    from api import app

    class FakeStore:
        def __init__(self):
            self.meta = {
                "intervention_provenance": {"revision": 1, "mode": "standard", "scale": 1.0},
                "model_session_id": 1,
            }
            self.updated = None

        def get_message(self, message_id):
            return {
                "id": message_id,
                "conversation_id": 99,
                "parent_id": 1,
                "role": "assistant",
                "content": "old",
                "meta": json.dumps(self.meta),
                "frames_file": None,
            }

        def path_to_root(self, message_id):
            return [{"role": "user", "content": "u"}, {"role": "assistant", "content": "old"}]

        def update_message_and_frames_if_unchanged(
            self, message_id, expected_version, content, meta=None, **kwargs
        ):
            assert expected_version == 0
            assert kwargs.get("frames") is None
            self.updated = (message_id, content, meta)
            return SimpleNamespace(state="committed", entity_id=str(message_id))

        def save_frames(self, *args, **kwargs):
            raise AssertionError("no lens frames expected")

    continuation_meta = {
        "intervention_provenance": {
            "revision": 2,
            "mode": "readthrough",
            "scale": 1.5,
            "model_session_id": 4,
            "lens_binding_id": 7,
            "summary": [{"id": 1}],
            "active_summary": [{"id": 1}],
        },
        "model_session_id": 4,
        "lens": None,
    }

    def fake_generate(**kwargs):
        kwargs["emit"]({"type": "done", "text": " new", "stats": {"tokens": 2}, "stopped": False, "meta": continuation_meta})

    fake_store = FakeStore()
    monkeypatch.setattr(app, "store", fake_store)
    monkeypatch.setattr(app.manager, "generate", fake_generate)
    emitted = []
    context = SimpleNamespace(lens=None, intervention_snapshot={"rules": []})

    app._persisted_continue({}, 10, threading.Event(), emitted.append, context)

    assert fake_store.updated[1] == "old new"
    saved_meta = fake_store.updated[2]
    assert saved_meta["intervention_provenance"]["revision"] == 2
    assert saved_meta["intervention_provenance"]["mode"] == "readthrough"
    assert saved_meta["intervention_provenance"]["scale"] == 1.5
    assert saved_meta["continuations"][0]["start_offset"] == 0
    assert saved_meta["continuations"][0]["end_offset"] == 3
    assert saved_meta["continuations"][0]["meta"]["intervention_provenance"]["revision"] == 1
    assert saved_meta["continuations"][-1]["start_offset"] == 3
    assert saved_meta["continuations"][-1]["end_offset"] == 7
    assert saved_meta["continuations"][-1]["meta"]["intervention_provenance"]["revision"] == 2
    assert emitted[-1]["meta"]["intervention_provenance"]["revision"] == 2
    assert emitted[-1]["text"] == " new"
    assert emitted[-1]["content"] == "old new"
    assert emitted[-1]["continued"] is True
    assert emitted[-1]["continuation_noop"] is False


def test_zero_length_continuation_does_not_reattribute_existing_text(monkeypatch):
    import json
    import threading

    from types import SimpleNamespace
    from api import app

    class FakeStore:
        def __init__(self):
            self.meta = {"intervention_provenance": {"revision": 1}}
            self.updated = None
        def get_message(self, message_id):
            return {"id": message_id, "conversation_id": 1, "parent_id": None, "role": "assistant", "content": "old", "meta": json.dumps(self.meta), "frames_file": None, "version": 0}
        def path_to_root(self, message_id):
            return [{"role": "assistant", "content": "old"}]
        def update_message_and_frames_if_unchanged(self, message_id, expected_version, content, meta=None, **kwargs):
            assert expected_version == 0
            self.updated = (content, meta)
            return SimpleNamespace(state="committed", entity_id=str(message_id))

    def fake_generate(**kwargs):
        kwargs["emit"]({"type": "frame", "pos": 1, "gen": 9, "generation_run_id": "b" * 32})
        kwargs["emit"]({
            "type": "done", "text": "", "stats": {"tokens": 0}, "stopped": True,
            "meta": {"intervention_provenance": {"revision": 2}}, "gen_id": 9,
            "generation_run_id": "b" * 32, "generated_token_start_pos": 1,
            "generated_token_end_pos": 1, "continuation_suffix_start_pos": 1,
            "continuation_suffix_end_pos": 1,
        })

    fake_store = FakeStore()
    monkeypatch.setattr(app, "store", fake_store)
    monkeypatch.setattr(app.manager, "generate", fake_generate)
    emitted = []
    app._persisted_continue({}, 1, threading.Event(), emitted.append, SimpleNamespace(lens=None, intervention_snapshot={"rules": []}))
    assert fake_store.updated is None
    assert emitted[-1]["text"] == ""
    assert emitted[-1]["content"] == "old"
    assert emitted[-1]["continued"] is False
    assert emitted[-1]["continuation_noop"] is True
    assert [frame["type"] for frame in emitted] == ["done"]
    for key in (
        "gen_id", "generation_run_id", "generated_token_start_pos", "generated_token_end_pos",
        "continuation_suffix_start_pos", "continuation_suffix_end_pos", "stats", "meta",
    ):
        assert key not in emitted[-1]


def test_lens_generation_provisional_finalize_prunes_and_publishes():
    import torch
    from core.lens_manager import LensManager

    lens = LensManager()
    lens.layers = [0]
    lens.model_session_id = 4
    lens.binding_id = 7
    gen_id = lens.start_gen(generation_run_id="a" * 32, provisional=True)
    store = lens._provisional_gen_store[gen_id]
    store["positions"] = [0, 10, 11, 12]
    store["token_ids"] = [1, 20, 21, 22]
    store["phases"] = ["reading", "thinking", "thinking", "thinking"]
    store["residuals"][0] = [torch.arange(8, dtype=torch.float16).reshape(4, 2)]

    assert gen_id not in lens.gen_store
    lens.finalize_gen(gen_id, retained_thinking=[(10, 20), (12, 22)], publish=False)
    assert gen_id not in lens.gen_store
    assert store["positions"] == [0, 10, 12]
    assert store["token_ids"] == [1, 20, 22]
    assert store["phases"] == ["reading", "thinking", "thinking"]
    assert store["residuals"][0][0].tolist() == [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0]]

    lens.finalize_gen(gen_id, publish=True)
    assert gen_id in lens.gen_store
    assert gen_id not in lens._provisional_gen_store


def test_provisional_generation_does_not_evict_committed_store():
    from core.lens_manager import GEN_STORE_MAX, LensManager

    lens = LensManager()
    committed = [lens.start_gen(generation_run_id=f"{index:032x}") for index in range(GEN_STORE_MAX)]
    attempt = lens.start_gen(generation_run_id="f" * 32, provisional=True)
    assert list(lens.gen_store) == committed
    assert attempt not in lens.gen_store
    with pytest.raises(ValueError, match="unknown generation"):
        lens.pin_ranks(attempt, [1], SimpleNamespace(), generation_run_id="f" * 32)
    assert lens.discard_gen(attempt) is True
    assert lens.discard_gen(attempt) is False
    assert list(lens.gen_store) == committed


def test_preset_save_persists_coordinated_provenance(monkeypatch):
    from api import app
    from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(object(), object(), object(), {"model_id": "m", "revision": "r1"}, {"ok": True}),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    session_id = coordinator.status_snapshot().model_session_id
    coordinator.bootstrap_interventions_for_test({
        "revision": 12,
        "model_session_id": session_id,
        "lens_binding_id": 44,
        "scale": 1.75,
        "mode": "readthrough",
        "rules": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.1, "layers": [0], "enabled": True}],
        "active_rules": [],
        "summary": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.1, "layers": [0], "enabled": True}],
        "active_summary": [],
    })
    captured = {}

    def save_preset(name, rules, model_id, scale=1.0, **provenance):
        captured.update(name=name, rules=rules, model_id=model_id, scale=scale, **provenance)
        return dict(captured, schema_version=2)

    monkeypatch.setattr(app.editing, "save_preset", save_preset)
    result = app.api_presets_save("preset")
    assert result["schema_version"] == 2
    assert captured["model_id"] == "m"
    assert captured["model_revision"] == "r1"
    assert captured["intervention_mode"] == "readthrough"
    assert captured["intervention_revision"] == 12
    assert captured["model_session_id"] == session_id
    assert captured["lens_binding_id"] == 44
    assert captured["rules"][0]["id"] == 1


def _frame(pos=0, phase="gen", layer_key=0):
    return {
        "type": "frame",
        "phase": phase,
        "pos": pos,
        "token_id": 1,
        "tok": "a",
        "layers": {
            layer_key: {
                "ids": [1], "strs": ["a"], "p": [1.0],
                "m_ids": [2], "m_strs": ["b"], "m_p": [0.5], "m_rank": [1],
            }
        },
    }


def test_store_round_trips_production_frame_phases(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "complete"}).value[0]
    s.save_frames(mid, [_frame(0, "reading", "0"), _frame(1, "thinking", "0")], [0], 1)
    loaded = s.load_frames(mid)
    assert [frame["phase"] for frame in loaded["frames"]] == ["reading", "thinking"]
    body, _ = s.export(cid, fmt="md", include_frames=True)
    assert "2 lens frames" in body
    s._discard_conn()
    reopened = store_mod.Store()
    loaded = reopened.load_frames(mid)
    assert [frame["phase"] for frame in loaded["frames"]] == ["reading", "thinking"]


def test_store_accepts_legacy_frame_phases(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T").value[0]
    s.save_frames(mid, [_frame(0, "prompt"), _frame(1, "gen")], [0], 1)
    loaded = s.load_frames(mid)
    assert [frame["phase"] for frame in loaded["frames"]] == ["prompt", "gen"]


def test_store_accepts_multiple_production_string_layer_keys(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T").value[0]
    frame = _frame(0, "reading", "0")
    frame["layers"]["12"] = dict(frame["layers"]["0"])
    s.save_frames(mid, [frame], [0, 12], 1)
    loaded = s.load_frames(mid)
    assert loaded["layers"] == [0, 12]
    assert set(loaded["frames"][0]["layers"]) == {0, 12}


def test_update_message_identical_stale_writer_does_not_reconcile_as_committed(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "old", return_version=True).value
    mid, version = inserted
    s.update_message(mid, "same", meta={"edited": True}, clear_frames=True, expected_version=version)
    stale = s.update_message(mid, "same", meta={"edited": True}, clear_frames=True, expected_version=version)
    assert stale.state == "stale"
    assert stale.observed_version == version + 1


def test_persisted_continue_merges_existing_production_phase_archive(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "complete"}).value[0]
    lens_meta = {"path": "/lens", "model_id": "m", "model_revision": "r", "fitted_layers_all": [0], "tapped_layers": [0], "k": 1}
    lens = type("Lens", (), {"layers": [0], "k": 1, "meta": lens_meta})()
    descriptor = app._frame_descriptor(lens, [0], 1)
    s.save_frames(mid, [_frame(0, "reading", "0"), _frame(1, "thinking", "0")], [0], 1, frame_descriptor=descriptor)
    emitted = []

    def generate(**kwargs):
        run_id = "1" * 32
        kwargs["emit"]({"type": "frame", **_frame(0, "reading", "0"), "generation_run_id": run_id})
        kwargs["emit"]({"type": "frame", **_frame(2, "thinking", "0"), "generation_run_id": run_id})
        kwargs["emit"]({"type": "done", "text": " plus", "meta": {"model_id": "m"}, "stats": {}, "stopped": False, "generation_run_id": run_id, "continuation_suffix_start_pos": 2, "continuation_suffix_end_pos": 3})

    monkeypatch.setattr(app.manager, "generate", generate)
    context = type("Ctx", (), {"lens": lens, "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)
    loaded = s.load_frames(mid)
    assert [frame["phase"] for frame in loaded["frames"]] == ["reading", "thinking"]
    assert emitted[-1]["type"] == "done"


def test_store_versioned_cas_rejects_stale_continuation_after_patch(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"intervention_provenance": {"revision": 1}}).value[0]
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1).value
    captured = s.get_message(mid)
    assert captured["version"] == 1
    assert (store_mod.FRAMES_DIR / old_frame).exists()

    s.update_message(mid, "edited", meta={"edited": True}, clear_frames=True)
    assert not (store_mod.FRAMES_DIR / old_frame).exists()

    ok = s.update_message_and_frames_if_unchanged(
        mid,
        captured["version"],
        "old suffix",
        {"intervention_provenance": {"revision": 2}},
        frames=[_frame(1)],
        layers=[0],
        k=1,
    )
    assert ok.state in {"stale", "superseded"}
    current = s.get_message(mid)
    assert current["content"] == "edited"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.tmp-*"))
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_store_versioned_frames_commit_uses_unique_file_and_cleans_old_after_commit(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1}).value[0]
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1).value
    captured = s.get_message(mid)
    assert old_frame.startswith(f"{mid}-v1-")

    ok = s.update_message_and_frames_if_unchanged(
        mid,
        captured["version"],
        "old new",
        {"m": 2},
        frames=[_frame(0), _frame(1)],
        layers=[0],
        k=1,
    )
    assert ok.state == "committed"
    updated = s.get_message(mid)
    assert updated["version"] == captured["version"] + 1
    assert updated["frames_file"] != old_frame
    assert updated["frames_file"].startswith(f"{mid}-v{updated['version']}-")
    assert not (store_mod.FRAMES_DIR / old_frame).exists()
    assert (store_mod.FRAMES_DIR / updated["frames_file"]).exists()
    assert len(s.load_frames(mid)["frames"]) == 2


def test_legacy_continued_message_gets_unknown_base_provenance():
    from api import app

    meta = app._continuation_metadata(
        {"continued": True, "intervention_provenance": {"revision": 9}},
        12,
        15,
        {"intervention_provenance": {"revision": 10}},
        {"tokens": 1},
        False,
    )
    assert meta["continuations"][0] == {
        "start_offset": 0,
        "end_offset": 12,
        "provenance": "legacy_mixed_unknown",
        "meta": None,
    }
    assert meta["continuations"][1]["meta"]["intervention_provenance"]["revision"] == 10


def test_preset_lens_descriptor_detects_reused_runtime_binding_id():
    from api import app

    saved = app._stable_lens_descriptor({
        "lens_binding_id": 7,
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": "ra",
        "tapped_layers": [0, 1],
        "k": 8,
    })
    current = app._stable_lens_descriptor({
        "lens_binding_id": 7,
        "repo_id": "lens-b",
        "filename": "lens.pt",
        "revision": "rb",
        "tapped_layers": [0, 1],
        "k": 8,
    })
    same_stable_new_runtime = dict(saved, runtime_lens_binding_id=99, runtime_model_session_id=123)
    assert app._lens_descriptor_mismatch(saved, current)
    assert not app._lens_descriptor_mismatch(saved, same_stable_new_runtime)


def test_preset_lens_descriptor_compares_missing_revision_symmetrically():
    from api import app

    saved = app._stable_lens_descriptor({
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": None,
        "tapped_layers": [0],
        "k": 4,
    })
    current = app._stable_lens_descriptor({
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": "rb",
        "tapped_layers": [0],
        "k": 4,
    })
    assert saved["revision"] is None
    assert app._lens_descriptor_mismatch(saved, current)


def test_store_load_frames_retries_when_old_pointer_is_retired(tmp_path, monkeypatch):
    from pathlib import Path
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1}).value[0]
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1).value
    captured = s.get_message(mid)
    original_read_bytes = Path.read_bytes
    retired = {"done": False}

    def racing_read_bytes(path):
        if path.name == old_frame and not retired["done"]:
            retired["done"] = True
            assert s.update_message_and_frames_if_unchanged(
                mid, captured["version"], "old new", {"m": 2},
                frames=[_frame(0), _frame(1)], layers=[0], k=1,
            ).state == "committed"
            raise FileNotFoundError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    loaded = s.load_frames(mid)
    assert retired["done"]
    assert len(loaded["frames"]) == 2


def test_store_save_frames_detects_stale_initial_attachment(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1}).value[0]
    original = s._write_unique_frame_candidate

    def racing_candidate(message_id, expected_version, blob):
        s.update_message(mid, "edited", meta={"edited": True}, clear_frames=True)
        return original(message_id, expected_version, blob)

    monkeypatch.setattr(s, "_write_unique_frame_candidate", racing_candidate)
    outcome = s.save_frames(mid, [_frame(0)], [0], 1)
    assert outcome.state == "stale"
    current = s.get_message(mid)
    assert current["content"] == "edited"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_persisted_continue_fails_closed_when_base_frames_cannot_load(monkeypatch):
    from api import app

    unchanged = {"content": "old", "version": 3, "updated": False}
    emitted = []

    class FakeStore:
        def get_message(self, message_id):
            return {
                "id": message_id, "conversation_id": 1, "parent_id": None,
                "role": "assistant", "content": unchanged["content"],
                "meta": "{}", "frames_file": "base.msgpack", "version": unchanged["version"],
            }
        def path_to_root(self, message_id):
            return [{"role": "assistant", "content": unchanged["content"]}]
        def load_frames(self, message_id):
            raise app.FrameStorageError("base corrupt")
        def update_message_and_frames_if_unchanged(self, *args, **kwargs):
            unchanged["updated"] = True
            return SimpleNamespace(state="committed", entity_id="1")

    def generate(**kwargs):
        kwargs["emit"]({"type": "frame", **_frame(1)})
        kwargs["emit"]({"type": "done", "text": " new", "meta": {"m": 2}, "stats": {}, "stopped": False})

    monkeypatch.setattr(app, "store", FakeStore())
    monkeypatch.setattr(app.manager, "generate", generate)
    context = type("Ctx", (), {"lens": None, "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, 1, threading.Event(), emitted.append, context)
    assert not unchanged["updated"]
    assert emitted[-1]["type"] == "error"


def test_initial_save_frames_uses_inserted_version_not_later_edit(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "frames_pending"}, return_version=True).value
    mid, version = inserted
    s.update_message(mid, "E", meta={"edited": True}, clear_frames=True)
    outcome = s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version, complete_meta={"publication_state": "complete"})
    assert outcome.state == "stale"
    current = s.get_message(mid)
    assert current["content"] == "E"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_store_save_frames_rolls_back_failed_commit_before_connection_reuse(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "frames_pending"}, return_version=True).value
    mid, version = inserted
    real_conn_method = s._conn
    conn = s._conn()
    fail_once = {"yes": True}

    class CommitFailProxy:
        def __getattr__(self, name):
            return getattr(conn, name)
        @property
        def in_transaction(self):
            return conn.in_transaction
        def execute(self, *args, **kwargs):
            return conn.execute(*args, **kwargs)
        def rollback(self):
            return conn.rollback()
        def commit(self):
            if fail_once["yes"]:
                fail_once["yes"] = False
                raise RuntimeError("commit failed")
            return conn.commit()

    monkeypatch.setattr(s, "_conn", lambda: CommitFailProxy())
    outcome = s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version, complete_meta={"publication_state": "complete"})
    assert outcome.state == "not_committed"
    monkeypatch.setattr(s, "_conn", real_conn_method)
    s.update_message(mid, "after", meta={"ok": True}, clear_frames=True)
    current = s.get_message(mid)
    assert current["content"] == "after"
    assert current["frames_file"] is None


def test_store_candidate_file_fsync_failure_is_actionable(tmp_path, monkeypatch):
    import errno
    import os
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "T", return_version=True).value
    mid, version = inserted

    def fail_fsync(fd):
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    outcome = s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version)
    assert outcome.state == "not_committed"
    assert s.get_message(mid)["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.tmp-*"))
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_delete_conversation_treats_frame_unlink_as_garbage_cleanup(tmp_path, monkeypatch):
    from pathlib import Path
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T").value[0]
    frame_file = s.save_frames(mid, [_frame(0)], [0], 1).value
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if path.name == frame_file:
            raise PermissionError("open reader")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    s.delete_conversation(cid)
    with pytest.raises(ValueError):
        s.get_conversation(cid)
    assert (store_mod.FRAMES_DIR / frame_file).exists()


def test_store_load_frames_corrupt_schema_maps_to_frame_storage_error(tmp_path, monkeypatch):
    import msgpack
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "T", return_version=True).value
    mid, version = inserted
    filename, _ = s._write_unique_frame_candidate(mid, version, msgpack.packb({"version": 1, "frames": [{}]}))
    conn = s._conn()
    conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", (filename, mid))
    conn.commit()
    with pytest.raises(store_mod.FrameStorageError):
        s.load_frames(mid)


def test_mark_frame_publication_failed_uses_exact_version_and_preserves_patch(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(
        cid, None, "assistant", "T",
        meta={"publication_state": "frames_pending", "frames_expected": True},
        return_version=True,
    ).value
    mid, version = inserted
    s.update_message(mid, "E", meta={"edited": True}, clear_frames=True)
    ok = s.mark_frame_publication_failed(
        mid,
        expected_version=version,
        failure_meta={"publication_state": "frames_publication_failed", "frames_expected": True},
    )
    assert ok.state == "superseded"
    current = s.get_message(mid)
    assert current["content"] == "E"
    meta = current["meta"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta == {"edited": True}


def test_mark_frame_publication_failed_marks_pending_without_rewriting_content(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(
        cid, None, "assistant", "T",
        meta={"publication_state": "frames_pending", "frames_expected": True},
        return_version=True,
    ).value
    mid, version = inserted
    ok = s.mark_frame_publication_failed(
        mid,
        expected_version=version,
        failure_meta={
            "publication_state": "frames_publication_failed",
            "frames_expected": True,
            "frames_error_kind": "FrameStorageError",
            "frames_error": "full",
        },
    )
    assert ok.state == "committed"
    current = s.get_message(mid)
    assert current["content"] == "T"
    assert current["frames_file"] is None
    assert current["version"] == version + 1
    meta = current["meta"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta["publication_state"] == "frames_publication_failed"


def test_load_frames_state_taxonomy(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T").value[0]
    with pytest.raises(store_mod.FramesNotAttached):
        s.load_frames(mid)
    conn = s._conn()
    conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", ("missing.msgpack", mid))
    conn.commit()
    with pytest.raises(store_mod.FrameFileMissing):
        s.load_frames(mid)


def test_load_frames_rejects_semantic_archive_corruption(tmp_path, monkeypatch):
    import msgpack
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value

    cases = [
        ("invalid-k", {"k": -1}),
        ("duplicate-layers", {"layers": [0, 0]}),
        ("invalid-phase", {"frames": [{"phase": "bad"}]}),
        ("invalid-layer-key", {"frames": [{"layers": {"01": {}}}]}),
        ("mixed-script-layer-key", {"frames": [{"layers": {"١": {}}}]}),
        ("bytes-layer-key", {"frames": [{"layers": {b"0": {}}}]}),
        ("float-layer-key", {"frames": [{"layers": {0.0: {}}}]}),
        ("bool-layer-key", {"frames": [{"layers": {True: {}}}]}),
        ("layer-mismatch", {"frames": [{"layers": {"12": {}}}]}),
        ("duplicate-normalized-layer", {"frames": [{"layers": {0: {}, "0": {}}}]}),
    ]
    for name, patch in cases:
        mid = s.add_message(cid, None, "assistant", name).value[0]
        blob = msgpack.unpackb(
            store_mod.Store._pack_frames_blob([_frame(0, "reading")], [0], 1),
            strict_map_key=False,
        )
        for key, value in patch.items():
            if key == "frames":
                blob["frames"][0].update(value[0])
            else:
                blob[key] = value
        frame_file = f"{name}.msgpack"
        store_mod.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        (store_mod.FRAMES_DIR / frame_file).write_bytes(msgpack.packb(blob, use_bin_type=True))
        conn = s._conn()
        conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", (frame_file, mid))
        conn.commit()
        with pytest.raises(store_mod.FrameStorageError):
            s.load_frames(mid)


@pytest.mark.parametrize("run_id", ["__missing__", None, "", "A" * 32, "g" * 32, "1" * 31, 7])
def test_schema_v3_requires_one_valid_homogeneous_run_id(tmp_path, monkeypatch, run_id):
    import msgpack
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "T").value[0]
    frame = _frame(0, "reading", "0")
    frame["generation_run_id"] = "1" * 32
    blob = msgpack.unpackb(store_mod.Store._pack_frames_blob([frame], [0], 1), strict_map_key=False)
    if run_id == "__missing__":
        del blob["frames"][0]["generation_run_id"]
    else:
        blob["frames"][0]["generation_run_id"] = run_id
    store_mod.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = store_mod.FRAMES_DIR / "bad-run.msgpack"
    path.write_bytes(msgpack.packb(blob, use_bin_type=True))
    conn = s._conn()
    conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", (path.name, mid))
    conn.commit()
    with pytest.raises(store_mod.FrameStorageError):
        s.load_frames(mid)


def test_persisted_continue_drops_overlapping_reread_frames_and_reloads(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(
        cid,
        None,
        "assistant",
        "old",
        meta={"publication_state": "complete", "frames_lens_binding_id": 7, "frames_layers": [0], "frames_k": 1},
    ).value[0]
    lens_meta = {"path": "/lens", "model_id": "m", "model_revision": "r", "fitted_layers_all": [0], "tapped_layers": [0], "k": 1}
    lens = type("Lens", (), {"layers": [0], "k": 1, "binding_id": 7, "meta": lens_meta})()
    descriptor = app._frame_descriptor(lens, [0], 1)
    old_reading = _frame(0, "reading", "0")
    old_reading["gen"] = 10
    old_thinking = _frame(1, "thinking", "0")
    old_thinking["gen"] = 10
    stale_tail = _frame(4, "thinking", "0")
    stale_tail["gen"] = 40
    s.save_frames(mid, [old_reading, old_thinking, stale_tail], [0], 1, frame_descriptor=descriptor)
    emitted = []

    def generate_first(**kwargs):
        assert kwargs["capture_full_input_frames"] is True
        run_id = "2" * 32
        kwargs["emit"]({"type": "frame", **_frame(1, "reading", "0"), "generation_run_id": run_id})
        new_frame = _frame(2, "thinking", "0")
        new_frame["gen"] = 20
        new_frame["generation_run_id"] = run_id
        kwargs["emit"]({"type": "frame", **new_frame})
        kwargs["emit"]({"type": "done", "text": " plus", "meta": {"model_id": "m"}, "stats": {}, "stopped": False, "generation_run_id": run_id, "continuation_suffix_start_pos": 2, "continuation_suffix_end_pos": 3})

    monkeypatch.setattr(app.manager, "generate", generate_first)
    context = type("Ctx", (), {"lens": lens, "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)
    loaded = s.load_frames(mid)
    assert [frame["pos"] for frame in loaded["frames"]] == [1, 2]
    assert [frame["phase"] for frame in loaded["frames"]] == ["reading", "thinking"]
    assert [frame["gen"] for frame in loaded["frames"]] == [None, 20]

    def generate_second(**kwargs):
        run_id = "3" * 32
        kwargs["emit"]({"type": "frame", **_frame(2, "reading", "0"), "generation_run_id": run_id})
        kwargs["emit"]({"type": "frame", **_frame(3, "thinking", "0"), "generation_run_id": run_id})
        kwargs["emit"]({"type": "done", "text": " again", "meta": {"model_id": "m"}, "stats": {}, "stopped": False, "generation_run_id": run_id, "continuation_suffix_start_pos": 3, "continuation_suffix_end_pos": 4})

    monkeypatch.setattr(app.manager, "generate", generate_second)
    lens.binding_id = 99  # process-local identity is not archive compatibility
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)
    loaded = s.load_frames(mid)
    assert [frame["pos"] for frame in loaded["frames"]] == [2, 3]
    body, _ = s.export(cid, fmt="md", include_frames=True)
    assert "2 lens frames" in body


def test_persisted_continue_rejects_incompatible_frame_configuration(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"frames_lens_binding_id": 7}).value[0]
    original_lens = type("Lens", (), {"layers": [0], "k": 1, "binding_id": 7, "meta": {"path": "/lens", "model_id": "m", "model_revision": "r", "fitted_layers_all": [0], "tapped_layers": [0], "k": 1}})()
    s.save_frames(mid, [_frame(0, "thinking", "0")], [0], 1, frame_descriptor=app._frame_descriptor(original_lens, [0], 1))
    before = s.get_message(mid)

    def generate(**kwargs):
        assert kwargs["capture_full_input_frames"] is True
        kwargs["emit"]({"type": "frame", **_frame(1, "thinking", "0")})
        kwargs["emit"]({"type": "done", "text": " plus", "meta": {}, "stats": {}, "stopped": False, "continuation_suffix_start_pos": 1, "continuation_suffix_end_pos": 2})

    monkeypatch.setattr(app.manager, "generate", generate)
    emitted = []
    changed_layers = type("Ctx", (), {"lens": type("Lens", (), {"layers": [1], "k": 1, "binding_id": 7, "meta": dict(original_lens.meta, tapped_layers=[1])})(), "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, mid, threading.Event(), emitted.append, changed_layers)
    after = s.get_message(mid)
    assert after["content"] == before["content"]
    assert after["version"] == before["version"]
    assert emitted[-1]["type"] == "error"


def test_pointerless_framed_continuation_replaces_stale_frame_metadata(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={
        "publication_state": "frames_publication_failed",
        "frames_expected": False,
        "frames_invalidated": True,
        "frames_invalidation_reason": "edited",
        "frames_error_kind": "write",
        "frames_error": "old failure",
    }).value[0]
    lens_meta = {"path": "/lens", "model_id": "m", "model_revision": "r", "fitted_layers_all": [0], "tapped_layers": [0], "k": 1}
    lens = type("Lens", (), {"layers": [0], "k": 1, "binding_id": 9, "meta": lens_meta})()
    run_id = "4" * 32

    def generate(**kwargs):
        kwargs["emit"]({"type": "frame", **_frame(0, "reading", "0"), "generation_run_id": run_id})
        kwargs["emit"]({"type": "frame", **_frame(1, "thinking", "0"), "generation_run_id": run_id})
        kwargs["emit"]({
            "type": "done", "text": " plus", "meta": {}, "stats": {}, "stopped": False,
            "generation_run_id": run_id,
            "continuation_suffix_start_pos": 1, "continuation_suffix_end_pos": 2,
        })

    monkeypatch.setattr(app.manager, "generate", generate)
    emitted = []
    context = type("Ctx", (), {"lens": lens, "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)

    current = s.get_message(mid)
    assert current["frames_file"] is not None
    meta = current["meta"] if isinstance(current["meta"], dict) else json.loads(current["meta"])
    assert meta["publication_state"] == "complete"
    assert meta["frames_expected"] is True
    for stale in ("frames_invalidated", "frames_invalidation_reason", "frames_error_kind", "frames_error"):
        assert stale not in meta
    loaded = s.load_frames(mid)
    assert loaded["version"] == 3
    assert [frame["generation_run_id"] for frame in loaded["frames"]] == [run_id, run_id]


def test_lens_disabled_continuation_invalidates_existing_frames(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old", meta={"publication_state": "complete"}).value[0]
    old_file = s.save_frames(mid, [_frame(0, "thinking", "0")], [0], 1).value

    def generate(**kwargs):
        kwargs["emit"]({"type": "done", "text": " plus", "meta": {}, "stats": {}, "stopped": False})

    monkeypatch.setattr(app.manager, "generate", generate)
    context = type("Ctx", (), {"lens": None, "intervention_snapshot": {"rules": []}})()
    emitted = []
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)
    current = s.get_message(mid)
    assert current["content"] == "old plus"
    assert current["frames_file"] is None
    meta = current["meta"] if isinstance(current["meta"], dict) else json.loads(current["meta"])
    assert meta["frames_invalidated"] is True
    assert not (store_mod.FRAMES_DIR / old_file).exists()


def test_lens_disabled_continuation_fails_closed_on_missing_frames(tmp_path, monkeypatch):
    from api import app
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    monkeypatch.setattr(app, "store", s)
    cid = s.create_conversation("c").value
    mid = s.add_message(cid, None, "assistant", "old").value[0]
    conn = s._conn()
    conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", ("missing.msgpack", mid))
    conn.commit()
    before = s.get_message(mid)

    def generate(**kwargs):
        kwargs["emit"]({"type": "done", "text": " plus", "meta": {}, "stats": {}, "stopped": False})

    monkeypatch.setattr(app.manager, "generate", generate)
    context = type("Ctx", (), {"lens": None, "intervention_snapshot": {"rules": []}})()
    emitted = []
    app._persisted_continue({}, mid, threading.Event(), emitted.append, context)
    after = s.get_message(mid)
    assert after["content"] == before["content"]
    assert after["version"] == before["version"]
    assert emitted[-1]["type"] == "error"


def test_frame_candidate_validation_failure_leaves_old_row_unchanged(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c").value
    inserted = s.add_message(cid, None, "assistant", "old", return_version=True).value
    mid, version = inserted
    old = s.get_message(mid)
    monkeypatch.setattr(s, "_validate_frame_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(store_mod.FrameStorageError("bad candidate")))
    outcome = s.update_message_and_frames_if_unchanged(mid, version, "new", {}, frames=[_frame(1)], layers=[0], k=1)
    assert outcome.state == "not_committed"
    assert s.get_message(mid)["content"] == old["content"]
    assert s.get_message(mid)["version"] == old["version"]
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_operation_handoff_can_own_acquisition_before_setup():
    from api import app

    c = ModelSessionCoordinator()
    handoff = app.OperationHandoff(c, threading.Event())
    assert handoff.state is app.HandoffState.NEW
    token, snap = handoff.acquire(OperationType.GENERATE)
    assert token is handoff.token
    assert handoff.state is app.HandoffState.PREPARING
    assert handoff.heavy["snap"] is snap
    handoff.clear_heavy()
    c.release(token)


def test_operation_handoff_acquire_rolls_back_snapshot_registration_failure(monkeypatch):
    from api import app
    from core.model_session import ModelSessionCoordinator, OperationType

    coordinator = ModelSessionCoordinator()
    handoff = app.OperationHandoff(coordinator, stop_event=threading.Event())
    monkeypatch.setattr(handoff, "set_heavy", lambda **_refs: (_ for _ in ()).throw(MemoryError("injected")))
    with pytest.raises(RuntimeError, match="MemoryError: injected") as raised:
        handoff.acquire(OperationType.GENERATE)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert coordinator.status_snapshot().operation is None
    assert handoff.state is app.HandoffState.CLOSED


def test_operation_handoff_close_before_dispatch_is_idempotent():
    from api import app
    from core.model_session import ModelSessionCoordinator, OperationType

    coordinator = ModelSessionCoordinator()
    handoff = app.OperationHandoff(coordinator, stop_event=threading.Event())
    handoff.acquire(OperationType.GENERATE)
    asyncio.run(handoff.close())
    asyncio.run(handoff.close())
    assert coordinator.status_snapshot().operation is None
    assert handoff.state is app.HandoffState.CLOSED


def test_first_party_pin_calls_propagate_generation_run_identity():
    from pathlib import Path

    root = Path(__file__).parents[1] / "ui" / "src"
    lens_view = (root / "LensView.jsx").read_text(encoding="utf-8")
    editor = (root / "Editor.jsx").read_text(encoding="utf-8")
    app_source = (root / "App.jsx").read_text(encoding="utf-8")
    assert "generation_run_id: generationRunId" in lens_view
    assert "generation_run_id: generationRunId" in editor
    assert "generationRunId={viewRunId" in app_source
    assert "generationRunId={currentGenView().generationRunId}" in app_source


def test_dispatched_routes_use_one_preacquisition_handoff():
    import inspect
    from api import app

    routes = (
        app.api_delete_model, app.api_lens_load, app.api_edit_export,
        app.api_edit_export_gguf, app.api_generate_sync, app.api_token_neighbors,
        app.api_lens_pin, app._setup_ws_worker,
    )
    for route in routes:
        source = inspect.getsource(route)
        assert "OperationHandoff.acquire_scope(" in source, route.__name__
        assert "handoff.acquire(" not in source, route.__name__
        assert "async with handoff.awaiter_scope()" not in source, route.__name__
        assert ".create_dispatch(" in source, route.__name__
        assert "manager.coordinator.acquire(" not in source, route.__name__
        assert "manager.coordinator.release(" not in source, route.__name__
        assert "_handoff_thread_worker" not in source, route.__name__


def test_marker_tail_recognizes_alternate_decoded_prefixes_without_full_history():
    from core.model_manager import _marker_prefix_suffix

    markers = ("\nUser:", "\nAssistant:")
    assert _marker_prefix_suffix("\nUs", markers) == "\nUs"
    assert _marker_prefix_suffix("ordinary text\nAss", markers) == "\nAss"
    assert _marker_prefix_suffix("ordinary text", markers) == ""
    long_text = "x" * 10000 + "\nUser"
    assert _marker_prefix_suffix(long_text, markers) == "\nUser"


def test_marker_tail_keeps_latest_eligible_token_boundary():
    from core.model_manager import (
        FALLBACK_MARKER_TOKEN_LIMIT,
        _smallest_marker_tail_start,
    )

    class Tokenizer:
        pieces = {1: "v\nU", 2: "v\nU"}

        def decode(self, ids, **_kwargs):
            return "".join(self.pieces[token_id] for token_id in ids)

    pending = [
        {"token_id": 1, "pos": 10, "frame": {"pos": 10}},
        {"token_id": 2, "pos": 11, "frame": {"pos": 11}},
    ]
    assert _smallest_marker_tail_start(
        pending, Tokenizer(), ("\nUser:", "\nAssistant:")
    ) == 1
    assert FALLBACK_MARKER_TOKEN_LIMIT == 32


def test_marker_tail_accepts_arbitrarily_long_decoded_token_prefix():
    from core.model_manager import _smallest_marker_tail_start

    class Tokenizer:
        def decode(self, ids, **_kwargs):
            return "".join({1: "x" * 10_000 + "\nU", 2: "ser:"}[token_id] for token_id in ids)

    pending = [{"token_id": 1, "pos": 0, "frame": None}]
    assert _smallest_marker_tail_start(
        pending, Tokenizer(), ("\nUser:", "\nAssistant:")
    ) == 0


def test_store_migrates_and_increments_conversation_versions(tmp_path, monkeypatch):
    import sqlite3
    from core import store as store_mod

    database = tmp_path / "db.sqlite3"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
        "tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(store_mod, "DB_PATH", database)
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    created = store.create_conversation("versioned")
    assert created.state == "committed"
    assert created.observed_version == 1
    updated = store.update_conversation(created.value, title="twice")
    assert updated.state == "committed"
    assert updated.expected_version == 1
    assert updated.observed_version == 2
    row = store._conn().execute(
        "SELECT version, incarnation_id FROM conversations WHERE id = ?", (created.value,)
    ).fetchone()
    assert row["version"] == 2
    assert len(row["incarnation_id"]) == 32
    assert row["incarnation_id"] == row["incarnation_id"].lower()


def test_store_migrates_stable_conversation_incarnation(tmp_path, monkeypatch):
    import sqlite3
    from core import store as store_mod

    database = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
        "tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO conversations VALUES (7, 'legacy', '[]', '2020-01-01', '2020-01-01')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(store_mod, "DB_PATH", database)
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")

    first_store = store_mod.Store()
    first = first_store._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = 7"
    ).fetchone()[0]
    first_store._discard_conn(first_store._conn())
    second = store_mod.Store()._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = 7"
    ).fetchone()[0]
    assert second == first
    assert len(first) == 32
    assert all(char in "0123456789abcdef" for char in first)


def test_conversation_delete_records_causal_tombstone(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    created = store.create_conversation("delete")
    deleted = store.delete_conversation(created.value)
    assert deleted.state == "committed"
    assert deleted.expected_version == 1
    assert deleted.observed_version == 2
    tombstone = store._conn().execute(
        "SELECT mutation_id, incarnation_id, deleted_version FROM conversation_tombstones WHERE conversation_id = ?",
        (created.value,),
    ).fetchone()
    assert tombstone["mutation_id"] == deleted.value
    assert tombstone["deleted_version"] == 2
    assert len(tombstone["incarnation_id"]) == 32


def test_conversation_recreation_preserves_prior_delete_identity(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    first = store.create_conversation("first")
    first_row = store._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = ?", (first.value,)
    ).fetchone()
    deleted = store.delete_conversation(first.value)
    second = store.create_conversation("second")
    second_row = store._conn().execute(
        "SELECT incarnation_id, version FROM conversations WHERE id = ?", (second.value,)
    ).fetchone()
    tombstone = store._conn().execute(
        "SELECT incarnation_id, mutation_id FROM conversation_tombstones WHERE mutation_id = ?",
        (deleted.value,),
    ).fetchone()

    assert second.value == first.value
    assert second_row["version"] == 1
    assert second_row["incarnation_id"] != first_row["incarnation_id"]
    assert tombstone["incarnation_id"] == first_row["incarnation_id"]
    assert tombstone["mutation_id"] == deleted.value


def test_delete_reconciliation_survives_same_id_recreation(tmp_path, monkeypatch):
    import sqlite3
    import uuid
    from core import store as store_mod

    database = tmp_path / "db.sqlite3"
    monkeypatch.setattr(store_mod, "DB_PATH", database)
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    created = store.create_conversation("first")
    original = store._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = ?", (created.value,)
    ).fetchone()[0]
    operational = store._conn()

    class CommitThenRecreate:
        def __getattr__(self, name):
            return getattr(operational, name)

        @property
        def in_transaction(self):
            return operational.in_transaction

        def commit(self):
            operational.commit()
            replacement = sqlite3.connect(database)
            replacement.execute(
                "INSERT INTO conversations "
                "(id, title, tags, created_at, updated_at, version, incarnation_id) "
                "VALUES (?, 'second', '[]', 'later', 'later', 1, ?)",
                (created.value, uuid.uuid4().hex),
            )
            replacement.commit()
            replacement.close()
            raise RuntimeError("commit acknowledgement lost")

    monkeypatch.setattr(store, "_conn", lambda: CommitThenRecreate())
    deleted = store.delete_conversation(created.value)
    replacement = sqlite3.connect(database).execute(
        "SELECT incarnation_id FROM conversations WHERE id = ?", (created.value,)
    ).fetchone()[0]
    tombstone = sqlite3.connect(database).execute(
        "SELECT incarnation_id FROM conversation_tombstones WHERE mutation_id = ?", (deleted.value,)
    ).fetchone()[0]

    assert deleted.state == "committed"
    assert replacement != original
    assert tombstone == original


def test_competing_delete_and_recreation_is_superseded(tmp_path, monkeypatch):
    import sqlite3
    import uuid
    from core import store as store_mod

    database = tmp_path / "db.sqlite3"
    monkeypatch.setattr(store_mod, "DB_PATH", database)
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    created = store.create_conversation("first")
    original = store._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = ?", (created.value,)
    ).fetchone()[0]
    operational = store._conn()
    competing_mutation = uuid.uuid4().hex
    replacement_incarnation = uuid.uuid4().hex

    class RollbackThenCompete:
        def __getattr__(self, name):
            return getattr(operational, name)

        @property
        def in_transaction(self):
            return operational.in_transaction

        def commit(self):
            operational.rollback()
            competing = sqlite3.connect(database)
            competing.execute(
                "INSERT INTO conversation_tombstones "
                "(mutation_id, conversation_id, incarnation_id, deleted_version, deleted_at) "
                "VALUES (?, ?, ?, 2, 'later')",
                (competing_mutation, created.value, original),
            )
            competing.execute("DELETE FROM conversations WHERE id = ?", (created.value,))
            competing.execute(
                "INSERT INTO conversations "
                "(id, title, tags, created_at, updated_at, version, incarnation_id) "
                "VALUES (?, 'replacement', '[]', 'later', 'later', 1, ?)",
                (created.value, replacement_incarnation),
            )
            competing.commit()
            competing.close()
            raise RuntimeError("first delete did not commit")

    monkeypatch.setattr(store, "_conn", lambda: RollbackThenCompete())
    outcome = store.delete_conversation(created.value)
    assert outcome.state == "superseded"
    assert outcome.observed_version == 1
    assert store_mod.Store()._conn().execute(
        "SELECT incarnation_id FROM conversations WHERE id = ?", (created.value,)
    ).fetchone()[0] == replacement_incarnation
    tombstones = sqlite3.connect(database).execute(
        "SELECT mutation_id FROM conversation_tombstones WHERE incarnation_id = ?", (original,)
    ).fetchall()
    assert [row[0] for row in tombstones] == [competing_mutation]


def _commit_then_raise_proxy(conn):
    class Proxy:
        def __getattr__(self, name):
            return getattr(conn, name)

        @property
        def in_transaction(self):
            return conn.in_transaction

        def commit(self):
            conn.commit()
            raise RuntimeError("commit acknowledgement lost")

    return Proxy()


def _assert_primitive_outcome(outcome):
    from pathlib import Path
    import sqlite3
    import types

    assert outcome.state in {"committed", "not_committed", "stale", "superseded", "ambiguous"}
    for value in vars(outcome).values():
        assert not isinstance(value, (BaseException, Path, sqlite3.Connection, sqlite3.Cursor, types.TracebackType))
    assert isinstance(outcome.value, (type(None), bool, int, float, str, tuple))


def test_store_create_and_insert_reconcile_commit_acknowledgement_loss(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    create_conn = store._conn()
    monkeypatch.setattr(store, "_conn", lambda: _commit_then_raise_proxy(create_conn))
    created = store.create_conversation("durable")
    _assert_primitive_outcome(created)
    assert created.state == "committed"
    assert created.value == 1

    # The reconciled operational connection was deliberately discarded.
    monkeypatch.setattr(store, "_conn", store_mod.Store._conn.__get__(store))
    insert_conn = store._conn()
    monkeypatch.setattr(store, "_conn", lambda: _commit_then_raise_proxy(insert_conn))
    inserted = store.add_message(created.value, None, "assistant", "answer")
    _assert_primitive_outcome(inserted)
    assert inserted.state == "committed"
    assert inserted.value == (1, 0)


def test_store_create_reports_definite_noncommit(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    conn = store._conn()

    class FailBeforeCommit:
        def __getattr__(self, name):
            return getattr(conn, name)

        @property
        def in_transaction(self):
            return conn.in_transaction

        def commit(self):
            raise RuntimeError("commit rejected")

    monkeypatch.setattr(store, "_conn", lambda: FailBeforeCommit())
    outcome = store.create_conversation("not durable")
    _assert_primitive_outcome(outcome)
    assert outcome.state == "not_committed"
    assert outcome.entity_id == "1"


def test_frame_attachment_and_lens_clear_reconcile_lost_commit_ack(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    store = store_mod.Store()
    cid = store.create_conversation("c").value
    mid, version = store.add_message(
        cid, None, "assistant", "answer", meta={"publication_state": "frames_pending"}
    ).value

    real_conn_method = store._conn
    attach_conn = store._conn()
    monkeypatch.setattr(store, "_conn", lambda: _commit_then_raise_proxy(attach_conn))
    attached = store.save_frames(
        mid, [_frame(0)], [0], 1, expected_version=version,
        complete_meta={"publication_state": "complete"},
    )
    _assert_primitive_outcome(attached)
    assert attached.state == "committed"
    old_file = attached.value
    assert (store_mod.FRAMES_DIR / old_file).exists()

    monkeypatch.setattr(store, "_conn", real_conn_method)
    clear_conn = store._conn()
    monkeypatch.setattr(store, "_conn", lambda: _commit_then_raise_proxy(clear_conn))
    cleared = store.update_message_and_frames_if_unchanged(
        mid, attached.observed_version, "answer plus", {"frames_invalidated": True},
        clear_frames=True,
    )
    _assert_primitive_outcome(cleared)
    assert cleared.state == "committed"
    assert not (store_mod.FRAMES_DIR / old_file).exists()
