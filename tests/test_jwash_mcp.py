import importlib
import sys
from types import ModuleType

import pytest


class _FakeFastMCP:
    def __init__(self, *_args, **_kwargs):
        pass

    def tool(self):
        return lambda fn: fn

    def run(self, **_kwargs):
        raise AssertionError("test must not start an MCP server")


@pytest.fixture
def jwash_mcp(monkeypatch):
    mcp_module = ModuleType("mcp")
    server_module = ModuleType("mcp.server")
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    sys.modules.pop("scripts.jwash_mcp", None)
    module = importlib.import_module("scripts.jwash_mcp")
    yield module
    sys.modules.pop("scripts.jwash_mcp", None)


@pytest.mark.parametrize("legacy", [
    {"rebase_supported": False},
    {},
    {"rebase_supported": "false", "readthrough_supported": {"malformed": True}},
])
def test_advisory_legacy_diagnostics_never_reroute_mcp_edit(jwash_mcp, monkeypatch,
                                                            legacy):
    loaded = {"model_id": "fake/model", "n_layers": 2, **legacy}
    statuses = iter([
        {"loaded": loaded, "lens": {"name": "fake"}},
        {"loaded": loaded, "lens": {"name": "fake"},
         "interventions_mode": "readthrough", "interventions_scale": 1.0,
         "interventions": []},
    ])
    monkeypatch.setattr(jwash_mcp, "_status", lambda: next(statuses))
    monkeypatch.setattr(jwash_mcp, "_resolve_token", lambda _token: {"id": 7})
    calls = []

    def fake_call(method, path, body=None, timeout=600):
        calls.append((method, path, body))
        return {}

    monkeypatch.setattr(jwash_mcp, "_call", fake_call)
    result = jwash_mcp._add_rule(" token", "scale", 0.0, None, "0")

    assert calls[0] == ("PATCH", "/api/interventions", {"mode": "readthrough"})
    assert calls[1][0:2] == ("POST", "/api/interventions")
    assert result["mode"] == "readthrough"
