from types import SimpleNamespace

import pytest

from rllm.engine.rollout import verl_compat


def test_get_llm_server_client_uses_verl_08_client():
    client = object()
    manager = SimpleNamespace(llm_client=client)

    assert verl_compat.get_llm_server_client(object(), manager) is client


def test_get_llm_server_client_reconstructs_verl_07_manager(monkeypatch):
    calls = []

    def fake_server_manager(config, *, servers, load_balancer_handle):
        calls.append((config, list(servers), load_balancer_handle))
        return "legacy-client"

    monkeypatch.setattr(verl_compat, "AsyncLLMServerManager", fake_server_manager)
    config = object()
    load_balancer = object()
    manager = SimpleNamespace(
        server_addresses=["server-0", "server-1"],
        server_handles=["handle-0", "handle-1"],
        global_load_balancer=load_balancer,
    )

    assert verl_compat.get_llm_server_client(config, manager) == "legacy-client"
    assert calls == [
        (
            config,
            [("server-0", "handle-0"), ("server-1", "handle-1")],
            load_balancer,
        )
    ]


def test_get_llm_server_client_reports_incomplete_verl_08_manager(monkeypatch):
    monkeypatch.setattr(verl_compat, "AsyncLLMServerManager", None)

    with pytest.raises(RuntimeError, match="does not expose the replacement llm_client"):
        verl_compat.get_llm_server_client(object(), SimpleNamespace())
