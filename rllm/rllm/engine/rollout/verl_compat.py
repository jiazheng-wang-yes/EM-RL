from typing import Any

try:
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
except ImportError:
    AsyncLLMServerManager = None


def get_llm_server_client(config: Any, rollout_manager: Any) -> Any:
    """Return the rollout client for both verl 0.7 and 0.8+ managers."""
    llm_client = getattr(rollout_manager, "llm_client", None)
    if llm_client is not None:
        return llm_client

    if AsyncLLMServerManager is None:
        raise RuntimeError("The installed verl no longer provides AsyncLLMServerManager, and the rollout manager does not expose the replacement llm_client.")

    server_handles = getattr(rollout_manager, "server_handles", None)
    load_balancer = getattr(rollout_manager, "global_load_balancer", None)

    import inspect
    try:
        sig = inspect.signature(AsyncLLMServerManager)
    except Exception:
        sig = inspect.signature(AsyncLLMServerManager.__init__)

    if load_balancer is not None and "servers" in sig.parameters:
        servers = zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True)
        return AsyncLLMServerManager(config, servers=servers, load_balancer_handle=load_balancer)

    if server_handles is not None and "server_handles" in sig.parameters:
        return AsyncLLMServerManager(config, server_handles=server_handles)

    if load_balancer is None:
        raise RuntimeError("The verl rollout manager does not expose a global load balancer.")

    servers = zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True)
    return AsyncLLMServerManager(config, servers=servers, load_balancer_handle=load_balancer)
