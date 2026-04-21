from dataclasses import dataclass
from typing import Dict, Type, Any, Optional


@dataclass
class UsageRecord:
    model_name: str
    provider_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class APIResponse:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    usage: UsageRecord | None = None
    error: str | None = None


class _Registry:
    def __init__(self):
        self.client_registry: Dict[str, Type] = {}
        self.client_instance_cache: Dict[str, Any] = {}

    @staticmethod
    def _build_cache_key(client_type: str, api_provider: Any = None) -> str:
        if api_provider is None:
            return client_type
        provider_name = getattr(api_provider, "name", None)
        if provider_name:
            return f"{client_type}:{provider_name}"
        provider_base_url = getattr(api_provider, "base_url", None)
        if provider_base_url:
            return f"{client_type}:{provider_base_url}"
        return client_type

    def register_client_class(self, client_type: str):
        def decorator(cls):
            self.client_registry[client_type] = cls
            return cls
        return decorator

    def get_client_class_instance(self, client_type: str, api_provider: Any = None, force_new: bool = False) -> Any:
        cache_key = self._build_cache_key(client_type, api_provider)
        if force_new:
            if client_type not in self.client_registry:
                raise ValueError(f"未注册的客户端类型: {client_type}")
            cls = self.client_registry[client_type]
            if api_provider:
                return cls(api_provider)
            return cls()
        if cache_key in self.client_instance_cache:
            return self.client_instance_cache[cache_key]
        if client_type not in self.client_registry:
            raise ValueError(f"未注册的客户端类型: {client_type}")
        cls = self.client_registry[client_type]
        if api_provider:
            instance = cls(api_provider)
        else:
            instance = cls()
        self.client_instance_cache[cache_key] = instance
        return instance

    def get_registered_clients(self) -> Dict[str, Type]:
        return self.client_registry.copy()

    def clear_client_instance_cache(
        self, client_type: Optional[str] = None, provider_name: Optional[str] = None
    ) -> None:
        if client_type is None and provider_name is None:
            self.client_instance_cache.clear()
            return
        cache_keys = list(self.client_instance_cache.keys())
        for cache_key in cache_keys:
            cached_client_type, _, cached_provider_name = cache_key.partition(":")
            if client_type is not None and cached_client_type != client_type:
                continue
            if provider_name is not None and cached_provider_name != provider_name:
                continue
            self.client_instance_cache.pop(cache_key, None)


client_registry = _Registry()
