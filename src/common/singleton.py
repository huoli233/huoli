# -*- coding: utf-8 -*-
"""
线程安全的单例模式工具
提供统一的单例和多实例管理机制，解决项目中的单例混乱问题
"""

import threading
from typing import TypeVar, Type, Optional, Any, Dict, Callable

T = TypeVar('T')

# 全局锁注册表
_locks: Dict[str, threading.Lock] = {}
_locks_meta_lock = threading.Lock()


def _get_class_lock(cls_name: str) -> threading.Lock:
    """获取或创建类级别的锁，保证线程安全"""
    if cls_name not in _locks:
        with _locks_meta_lock:
            if cls_name not in _locks:
                _locks[cls_name] = threading.Lock()
    return _locks[cls_name]


class SingletonMeta(type):
    """元类方式实现的线程安全单例"""

    _instances: Dict[str, Any] = {}

    def __call__(cls, *args, **kwargs):
        cls_name = cls.__name__
        if cls_name not in cls._instances:
            lock = _get_class_lock(cls_name)
            with lock:
                if cls_name not in cls._instances:
                    cls._instances[cls_name] = super().__call__(*args, **kwargs)
        return cls._instances[cls_name]


def singleton(cls: Type[T]) -> Type[T]:
    """
    装饰器方式的线程安全单例
    使用示例：
        @singleton
        class MyClass:
            pass
        instance = MyClass()
    """
    _instance: Optional[T] = None
    lock = _get_class_lock(cls.__name__)

    def get_instance(*args, **kwargs) -> T:
        nonlocal _instance
        if _instance is None:
            with lock:
                if _instance is None:
                    _instance = cls(*args, **kwargs)
        return _instance

    # 替换类的 __new__ 方法，使其返回单例
    original_new = cls.__new__

    def __new__(cls_inner, *args, **kwargs):
        return get_instance(*args, **kwargs)

    cls.__new__ = staticmethod(__new__)  # type: ignore
    return cls


class SingletonFactory:
    """工厂模式的单例管理器，支持带参数的单例"""

    _factories: Dict[str, Callable[..., Any]] = {}
    _instances: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., T]) -> None:
        """注册单例工厂函数"""
        cls._factories[name] = factory

    @classmethod
    def get_instance(cls, name: str, *args, **kwargs) -> Any:
        """获取或创建单例实例"""
        if name not in cls._instances:
            lock = _get_class_lock(name)
            with lock:
                if name not in cls._instances:
                    if name not in cls._factories:
                        raise ValueError(f"未注册的单例: {name}")
                    cls._instances[name] = cls._factories[name](*args, **kwargs)
        return cls._instances[name]

    @classmethod
    def clear_instance(cls, name: str) -> None:
        """清除指定单例实例"""
        lock = _get_class_lock(name)
        with lock:
            cls._instances.pop(name, None)


class MultiInstanceManager:
    """多实例管理器，支持按key管理多个实例（线程安全）"""

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._item_locks: Dict[str, threading.Lock] = {}

    def get_or_create(self, key: str, factory: Callable[..., T], *args, **kwargs) -> T:
        """获取或创建指定key的实例"""
        if key not in self._instances:
            with self._lock:
                if key not in self._instances:
                    self._instances[key] = factory(*args, **kwargs)
        return self._instances[key]

    def remove(self, key: str) -> None:
        """移除指定key的实例"""
        with self._lock:
            self._instances.pop(key, None)

    def clear(self) -> None:
        """清空所有实例"""
        with self._lock:
            self._instances.clear()

    def contains(self, key: str) -> bool:
        """检查是否包含指定key的实例"""
        return key in self._instances

    def get_all_keys(self) -> list:
        """获取所有已存在的key"""
        return list(self._instances.keys())
