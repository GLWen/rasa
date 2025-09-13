import asyncio
import functools
import importlib
import inspect
import logging
from typing import Text, Dict, Optional, Any, List, Callable, Collection, Type

from rasa.shared.exceptions import RasaException

# =============================================================================
# 通用工具函数模块 - 提供 Rasa 框架中常用的工具函数
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# 动态类加载和反射相关函数
# =============================================================================

def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Type:
    """根据模块名称和类路径尝试检索类。

    加载的类可用于实例化新对象。支持绝对路径和相对路径查找。

    Args:
        module_path: Python 类的绝对路径，或在本地/全局作用域中的类名
        lookup_path: 如果无法在本地/全局作用域中找到类，则从此路径加载类

    Returns:
        一个 Python 类

    Raises:
        ImportError: 当无法找到 Python 类时
        RasaException: 当导入的结果不是类时
    """
    klass = None
    if "." in module_path:
        # 处理绝对路径，如 "rasa.core.policies.Policy"
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        klass = getattr(m, class_name, None)
    elif lookup_path:
        # 尝试从查找路径导入类
        m = importlib.import_module(lookup_path)
        klass = getattr(m, module_path, None)

    if klass is None:
        raise ImportError(f"无法从路径 {module_path} 检索类。")

    if not inspect.isclass(klass):
        raise RasaException(
            f"`class_from_module_path()` 期望返回一个类，"
            f"但对于 {module_path} 我们得到了 {type(klass)}。"
        )
    return klass


def all_subclasses(cls: Any) -> List[Any]:
    """返回类的所有已知（已导入）子类。
    
    递归查找所有子类，包括子类的子类。

    Args:
        cls: 要查找子类的基类

    Returns:
        所有非抽象子类的列表
    """
    classes = cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in all_subclasses(s)
    ]

    return [subclass for subclass in classes if not inspect.isabstract(subclass)]


def module_path_from_instance(inst: Any) -> Text:
    """返回实例类的模块路径。
    
    Args:
        inst: 要获取模块路径的实例

    Returns:
        格式为 "module.name.ClassName" 的模块路径
    """
    return inst.__module__ + "." + inst.__class__.__name__


def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]:
    """按第一个键对字典列表进行排序。
    
    Args:
        dicts: 要排序的字典列表

    Returns:
        按第一个键排序的字典列表
    """
    return sorted(dicts, key=lambda d: list(d.keys())[0])


# =============================================================================
# 装饰器和缓存相关函数
# =============================================================================

def lazy_property(function: Callable) -> Any:
    """允许避免重复计算属性。

    结果存储在局部变量中。属性的计算将在第一次调用属性时发生一次。
    所有后续调用将使用存储在私有属性中的值。

    Args:
        function: 要装饰的函数

    Returns:
        装饰后的属性对象
    """
    attr_name = "_lazy_" + function.__name__

    def _lazyprop(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return property(_lazyprop)


def cached_method(f: Callable[..., Any]) -> Callable[..., Any]:
    """基于调用的 `args` 和 `kwargs` 缓存方法调用。

    适用于 `async` 和 `sync` 方法。不要将此装饰器应用于函数。

    Args:
        f: 要缓存返回值的装饰方法

    Returns:
        方法在给定参数下第一次调用时给出的返回值
    """
    assert "self" in arguments_of(f), "此装饰器只能用于方法。"

    class Cache:
        """辅助类，用于抽象缓存细节。"""

        def __init__(self, caching_object: object, args: Any, kwargs: Any) -> None:
            self.caching_object = caching_object
            self.cache = getattr(caching_object, self._cache_name(), {})
            # noinspection PyUnresolvedReferences
            self.cache_key = functools._make_key(args, kwargs, typed=False)

        def _cache_name(self) -> Text:
            return f"_cached_{self.caching_object.__class__.__name__}_{f.__name__}"

        def is_cached(self) -> bool:
            return self.cache_key in self.cache

        def cache_result(self, result: Any) -> None:
            self.cache[self.cache_key] = result
            setattr(self.caching_object, self._cache_name(), self.cache)

        def cached_result(self) -> Any:
            return self.cache[self.cache_key]

    if asyncio.iscoroutinefunction(f):
        # 处理异步方法
        @functools.wraps(f)
        async def decorated(self: object, *args: Any, **kwargs: Any) -> Any:
            cache = Cache(self, args, kwargs)
            if not cache.is_cached():
                # 立即存储任务，以便该方法的其他并发调用可以重用同一个任务
                # 而不会安排第二次执行
                to_cache = asyncio.ensure_future(f(self, *args, **kwargs))
                cache.cache_result(to_cache)
            return await cache.cached_result()

        return decorated
    else:
        # 处理同步方法
        @functools.wraps(f)
        def decorated(self: object, *args: Any, **kwargs: Any) -> Any:
            cache = Cache(self, args, kwargs)
            if not cache.is_cached():
                to_cache = f(self, *args, **kwargs)
                cache.cache_result(to_cache)
            return cache.cached_result()

        return decorated


# =============================================================================
# 字符串和集合处理函数
# =============================================================================

def transform_collection_to_sentence(collection: Collection[Text]) -> Text:
    """将集合转换为句子格式。
    
    例如，将列表 ['A', 'B', 'C'] 转换为句子 'A, B and C'。

    Args:
        collection: 要转换的文本集合

    Returns:
        转换后的句子字符串
    """
    x = list(collection)
    if len(x) >= 2:
        return ", ".join(map(str, x[:-1])) + " and " + x[-1]
    return "".join(collection)


# =============================================================================
# 函数参数处理函数
# =============================================================================

def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """返回函数所需的 kwargs。
    
    只返回函数接受的参数，排除在异常列表中的键。

    Args:
        kwargs: 所有可用的 kwargs
        func: 要调用的函数
        excluded_keys: 要从结果中排除的键

    Returns:
        被 `func` 接受的 kwargs 子集
    """
    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }


# =============================================================================
# 警告和日志相关函数
# =============================================================================

def mark_as_experimental_feature(feature_name: Text) -> None:
    """警告用户他们正在使用实验性功能。"""

    logger.warning(
        f"{feature_name} 目前是实验性的，可能会在未来发生变化或被移除 🔬 "
        "请在论坛 (https://forum.rasa.com) 分享您的反馈，"
        "帮助我们使此功能准备好投入生产。"
    )


# =============================================================================
# 函数内省和参数处理函数
# =============================================================================

def arguments_of(func: Callable) -> List[Text]:
    """返回函数 `func` 的参数作为名称列表。
    
    Args:
        func: 要检查参数的函数

    Returns:
        函数参数名称列表
    """
    import inspect

    return list(inspect.signature(func).parameters.keys())


# =============================================================================
# 列表和字典处理函数
# =============================================================================

def extract_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
    """从两个列表中提取重复项。
    
    Args:
        list1: 第一个列表
        list2: 第二个列表

    Returns:
        两个列表中的重复项列表
    """
    if list1:
        dict1 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list1
        }
    else:
        dict1 = {}

    if list2:
        dict2 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list2
        }
    else:
        dict2 = {}

    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    dupes = set1.intersection(set2)
    return sorted(list(dupes))


def clean_duplicates(dupes: Dict[Text, Any]) -> Dict[Text, Any]:
    """移除空值的键。
    
    Args:
        dupes: 要清理的字典

    Returns:
        清理后的字典
    """
    duplicates = dupes.copy()
    for k in dupes:
        if not dupes[k]:
            duplicates.pop(k)

    return duplicates


# =============================================================================
# 数据合并函数
# =============================================================================

def merge_dicts(
    tempDict1: Dict[Text, Any],
    tempDict2: Dict[Text, Any],
    override_existing_values: bool = False,
) -> Dict[Text, Any]:
    """合并两个字典。
    
    Args:
        tempDict1: 第一个字典
        tempDict2: 第二个字典
        override_existing_values: 是否覆盖现有值

    Returns:
        合并后的字典
    """
    if override_existing_values:
        merged_dicts, b = tempDict1.copy(), tempDict2.copy()
    else:
        merged_dicts, b = tempDict2.copy(), tempDict1.copy()
    merged_dicts.update(b)
    return merged_dicts


def merge_lists(
    list1: List[Any], list2: List[Any], override: bool = False
) -> List[Any]:
    """合并两个列表。
    
    Args:
        list1: 第一个列表
        list2: 第二个列表
        override: 是否覆盖（当前未使用）

    Returns:
        合并并去重后的排序列表
    """
    return sorted(list(set(list1 + list2)))


def merge_lists_of_dicts(
    dict_list1: List[Dict],
    dict_list2: List[Dict],
    override_existing_values: bool = False,
) -> List[Dict]:
    """合并两个字典列表。
    
    Args:
        dict_list1: 第一个字典列表
        dict_list2: 第二个字典列表
        override_existing_values: 是否覆盖现有值

    Returns:
        合并后的字典列表
    """
    dict1 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list1
    }
    dict2 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list2
    }
    merged_dicts = merge_dicts(dict1, dict2, override_existing_values)
    return list(merged_dicts.values())
