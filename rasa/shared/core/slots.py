# =============================================================================
# 槽位系统模块 - 定义对话中的槽位类型和功能
# =============================================================================
# 此模块定义了 Rasa Core 中的槽位系统，包括各种槽位类型（文本、数值、
# 分类、列表等）和槽位管理功能。槽位用于在对话过程中存储和跟踪信息。

# 标准库导入
import logging                    # 日志记录
from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from typing import Any, Dict, List, Optional, Text, Type  # 类型提示

# Rasa 内部模块导入
import rasa.shared.core.constants  # Core 常量
from rasa.shared.exceptions import RasaException  # Rasa 异常
import rasa.shared.utils.common    # 通用工具函数
import rasa.shared.utils.io        # IO 工具函数
from rasa.shared.constants import DOCS_URL_SLOTS  # 槽位文档URL

# 日志记录器
logger = logging.getLogger(__name__)


# =============================================================================
# 异常类定义
# =============================================================================

class InvalidSlotTypeException(RasaException):
    """当槽位类型无效时抛出此异常。"""


class InvalidSlotConfigError(RasaException, ValueError):
    """当槽位配置无效时抛出此异常。"""


# =============================================================================
# 槽位基类定义
# =============================================================================

class Slot(ABC):
    """用于在对话过程中存储信息的键值存储。
    
    槽位是对话状态跟踪的核心组件，用于存储和跟踪
    对话过程中的各种信息，如用户偏好、实体值等。
    """

    @property
    @abstractmethod
    def type_name(self) -> Text:
        """槽位类型的名称。
        
        Returns:
            槽位类型的字符串名称
        """
        ...

    def __init__(
        self,
        name: Text,                           # 槽位名称
        mappings: List[Dict[Text, Any]],      # 槽位映射列表
        initial_value: Any = None,            # 初始值
        value_reset_delay: Optional[int] = None,  # 值重置延迟
        influence_conversation: bool = True,  # 是否影响对话
    ) -> None:
        """创建槽位。

        Args:
            name: 槽位的名称
            initial_value: 槽位的初始值
            mappings: 包含槽位映射的列表
            value_reset_delay: 槽位应在多少轮后重置为初始值。
                此行为目前尚未实现。
            influence_conversation: 如果为 `True`，槽位将被特征化，
                从而影响对话策略的预测。
        """
        self.name = name                      # 槽位名称
        self.mappings = mappings              # 槽位映射
        self._value = initial_value           # 槽位值
        self.initial_value = initial_value    # 初始值
        self._value_reset_delay = value_reset_delay  # 值重置延迟
        self.influence_conversation = influence_conversation  # 是否影响对话
        self._has_been_set = False            # 是否已被设置

    def feature_dimensionality(self) -> int:
        """此单个槽位创建多少个特征。

        Returns:
            特征数量。如果槽位未被特征化则返回 `0`。
            `as_feature` 返回的数组的维度必须与此值对应。
        """
        if not self.influence_conversation:  # 如果槽位不影响对话
            return 0  # 返回0个特征

        return self._feature_dimensionality()  # 返回特征维度

    def _feature_dimensionality(self) -> int:
        """参见 `feature_dimensionality` 的文档字符串。
        
        Returns:
            特征维度数
        """
        return 1  # 默认返回1个特征

    def has_features(self) -> bool:
        """指示槽位是否创建任何特征。
        
        Returns:
            如果槽位创建特征则返回True，否则返回False
        """
        return self.feature_dimensionality() != 0  # 检查特征维度是否为0

    def value_reset_delay(self) -> Optional[int]:
        """槽位应在多少轮后重置为初始值。

        如果延迟设置为 `None`，槽位将永远保持其值。
        
        Returns:
            重置延迟轮数，如果为None则表示不重置
        """
        # TODO: FUTURE 这需要实现 - 槽位尚未重置
        return self._value_reset_delay  # 返回重置延迟

    def as_feature(self) -> List[float]:
        """将槽位值转换为特征向量。
        
        Returns:
            特征向量列表
        """
        if not self.influence_conversation:  # 如果槽位不影响对话
            return []  # 返回空列表

        return self._as_feature()  # 调用子类实现的特征化方法

    @abstractmethod
    def _as_feature(self) -> List[float]:
        """将槽位值转换为特征向量的抽象方法。
        
        每个槽位类型都需要指定如何将其值转换为特征。
        
        Returns:
            特征向量列表
        """
        raise NotImplementedError(
            "每个槽位类型都需要指定如何将其值转换为特征。槽位 "
            "'{}' 是一个通用槽位，不能用于预测。请确保将此 "
            "槽位添加到您的域定义中，指定槽位的类型。如果您实现了 "
            "自定义槽位类型类，请确保实现 `.as_feature()`。"
            "".format(self.name)
        )

    def reset(self) -> None:
        """将槽位的值重置为初始值。"""
        self.value = self.initial_value  # 设置值为初始值
        self._has_been_set = False       # 标记为未设置

    @property
    def value(self) -> Any:
        """获取槽位的值。
        
        Returns:
            槽位的当前值
        """
        return self._value  # 返回内部值

    @value.setter
    def value(self, value: Any) -> None:
        """设置槽位的值。
        
        Args:
            value: 要设置的值
        """
        self._value = value        # 设置内部值
        self._has_been_set = True  # 标记为已设置

    @property
    def has_been_set(self) -> bool:
        """指示槽位的值是否已被设置。
        
        Returns:
            如果槽位值已被设置则返回True，否则返回False
        """
        return self._has_been_set  # 返回设置状态

    def __str__(self) -> Text:
        """返回槽位的字符串表示。
        
        Returns:
            格式化的槽位字符串
        """
        return f"{self.__class__.__name__}({self.name}: {self.value})"

    def __repr__(self) -> Text:
        """返回槽位的调试字符串表示。
        
        Returns:
            格式化的槽位调试字符串
        """
        return f"<{self.__class__.__name__}({self.name}: {self.value})>"

    @staticmethod
    def resolve_by_type(type_name: Text) -> Type["Slot"]:
        """根据类型名称返回槽位类。
        
        Args:
            type_name: 槽位类型名称
            
        Returns:
            对应的槽位类
            
        Raises:
            InvalidSlotTypeException: 如果找不到对应的槽位类型
        """
        for cls in rasa.shared.utils.common.all_subclasses(Slot):  # 遍历所有槽位子类
            if cls.type_name == type_name:  # 如果类型名称匹配
                return cls  # 返回对应的类
        try:
            return rasa.shared.utils.common.class_from_module_path(type_name)  # 尝试从模块路径加载
        except (ImportError, AttributeError):  # 如果导入失败
            raise InvalidSlotTypeException(
                f"无法找到槽位类型，'{type_name}' 既不是已知类型也不是 "
                f"用户定义的。如果您正在创建自己的槽位类型，请确保 "
                f"其模块路径正确。您可以在 {DOCS_URL_SLOTS} 找到所有内置类型"
            )

    def persistence_info(self) -> Dict[str, Any]:
        """返回持久化此槽位所需的相关信息。
        
        Returns:
            包含槽位持久化信息的字典
        """
        return {
            "type": rasa.shared.utils.common.module_path_from_instance(self),  # 槽位类型
            "initial_value": self.initial_value,  # 初始值
            "influence_conversation": self.influence_conversation,  # 是否影响对话
            "mappings": self.mappings,  # 槽位映射
        }

    def fingerprint(self) -> Text:
        """返回槽位的唯一哈希值，在Python运行之间保持稳定。

        Returns:
            槽位的指纹
        """
        data = {"slot_name": self.name, "slot_value": self.value}  # 基本数据
        data.update(self.persistence_info())  # 添加持久化信息
        return rasa.shared.utils.io.get_dictionary_fingerprint(data)  # 生成指纹


# =============================================================================
# 具体槽位类型实现
# =============================================================================

class FloatSlot(Slot):
    """存储浮点数值的槽位。
    
    用于存储数值信息，支持最小值和最大值限制，
    可以将数值特征化为机器学习模型可用的特征向量。
    """

    type_name = "float"  # 槽位类型名称

    def __init__(
        self,
        name: Text,                           # 槽位名称
        mappings: List[Dict[Text, Any]],      # 槽位映射
        initial_value: Optional[float] = None,  # 初始值
        value_reset_delay: Optional[int] = None,  # 值重置延迟
        max_value: float = 1.0,               # 最大值
        min_value: float = 0.0,               # 最小值
        influence_conversation: bool = True,  # 是否影响对话
    ) -> None:
        """创建浮点槽位。

        Args:
            name: 槽位名称
            mappings: 槽位映射列表
            initial_value: 初始值
            value_reset_delay: 值重置延迟
            max_value: 最大值
            min_value: 最小值
            influence_conversation: 是否影响对话
            
        Raises:
            InvalidSlotConfigError: 如果最小-最大范围无效
            UserWarning: 如果初始值超出最小-最大范围
        """
        super().__init__(
            name, mappings, initial_value, value_reset_delay, influence_conversation
        )
        self.max_value = max_value  # 设置最大值
        self.min_value = min_value  # 设置最小值

        if min_value >= max_value:  # 如果最小值大于等于最大值
            raise InvalidSlotConfigError(
                "浮点槽位 ('{}') 使用无效范围创建，"
                "最小值为 ({})，最大值为 ({})。请确保 "
                "最小值小于最大值。"
                "".format(self.name, self.min_value, self.max_value)
            )

        if initial_value is not None and not (min_value <= initial_value <= max_value):  # 如果初始值超出范围
            rasa.shared.utils.io.raise_warning(
                f"浮点槽位 ('{self.name}') 使用初始值 "
                f"{self.value} 创建。此值超出配置的最小值 "
                f"({self.min_value}) 和最大值 ({self.max_value}) 范围。"
            )

    def _as_feature(self) -> List[float]:
        """将浮点槽位值转换为特征向量。
        
        Returns:
            包含两个元素的特征向量：[存在标志, 归一化值]
        """
        try:
            # 将值限制在最小值和最大值之间
            capped_value = max(self.min_value, min(self.max_value, float(self.value)))
            if abs(self.max_value - self.min_value) > 0:  # 如果范围大于0
                covered_range = abs(self.max_value - self.min_value)  # 计算范围
            else:
                covered_range = 1  # 避免除零错误
            # 返回特征向量：[存在标志, 归一化值]
            return [1.0, (capped_value - self.min_value) / covered_range]
        except (TypeError, ValueError):  # 如果转换失败
            return [0.0, 0.0]  # 返回默认特征向量

    def persistence_info(self) -> Dict[Text, Any]:
        """返回持久化此槽位所需的相关信息。
        
        Returns:
            包含槽位持久化信息的字典
        """
        d = super().persistence_info()  # 获取父类信息
        d["max_value"] = self.max_value  # 添加最大值
        d["min_value"] = self.min_value  # 添加最小值
        return d  # 返回完整信息

    def _feature_dimensionality(self) -> int:
        """返回特征维度数。
        
        Returns:
            特征向量的长度
        """
        return len(self.as_feature())  # 返回特征向量长度


class BooleanSlot(Slot):
    """存储布尔值的槽位。
    
    用于存储真/假值，支持从多种数据类型转换为布尔值，
    可以将布尔值特征化为机器学习模型可用的特征向量。
    """

    type_name = "bool"  # 槽位类型名称

    def _as_feature(self) -> List[float]:
        """将布尔槽位值转换为特征向量。
        
        Returns:
            包含两个元素的特征向量：[存在标志, 布尔值]
        """
        try:
            if self.value is not None:  # 如果值不为空
                return [1.0, float(bool_from_any(self.value))]  # 返回存在标志和布尔值
            else:
                return [0.0, 0.0]  # 返回默认特征向量
        except (TypeError, ValueError):  # 如果转换失败
            # 我们无法将值转换为浮点数 - 使用默认值
            return [0.0, 0.0]  # 返回默认特征向量

    def _feature_dimensionality(self) -> int:
        """返回特征维度数。
        
        Returns:
            特征向量的长度
        """
        return len(self.as_feature())  # 返回特征向量长度


def bool_from_any(x: Any) -> bool:
    """将 bool/float/int/str 转换为 bool 或抛出错误。
    
    Args:
        x: 要转换的值
        
    Returns:
        转换后的布尔值
        
    Raises:
        ValueError: 如果字符串无法转换为布尔值
        TypeError: 如果类型无法转换为布尔值
    """
    if isinstance(x, bool):  # 如果已经是布尔值
        return x  # 直接返回
    elif isinstance(x, (float, int)):  # 如果是数值类型
        return x == 1.0  # 检查是否等于1.0
    elif isinstance(x, str):  # 如果是字符串
        if x.isnumeric():  # 如果是数字字符串
            return float(x) == 1.0  # 转换为浮点数并检查是否等于1.0
        elif x.strip().lower() == "true":  # 如果是"true"字符串
            return True  # 返回True
        elif x.strip().lower() == "false":  # 如果是"false"字符串
            return False  # 返回False
        else:
            raise ValueError("无法将字符串转换为布尔值")  # 抛出错误
    else:
        raise TypeError("无法转换为布尔值")  # 抛出类型错误


class TextSlot(Slot):
    """存储文本值的槽位。
    
    用于存储文本信息，只关心文本是否存在，
    不关心文本的具体内容。
    """
    type_name = "text"  # 槽位类型名称

    def _as_feature(self) -> List[float]:
        """将文本槽位值转换为特征向量。
        
        Returns:
            包含一个元素的特征向量：[存在标志]
        """
        return [1.0 if self.value is not None else 0.0]  # 返回存在标志


class ListSlot(Slot):
    """存储列表值的槽位。
    
    用于存储列表信息，自动将单个值转换为列表，
    只关心列表是否为空，不关心列表的具体内容。
    """
    type_name = "list"  # 槽位类型名称

    def _as_feature(self) -> List[float]:
        """将列表槽位值转换为特征向量。
        
        Returns:
            包含一个元素的特征向量：[存在标志]
        """
        try:
            if self.value is not None and len(self.value) > 0:  # 如果值不为空且列表不为空
                return [1.0]  # 返回存在标志
            else:
                return [0.0]  # 返回不存在标志
        except (TypeError, ValueError):  # 如果转换失败
            # 我们无法将值转换为列表 - 使用默认值
            return [0.0]  # 返回默认特征向量

    # FIXME: https://github.com/python/mypy/issues/8085
    @Slot.value.setter  # type: ignore[attr-defined,misc]
    def value(self, value: Any) -> None:
        """设置槽位的值。
        
        Args:
            value: 要设置的值
        """
        if value and not isinstance(value, list):  # 如果值不为空且不是列表
            # 确保我们总是存储列表项
            value = [value]  # 将单个值转换为列表

        # 调用父类的属性设置器
        # FIXME: https://github.com/python/mypy/issues/8085
        super(ListSlot, self.__class__).value.fset(self, value)  # type: ignore[attr-defined] # noqa: E501


class CategoricalSlot(Slot):
    """可用于根据其值分支对话的槽位类型。
    
    用于存储分类信息，支持预定义的值列表，
    可以将分类值特征化为机器学习模型可用的特征向量。
    """

    type_name = "categorical"  # 槽位类型名称

    def __init__(
        self,
        name: Text,                           # 槽位名称
        mappings: List[Dict[Text, Any]],      # 槽位映射
        values: Optional[List[Any]] = None,   # 可能的值列表
        initial_value: Any = None,            # 初始值
        value_reset_delay: Optional[int] = None,  # 值重置延迟
        influence_conversation: bool = True,  # 是否影响对话
    ) -> None:
        """创建分类槽位（参见父类获取详细文档字符串）。
        
        Args:
            name: 槽位名称
            mappings: 槽位映射列表
            values: 可能的值列表
            initial_value: 初始值
            value_reset_delay: 值重置延迟
            influence_conversation: 是否影响对话
        """
        super().__init__(
            name, mappings, initial_value, value_reset_delay, influence_conversation
        )
        if values and None in values:  # 如果值列表包含None
            rasa.shared.utils.io.raise_warning(
                f"分类槽位 '{self.name}' 在域文件中列出了 `null` 作为可能值，"
                f"这在Python中转换为 `None`。此值保留用于槽位未设置时，"
                f"不应在槽位定义中列为值。"
                f" Rasa将忽略 '{self.name}' 槽位的 `null` 作为可能值。"
                f" 请考虑在域文件中将此值更改为，例如 `unset`，"
                f"或通过使用引号显式提供字符串值：`"null"`。",
                category=UserWarning,
            )
        # 将值转换为小写字符串，过滤掉None值
        self.values = (
            [str(v).lower() for v in values if v is not None] if values else []
        )

    def add_default_value(self) -> None:
        """将特殊默认值添加到可能值列表中。"""
        values = set(self.values)  # 获取当前值集合
        if rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE not in values:  # 如果默认值不在集合中
            self.values.append(
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
            )  # 添加默认值

    def persistence_info(self) -> Dict[Text, Any]:
        """返回序列化的槽位信息。
        
        Returns:
            包含槽位持久化信息的字典
        """
        d = super().persistence_info()  # 获取父类信息
        d["values"] = [
            value
            for value in self.values
            # 持久化时不添加默认槽位。
            # 我们将在创建域时动态重新添加它。
            if value != rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
        ]  # 过滤掉默认值
        return d  # 返回完整信息

    def _as_feature(self) -> List[float]:
        """将分类槽位值转换为特征向量。
        
        Returns:
            独热编码的特征向量
        """
        r = [0.0] * self.feature_dimensionality()  # 初始化零向量

        # 如果槽位未设置（即设置为None），返回零填充数组。
        # 从概念上讲，这类似于特征化过程失败的情况，
        # 因此这里返回的特征与那种情况相同。
        if self.value is None:  # 如果值为None
            return r  # 返回零向量

        try:
            for i, v in enumerate(self.values):  # 遍历可能的值
                if v == str(self.value).lower():  # 如果找到匹配的值
                    r[i] = 1.0  # 设置对应位置为1
                    break  # 跳出循环
            else:  # 如果没有找到匹配的值
                if (
                    rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
                    in self.values
                ):  # 如果存在默认值
                    i = self.values.index(
                        rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE
                    )  # 找到默认值的索引
                    r[i] = 1.0  # 设置默认值位置为1
                else:  # 如果没有默认值
                    rasa.shared.utils.io.raise_warning(
                        f"分类槽位 '{self.name}' 设置为值 "
                        f"('{self.value}') "
                        "在域中未指定。值将被忽略，槽位将 "
                        "表现得好像没有设置值。 "
                        "请确保将所有分类槽位应存储的值添加到域中。"
                    )  # 发出警告
        except (TypeError, ValueError):  # 如果转换失败
            logger.exception("分类槽位特征化失败。")  # 记录异常
            return r  # 返回零向量
        return r  # 返回特征向量

    def _feature_dimensionality(self) -> int:
        """返回特征维度数。
        
        Returns:
            可能值的数量
        """
        return len(self.values)  # 返回可能值的数量


class AnySlot(Slot):
    """可用于存储任何值的槽位。
    
    用户需要创建 `Slot` 的子类，
    如果信息应该被特征化的话。
    """

    type_name = "any"  # 槽位类型名称

    def __init__(
        self,
        name: Text,                           # 槽位名称
        mappings: List[Dict[Text, Any]],      # 槽位映射
        initial_value: Any = None,            # 初始值
        value_reset_delay: Optional[int] = None,  # 值重置延迟
        influence_conversation: bool = False, # 是否影响对话（默认为False）
    ) -> None:
        """创建任意槽位（参见父类获取详细文档字符串）。
        
        Args:
            name: 槽位名称
            mappings: 槽位映射列表
            initial_value: 初始值
            value_reset_delay: 值重置延迟
            influence_conversation: 是否影响对话
            
        Raises:
            InvalidSlotConfigError: 如果槽位被特征化
        """
        if influence_conversation:  # 如果尝试特征化
            raise InvalidSlotConfigError(
                f"{AnySlot.__name__} 不能被特征化。 "
                f"请为槽位 '{name}' 使用不同的槽位类型。如果您 "
                f"需要特征化不支持开箱即用的数据类型， "
                f"请通过子类化 '{Slot.__name__}' 实现自定义槽位类型。 "
                f"有关更多信息，请参阅文档：{DOCS_URL_SLOTS}"
            )  # 抛出错误

        super().__init__(
            name, mappings, initial_value, value_reset_delay, influence_conversation
        )  # 调用父类构造函数

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。
        
        Args:
            other: 要比较的对象
            
        Returns:
            如果对象相等则返回True，否则返回False
        """
        if not isinstance(other, AnySlot):  # 如果类型不同
            return NotImplemented  # 返回NotImplemented

        return (
            self.name == other.name  # 比较名称
            and self.initial_value == other.initial_value  # 比较初始值
            and self._value_reset_delay == other._value_reset_delay  # 比较重置延迟
            and self.value == other.value  # 比较当前值
        )  # 返回比较结果

    def _as_feature(self) -> List[float]:
        """任意槽位不能特征化。
        
        Raises:
            InvalidSlotConfigError: 总是抛出此异常
        """
        raise InvalidSlotConfigError(
            f"{AnySlot.__name__} 不能被特征化。 "
            f"请为槽位 '{self.name}' 使用不同的槽位类型。如果您 "
            f"需要特征化不支持开箱即用的数据类型， "
            f"请通过子类化 '{Slot.__name__}' 实现自定义槽位类型。 "
            f"有关更多信息，请参阅文档：{DOCS_URL_SLOTS}"
        )  # 抛出错误
