# =============================================================================
# Rasa NLU Featurizer 特征化器基类模块
# 本模块定义了所有特征化器的抽象基类，提供了特征提取和管理的通用接口
# =============================================================================

# 导入未来版本注解支持
from __future__ import annotations

# 导入抽象基类相关模块
from abc import abstractmethod, ABC  # 抽象方法和抽象基类
from collections import Counter  # 计数器
from typing import Generic, Iterable, Text, Optional, Dict, Any, TypeVar  # 类型注解

# 导入 Rasa 核心模块
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS  # 特征化器类别名常量
from rasa.shared.nlu.training_data.features import Features  # 特征类
from rasa.shared.nlu.training_data.message import Message  # 消息类
from rasa.shared.exceptions import InvalidConfigException  # 配置异常
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE  # 特征类型常量

# 定义特征类型类型变量
FeatureType = TypeVar("FeatureType")


class Featurizer(Generic[FeatureType], ABC):
    """所有特征化器的基类。
    
    该类定义了特征化器的通用接口，包括配置验证、特征添加等功能。
    所有具体的特征化器都应该继承此类并实现抽象方法。
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置。
        
        Returns:
            默认配置字典
        """
        return {FEATURIZER_CLASS_ALIAS: None}  # 特征化器类别名默认为 None

    def __init__(self, name: Text, config: Dict[Text, Any]) -> None:
        """实例化一个新的特征化器。

        Args:
            name: 可用作标识符的名称，在配置未指定 `alias` 时使用
            config: 配置字典
        """
        super().__init__()  # 调用父类初始化
        self.validate_config(config)  # 验证配置
        self._config = config  # 存储配置
        self._identifier = self._config[FEATURIZER_CLASS_ALIAS] or name  # 设置标识符

    @classmethod
    @abstractmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """验证组件配置是否正确。
        
        Args:
            config: 要验证的配置字典
            
        Raises:
            InvalidConfigException: 如果配置无效
        """
        ...

    def add_features_to_message(
        self,
        sequence: FeatureType,
        sentence: Optional[FeatureType],
        attribute: Text,
        message: Message,
    ) -> None:
        """将属性的序列和句子特征添加到给定消息中。

        Args:
            sequence: 序列特征矩阵
            sentence: 句子特征矩阵
            attribute: 两个特征描述的属性
            message: 要添加特征的消息
        """
        # 遍历序列和句子特征
        for type, features in [
            (FEATURE_TYPE_SEQUENCE, sequence),  # 序列特征
            (FEATURE_TYPE_SENTENCE, sentence),  # 句子特征
        ]:
            if features is not None:  # 如果特征不为空
                # 创建特征包装器
                wrapped_feature = Features(features, type, attribute, self._identifier)
                # 将特征添加到消息中
                message.add_features(wrapped_feature)

    @staticmethod
    def raise_if_featurizer_configs_are_not_compatible(
        featurizer_configs: Iterable[Dict[Text, Any]]
    ) -> None:
        """验证给定的特征化器配置是否可以一起使用。

        Args:
            featurizer_configs: 特征化器配置的可迭代对象
            
        Raises:
            InvalidConfigException: 如果给定的特征化器不应在同一图中使用
        """
        # 注意：这假设通过执行上下文给出的名称是唯一的
        alias_counter = Counter(  # 创建别名计数器
            config[FEATURIZER_CLASS_ALIAS]  # 获取配置中的别名
            for config in featurizer_configs  # 遍历所有配置
            if FEATURIZER_CLASS_ALIAS in config  # 如果配置中包含别名
        )
        if not alias_counter:  # 没有找到别名
            return
        if alias_counter.most_common(1)[0][1] > 1:  # 如果最常见的别名出现次数大于1
            raise InvalidConfigException(
                f"Expected the featurizers to have unique names but found "
                f" (name, count): {alias_counter.most_common()}. "
                f"Please update your config such that each featurizer has a unique "
                f"alias."
            )
