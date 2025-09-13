# 导入未来版本的注解支持，允许在类型注解中使用前向引用
from __future__ import annotations
# 导入抽象基类模块，用于定义抽象方法
import abc
# 导入深拷贝模块，用于创建对象的深拷贝
import copy
# 导入日志模块，用于记录日志信息
import logging
# 导入枚举模块，用于定义枚举类型
from enum import Enum
# 导入路径处理模块，用于处理文件路径
from pathlib import Path
# 导入事件类，用于处理对话中的事件
from rasa.shared.core.events import Event
# 导入类型注解相关的模块
from typing import (
    Any,        # 任意类型
    List,       # 列表类型
    Optional,   # 可选类型
    Text,       # 文本类型（字符串的别名）
    Dict,       # 字典类型
    Callable,   # 可调用类型
    Tuple,      # 元组类型
    TypeVar,    # 类型变量
    TYPE_CHECKING,  # 类型检查标志
)

# 导入numpy数值计算库
import numpy as np

# 导入Rasa引擎相关模块
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入特征化相关模块
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
# 导入工具模块
import rasa.utils.common
import rasa.shared.utils.io
# 导入异常类
from rasa.shared.exceptions import RasaException, FileIOException
# 导入NLU相关常量
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT, ACTION_TEXT, ACTION_NAME
# 导入核心领域和状态相关模块
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
# 导入核心常量
from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,  # 默认策略优先级
    POLICY_PRIORITY,          # 策略优先级键名
    POLICY_MAX_HISTORY,       # 策略最大历史长度键名
)
from rasa.shared.core.constants import USER, SLOTS, PREVIOUS_ACTION, ACTIVE_LOOP
import rasa.shared.utils.common


# 类型检查时的导入，避免循环导入
if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


# 创建日志记录器
logger = logging.getLogger(__name__)

# 定义跟踪器列表的类型变量，可以是DialogueStateTracker或TrackerWithCachedStates的列表
TrackerListTypeVar = TypeVar(
    "TrackerListTypeVar", List[DialogueStateTracker], List[TrackerWithCachedStates]
)


class SupportedData(Enum):
    """策略支持的训练数据类型枚举。"""

    # 策略仅支持基于机器学习的训练数据（"stories"）
    ML_DATA = 1

    # 策略仅支持基于规则的数据（"rules"）
    RULE_DATA = 2

    # 策略同时支持基于机器学习和基于规则的数据（"stories"和"rules"）
    ML_AND_RULE_DATA = 3

    @staticmethod
    def trackers_for_supported_data(
        supported_data: SupportedData,
        trackers: TrackerListTypeVar,
    ) -> TrackerListTypeVar:
        """根据策略支持的数据类型返回相应的跟踪器。

        Args:
            supported_data: 用于过滤跟踪器的支持数据类型。
            trackers: 要分割的跟踪器列表。

        Returns:
            来自基于机器学习的训练数据和/或基于规则数据的跟踪器。
        """
        # 如果策略只支持规则数据，返回所有规则跟踪器
        if supported_data == SupportedData.RULE_DATA:
            return [tracker for tracker in trackers if tracker.is_rule_tracker]

        # 如果策略只支持机器学习数据，返回所有非规则跟踪器
        if supported_data == SupportedData.ML_DATA:
            return [tracker for tracker in trackers if not tracker.is_rule_tracker]

        # 如果策略支持两种数据类型，返回所有跟踪器
        # `supported_data` 是 `SupportedData.ML_AND_RULE_DATA`
        return trackers


class Policy(GraphComponent):
    """所有对话策略的通用父类。"""

    @staticmethod
    def supported_data() -> SupportedData:
        """此策略支持的数据类型。

        默认情况下，仅支持基于机器学习的训练数据。如果策略支持规则数据，
        或同时支持基于机器学习的数据和规则数据，则需要重写此方法。

        Returns:
            此策略支持的数据类型（基于机器学习的训练数据）。
        """
        return SupportedData.ML_DATA

    def __init__(
        self,
        config: Dict[Text, Any],           # 策略配置字典
        model_storage: ModelStorage,       # 模型存储对象
        resource: Resource,                # 资源对象
        execution_context: ExecutionContext,  # 执行上下文
        featurizer: Optional[TrackerFeaturizer] = None,  # 可选的跟踪器特征化器
    ) -> None:
        """构造一个新的策略对象。"""
        # 保存配置
        self.config = config
        # 如果没有提供特征化器，则创建一个默认的
        if featurizer is None:
            featurizer = self._create_featurizer()
        # 保存特征化器（使用私有属性）
        self.__featurizer = featurizer

        # 从配置中获取策略优先级，如果没有则使用默认值
        self.priority = config.get(POLICY_PRIORITY, DEFAULT_POLICY_PRIORITY)
        # 设置微调模式标志
        self.finetune_mode = execution_context.is_finetuning

        # 保存模型存储和资源对象
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],           # 策略配置
        model_storage: ModelStorage,       # 模型存储
        resource: Resource,                # 资源
        execution_context: ExecutionContext,  # 执行上下文
        **kwargs: Any,                     # 其他关键字参数
    ) -> Policy:
        """创建一个新的未训练策略（完整文档字符串请参见父类）。"""
        return cls(config, model_storage, resource, execution_context)

    def _create_featurizer(self) -> TrackerFeaturizer:
        """创建策略的特征化器。"""
        # 深拷贝策略配置，避免修改原始配置
        policy_config = copy.deepcopy(self.config)

        # 从配置中获取特征化器配置
        featurizer_configs = policy_config.get("featurizer")

        # 如果没有特征化器配置，使用标准特征化器
        if not featurizer_configs:
            return self._standard_featurizer()

        # 从配置中获取特征化器函数
        featurizer_func = _get_featurizer_from_config(
            featurizer_configs,
            self.__class__.__name__,
            lookup_path="rasa.core.featurizers.tracker_featurizers",
        )
        # 获取第一个特征化器配置
        featurizer_config = featurizer_configs[0]

        # 检查是否有状态特征化器配置
        state_featurizer_configs = featurizer_config.pop("state_featurizer", None)
        if state_featurizer_configs:
            # 获取状态特征化器函数
            state_featurizer_func = _get_featurizer_from_config(
                state_featurizer_configs,
                self.__class__.__name__,
                lookup_path="rasa.core.featurizers.single_state_featurizer",
            )
            # 获取第一个状态特征化器配置
            state_featurizer_config = state_featurizer_configs[0]

            # 创建状态特征化器并添加到特征化器配置中
            featurizer_config["state_featurizer"] = state_featurizer_func(
                **state_featurizer_config
            )

        # 创建特征化器实例
        featurizer = featurizer_func(**featurizer_config)
        # 如果是最大历史跟踪器特征化器，且策略配置中有最大历史长度设置
        # 但特征化器配置中没有，则从策略配置中设置
        if (
            isinstance(featurizer, MaxHistoryTrackerFeaturizer)
            and POLICY_MAX_HISTORY in policy_config
            and POLICY_MAX_HISTORY not in featurizer_config
        ):
            featurizer.max_history = policy_config[POLICY_MAX_HISTORY]
        return featurizer

    def _standard_featurizer(self) -> MaxHistoryTrackerFeaturizer:
        """为此策略初始化标准特征化器。"""
        return MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), self.config.get(POLICY_MAX_HISTORY)
        )

    @property
    def featurizer(self) -> TrackerFeaturizer:
        """返回策略的特征化器。"""
        return self.__featurizer

    @staticmethod
    def _get_valid_params(func: Callable, **kwargs: Any) -> Dict:
        """过滤出可以传递给函数的参数。

        Args:
            func: 可调用函数

        Returns:
            参数字典
        """
        # 获取函数接受的参数名
        valid_keys = rasa.shared.utils.common.arguments_of(func)

        # 过滤出有效的参数
        params = {key: kwargs.get(key) for key in valid_keys if kwargs.get(key)}
        # 记录被忽略的参数
        ignored_params = {
            key: kwargs.get(key) for key in kwargs.keys() if not params.get(key)
        }
        logger.debug(f"Parameters ignored by `model.fit(...)`: {ignored_params}")
        return params

    def _featurize_for_training(
        self,
        training_trackers: List[DialogueStateTracker],  # 训练跟踪器列表
        domain: Domain,                                  # 领域对象
        precomputations: Optional[MessageContainerForCoreFeaturization],  # 预计算特征
        bilou_tagging: bool = False,                     # 是否使用BILOU标记
        **kwargs: Any,                                   # 其他关键字参数
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],  # 状态特征
        np.ndarray,                               # 标签ID
        List[List[Dict[Text, List[Features]]]],  # 实体标签
    ]:
        """将训练跟踪器转换为向量表示。

        由多个轮次组成的跟踪器将被转换为机器学习模型可以使用的浮点向量。

        Args:
            training_trackers: 对话状态跟踪器列表
            domain: 领域对象
            precomputations: 包含预计算特征和属性
            bilou_tagging: 指示是否应使用BILOU标记

        Returns:
            - 属性字典（INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, FORM）
              到所有训练跟踪器中所有对话轮次特征列表的映射
            - 所有训练跟踪器中每个对话轮次的标签ID（例如动作ID）
            - 实体类型字典（ENTITY_TAGS）到包含文本用户输入实体标签ID的特征列表，
              否则为空字典，用于所有训练跟踪器中的所有对话轮次
        """
        # 使用特征化器对跟踪器进行特征化
        state_features, label_ids, entity_tags = self.featurizer.featurize_trackers(
            training_trackers,
            domain,
            precomputations=precomputations,
            bilou_tagging=bilou_tagging,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

        # 检查是否有最大训练样本数限制
        max_training_samples = kwargs.get("max_training_samples")
        if max_training_samples is not None:
            logger.debug(
                "Limit training data to {} training samples."
                "".format(max_training_samples)
            )
            # 限制训练数据到指定的样本数
            state_features = state_features[:max_training_samples]
            label_ids = label_ids[:max_training_samples]
            entity_tags = entity_tags[:max_training_samples]

        return state_features, label_ids, entity_tags

    def _prediction_states(
        self,
        tracker: DialogueStateTracker,                    # 要特征化的跟踪器
        domain: Domain,                                   # 领域对象
        use_text_for_last_user_input: bool = False,       # 是否使用文本而非意图标签
        rule_only_data: Optional[Dict[Text, Any]] = None, # 仅规则数据
    ) -> List[State]:
        """将跟踪器转换为用于预测的状态。

        Args:
            tracker: 要特征化的跟踪器
            domain: 领域对象
            use_text_for_last_user_input: 指示是否使用文本或意图标签
                来特征化最后一个用户输入
            rule_only_data: 特定于规则的槽和循环，因此应被此策略忽略

        Returns:
            状态列表
        """
        return self.featurizer.prediction_states(
            [tracker],
            domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )[0]

    def _featurize_for_prediction(
        self,
        tracker: DialogueStateTracker,                    # 要特征化的跟踪器
        domain: Domain,                                   # 领域对象
        precomputations: Optional[MessageContainerForCoreFeaturization],  # 预计算特征
        rule_only_data: Optional[Dict[Text, Any]],        # 仅规则数据
        use_text_for_last_user_input: bool = False,       # 是否使用文本而非意图标签
    ) -> List[List[Dict[Text, List[Features]]]]:
        """将训练跟踪器转换为向量表示。

        由多个轮次组成的跟踪器将被转换为机器学习模型可以使用的浮点向量。

        Args:
            tracker: 要特征化的跟踪器
            domain: 领域对象
            precomputations: 包含预计算特征和属性
            use_text_for_last_user_input: 指示是否使用文本或意图标签
                来特征化最后一个用户输入
            rule_only_data: 特定于规则的槽和循环，因此应被此策略忽略

        Returns:
            列表（对应跟踪器列表）
            的列表（对应所有对话轮次）
            的字典，状态类型（INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, ACTIVE_LOOP）
            到所有跟踪器中所有对话轮次特征列表的映射
        """
        return self.featurizer.create_state_features(
            [tracker],
            domain,
            precomputations=precomputations,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

    @abc.abstractmethod
    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],  # 训练跟踪器列表
        domain: Domain,                                     # 模型领域
        **kwargs: Any,                                      # 其他关键字参数
    ) -> Resource:
        """训练策略。

        Args:
            training_trackers: 来自训练数据的故事和规则跟踪器
            domain: 模型的领域
            **kwargs: 根据指定的`needs`部分和生成的图结构，
                策略可以使用不同的输入来训练自己

        Returns:
            策略必须返回其资源定位符，以便潜在的子节点可以从资源中加载策略
        """
        raise NotImplementedError("Policy must have the capacity to train.")

    @abc.abstractmethod
    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,                    # 包含对话历史的跟踪器
        domain: Domain,                                   # 模型领域
        rule_only_data: Optional[Dict[Text, Any]] = None, # 仅规则数据
        **kwargs: Any,                                    # 其他关键字参数
    ) -> PolicyPrediction:
        """预测机器人在看到跟踪器后应该采取的下一步动作。

        Args:
            tracker: 包含到目前为止对话历史的跟踪器
            domain: 模型的领域
            rule_only_data: 特定于规则的槽和循环，因此应被此策略忽略
            **kwargs: 根据指定的`needs`部分和生成的图结构，
                策略可以使用不同的输入来进行预测

        Returns:
            预测结果
        """
        raise NotImplementedError("Policy must have the capacity to predict.")

    def _prediction(
        self,
        probabilities: List[float],                        # 概率列表
        events: Optional[List[Event]] = None,              # 事件列表
        optional_events: Optional[List[Event]] = None,     # 可选事件列表
        is_end_to_end_prediction: bool = False,            # 是否为端到端预测
        is_no_user_prediction: bool = False,               # 是否为无用户预测
        diagnostic_data: Optional[Dict[Text, Any]] = None, # 诊断数据
        action_metadata: Optional[Dict[Text, Any]] = None, # 动作元数据
    ) -> PolicyPrediction:
        """创建策略预测对象。"""
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events,
            optional_events,
            is_end_to_end_prediction,
            is_no_user_prediction,
            diagnostic_data,
            action_metadata=action_metadata,
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],           # 策略配置
        model_storage: ModelStorage,       # 模型存储
        resource: Resource,                # 资源
        execution_context: ExecutionContext,  # 执行上下文
        **kwargs: Any,                     # 其他关键字参数
    ) -> Policy:
        """加载已训练的策略（完整文档字符串请参见父类）。"""
        featurizer = None

        try:
            # 从资源中读取模型存储路径
            with model_storage.read_from(resource) as path:
                # 检查是否存在特征化器文件
                if (Path(path) / FEATURIZER_FILE).is_file():
                    featurizer = TrackerFeaturizer.load(path)

                # 更新配置
                config.update(kwargs)

        except (ValueError, FileNotFoundError, FileIOException):
            logger.debug(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )

        return cls(
            config, model_storage, resource, execution_context, featurizer=featurizer
        )

    def _default_predictions(self, domain: Domain) -> List[float]:
        """创建零值列表。

        Args:
            domain: 领域对象
        Returns:
            长度为动作数量的零值列表
        """
        return [0.0] * domain.num_actions

    @staticmethod
    def format_tracker_states(states: List[Dict]) -> Text:
        """将跟踪器状态格式化为调试日志中的人类可读格式。

        Args:
            states: 跟踪器状态字典列表

        Returns:
            包含用户意图和动作的状态字符串
        """
        # 空字符串用于在第一个状态前插入换行符
        formatted_states = [""]
        if states:
            for index, state in enumerate(states):
                state_messages = []
                if state:
                    # 处理用户相关信息
                    if USER in state:
                        if TEXT in state[USER]:
                            state_messages.append(
                                f"user text: {str(state[USER][TEXT])}"
                            )
                        if INTENT in state[USER]:
                            state_messages.append(
                                f"user intent: {str(state[USER][INTENT])}"
                            )
                        if ENTITIES in state[USER]:
                            state_messages.append(
                                f"user entities: {str(state[USER][ENTITIES])}"
                            )
                    # 处理前一个动作信息
                    if PREVIOUS_ACTION in state:
                        if ACTION_NAME in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action name: "
                                f"{str(state[PREVIOUS_ACTION][ACTION_NAME])}"
                            )
                        if ACTION_TEXT in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action text: "
                                f"{str(state[PREVIOUS_ACTION][ACTION_TEXT])}"
                            )
                    # 处理活跃循环信息
                    if ACTIVE_LOOP in state:
                        state_messages.append(f"active loop: {str(state[ACTIVE_LOOP])}")
                    # 处理槽信息
                    if SLOTS in state:
                        state_messages.append(f"slots: {str(state[SLOTS])}")
                    # 格式化状态消息
                    state_message_formatted = " | ".join(state_messages)
                    state_formatted = f"[state {str(index)}] {state_message_formatted}"
                    formatted_states.append(state_formatted)

        return "\n".join(formatted_states)

    def __repr__(self) -> Text:
        """返回对象的文本表示。"""
        return f"{self.__class__.__name__}@{id(self)}"


class PolicyPrediction:
    """存储策略预测信息的类。"""

    def __init__(
        self,
        probabilities: List[float],                        # 每个动作的概率
        policy_name: Optional[Text],                       # 进行预测的策略名称
        policy_priority: int = 1,                          # 策略优先级
        events: Optional[List[Event]] = None,              # 事件列表
        optional_events: Optional[List[Event]] = None,     # 可选事件列表
        is_end_to_end_prediction: bool = False,            # 是否为端到端预测
        is_no_user_prediction: bool = False,               # 是否为无用户预测
        diagnostic_data: Optional[Dict[Text, Any]] = None, # 诊断数据
        hide_rule_turn: bool = False,                      # 是否隐藏规则轮次
        action_metadata: Optional[Dict[Text, Any]] = None, # 动作元数据
    ) -> None:
        """创建策略预测对象。

        Args:
            probabilities: 每个动作的概率
            policy_name: 进行预测的策略名称
            policy_priority: 进行预测的策略优先级
            events: 策略需要在预测后应用到跟踪器的事件。
                这些事件的应用与策略是否胜过其他策略无关。
                请注意返回的事件，因为它们可能会影响对话流程
            optional_events: 策略在获胜情况下需要在预测后应用到跟踪器的事件。
                这些事件仅在策略预测获胜时应用。
                请注意返回的事件，因为它们可能会影响对话流程
            is_end_to_end_prediction: 如果预测使用用户消息的文本而不是意图，则为`True`
            is_no_user_prediction: 如果预测既不使用用户消息的文本也不使用意图，则为`True`。
                例如，这是快乐循环路径的情况
            diagnostic_data: 中间结果或其他信息，这些信息对Rasa的功能不是必需的，
                但用于调试和微调目的
            hide_rule_turn: 如果预测是由不出现在故事中的规则做出的，则为`True`
            action_metadata: 指定策略可以传递的额外元数据
        """
        self.probabilities = probabilities
        self.policy_name = policy_name
        self.policy_priority = policy_priority
        self.events = events or []
        self.optional_events = optional_events or []
        self.is_end_to_end_prediction = is_end_to_end_prediction
        self.is_no_user_prediction = is_no_user_prediction
        self.diagnostic_data = diagnostic_data or {}
        self.hide_rule_turn = hide_rule_turn
        self.action_metadata = action_metadata

    @staticmethod
    def for_action_name(
        domain: Domain,                                    # 当前模型领域
        action_name: Text,                                 # 要预测的动作名称
        policy_name: Optional[Text] = None,                # 进行预测的策略名称
        confidence: float = 1.0,                           # 预测置信度
        action_metadata: Optional[Dict[Text, Any]] = None, # 动作元数据
    ) -> "PolicyPrediction":
        """为给定动作创建预测。

        Args:
            domain: 当前模型领域
            action_name: 应该预测的动作
            policy_name: 进行预测的策略
            confidence: 预测置信度
            action_metadata: 要附加到预测的额外元数据

        Returns:
            预测结果
        """
        # 为指定动作创建置信度分数
        probabilities = confidence_scores_for(action_name, confidence, domain)

        return PolicyPrediction(
            probabilities, policy_name, action_metadata=action_metadata
        )

    def __eq__(self, other: Any) -> bool:
        """检查两个对象是否相等。

        Args:
            other: 任何其他对象

        Returns:
            如果other具有相同类型且值相同，则为`True`
        """
        if not isinstance(other, PolicyPrediction):
            return False

        return (
            self.probabilities == other.probabilities
            and self.policy_name == other.policy_name
            and self.policy_priority == other.policy_priority
            and self.events == other.events
            and self.optional_events == other.optional_events
            and self.is_end_to_end_prediction == other.is_end_to_end_prediction
            and self.is_no_user_prediction == other.is_no_user_prediction
            and self.hide_rule_turn == other.hide_rule_turn
            and self.action_metadata == other.action_metadata
            # 我们不比较`diagnostic_data`，因为它对动作预测没有影响
        )

    @property
    def max_confidence_index(self) -> int:
        """获取具有最高置信度的动作预测的索引。

        Returns:
            具有最高置信度的动作的索引
        """
        return self.probabilities.index(self.max_confidence)

    @property
    def max_confidence(self) -> float:
        """获取最高预测置信度。

        Returns:
            最高预测置信度
        """
        return max(self.probabilities, default=0.0)


def confidence_scores_for(
    action_name: Text, value: float, domain: Domain
) -> List[float]:
    """如果预测单个动作，返回置信度分数。

    Args:
        action_name: 应设置分数的动作名称
        value: `action_name`的置信度
        domain: 领域对象

    Returns:
        长度为动作数量的列表
    """
    # 创建全零的置信度分数列表
    results = [0.0] * domain.num_actions
    # 获取动作的索引
    idx = domain.index_for_action(action_name)
    # 设置指定动作的置信度分数
    results[idx] = value

    return results


class InvalidPolicyConfig(RasaException):
    """当策略配置无效时可以引发的异常。"""


def _get_featurizer_from_config(
    config: List[Dict[Text, Any]], policy_name: Text, lookup_path: Text
) -> Callable[..., TrackerFeaturizer]:
    """从策略配置中获取特征化器初始化器及其参数。"""
    # 只允许1个特征化器
    if len(config) > 1:
        featurizer_names = [
            featurizer_config.get("name") for featurizer_config in config
        ]
        raise InvalidPolicyConfig(
            f"Every policy can only have 1 featurizer but '{policy_name}' "
            f"uses {len(config)} featurizers ('{', '.join(featurizer_names)}')."
        )

    # 获取第一个特征化器配置
    featurizer_config = config[0]
    # 提取特征化器名称
    featurizer_name = featurizer_config.pop("name")
    # 从模块路径获取特征化器类
    featurizer_func = rasa.shared.utils.common.class_from_module_path(
        featurizer_name, lookup_path=lookup_path
    )

    return featurizer_func
