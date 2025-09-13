# =============================================================================
# 事件系统模块 - 定义对话中的各种事件类型
# =============================================================================
# 此模块定义了 Rasa Core 中所有的事件类型，包括用户消息、机器人响应、
# 槽位设置、动作执行等。事件是对话状态跟踪的基础，用于记录对话中
# 发生的所有重要变化。

# 标准库导入
import abc                    # 抽象基类支持
import copy                   # 深拷贝功能
import json                   # JSON 处理
import logging                # 日志记录
import structlog              # 结构化日志
import re                     # 正则表达式
from abc import ABC           # 抽象基类

# 第三方库导入
import jsonpickle             # JSON 序列化库
import time                   # 时间处理
import uuid                   # UUID 生成
from dateutil import parser   # 日期解析
from datetime import datetime # 日期时间类

# 类型提示导入
from typing import (
    List,                     # 列表类型
    Dict,                     # 字典类型
    Text,                     # 文本类型
    Any,                      # 任意类型
    Type,                     # 类型类型
    Optional,                 # 可选类型
    TYPE_CHECKING,            # 类型检查
    Iterable,                 # 可迭代类型
    cast,                     # 类型转换
    Tuple,                    # 元组类型
    TypeVar,                  # 类型变量
)

# Rasa 内部模块导入
import rasa.shared.utils.common  # 通用工具函数
import rasa.shared.utils.io      # IO 工具函数
from typing import Union          # 联合类型

# 常量导入
from rasa.shared.constants import DOCS_URL_TRAINING_DATA  # 训练数据文档URL
from rasa.shared.core.constants import (
    LOOP_NAME,                        # 循环名称
    EXTERNAL_MESSAGE_PREFIX,          # 外部消息前缀
    ACTION_NAME_SENDER_ID_CONNECTOR_STR,  # 动作名称发送者ID连接符
    IS_EXTERNAL,                      # 是否为外部事件
    USE_TEXT_FOR_FEATURIZATION,       # 是否使用文本进行特征化
    LOOP_INTERRUPTED,                 # 循环中断
    ENTITY_LABEL_SEPARATOR,           # 实体标签分隔符
    ACTION_SESSION_START_NAME,        # 会话开始动作名称
    ACTION_LISTEN_NAME,               # 监听动作名称
)
from rasa.shared.exceptions import UnsupportedFeatureException  # 不支持功能异常
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_TYPE,            # 实体属性类型
    INTENT,                           # 意图
    TEXT,                             # 文本
    ENTITIES,                         # 实体
    ENTITY_ATTRIBUTE_VALUE,           # 实体属性值
    ACTION_TEXT,                      # 动作文本
    ACTION_NAME,                      # 动作名称
    INTENT_NAME_KEY,                  # 意图名称键
    ENTITY_ATTRIBUTE_ROLE,            # 实体属性角色
    ENTITY_ATTRIBUTE_GROUP,           # 实体属性组
    PREDICTED_CONFIDENCE_KEY,         # 预测置信度键
    INTENT_RANKING_KEY,               # 意图排名键
    ENTITY_ATTRIBUTE_TEXT,            # 实体属性文本
    ENTITY_ATTRIBUTE_START,           # 实体属性开始位置
    ENTITY_ATTRIBUTE_CONFIDENCE,      # 实体属性置信度
    ENTITY_ATTRIBUTE_END,             # 实体属性结束位置
    FULL_RETRIEVAL_INTENT_NAME_KEY,   # 完整检索意图名称键
)


# =============================================================================
# 类型定义和日志配置
# =============================================================================
if TYPE_CHECKING:
    # 仅在类型检查时导入，避免循环导入
    from typing_extensions import TypedDict  # 类型字典

    from rasa.shared.core.trackers import DialogueStateTracker  # 对话状态跟踪器

    # 实体预测类型定义
    EntityPrediction = TypedDict(
        "EntityPrediction",
        {
            ENTITY_ATTRIBUTE_TEXT: Text,  # type: ignore[misc]  # 实体文本
            ENTITY_ATTRIBUTE_START: Optional[float],           # 实体开始位置
            ENTITY_ATTRIBUTE_END: Optional[float],             # 实体结束位置
            ENTITY_ATTRIBUTE_VALUE: Text,                      # 实体值
            ENTITY_ATTRIBUTE_CONFIDENCE: float,                # 实体置信度
            ENTITY_ATTRIBUTE_TYPE: Text,                       # 实体类型
            ENTITY_ATTRIBUTE_GROUP: Optional[Text],            # 实体组
            ENTITY_ATTRIBUTE_ROLE: Optional[Text],             # 实体角色
            "additional_info": Any,                            # 附加信息
        },
        total=False,  # 允许部分字段缺失
    )

    # 意图预测类型定义
    IntentPrediction = TypedDict(
        "IntentPrediction", 
        {
            INTENT_NAME_KEY: Text,                    # type: ignore[misc]  # 意图名称
            PREDICTED_CONFIDENCE_KEY: float           # 预测置信度
        }
    )
    
    # NLU 预测数据类型定义
    NLUPredictionData = TypedDict(
        "NLUPredictionData",
        {
            TEXT: Text,  # type: ignore[misc]                    # 文本
            INTENT: IntentPrediction,                            # 意图预测
            INTENT_RANKING_KEY: List[IntentPrediction],         # 意图排名
            ENTITIES: List[EntityPrediction],                    # 实体列表
            "message_id": Optional[Text],                        # 消息ID
            "metadata": Dict,                                   # 元数据
        },
        total=False,  # 允许部分字段缺失
    )

# 日志记录器配置
logger = logging.getLogger(__name__)        # 标准日志记录器
structlogger = structlog.get_logger()      # 结构化日志记录器


# =============================================================================
# 事件序列化和反序列化工具函数
# =============================================================================

def deserialise_events(serialized_events: List[Dict[Text, Any]]) -> List["Event"]:
    """将字典列表转换为对应的事件列表。
    
    此函数用于从序列化的事件数据中重建事件对象，
    支持从数据库或文件中恢复对话历史。

    Args:
        serialized_events: 序列化的事件字典列表
        
    Returns:
        重建的事件对象列表
        
    Example:
        [{"event": "slot", "value": 5, "name": "my_slot"}]
    """
    deserialised = []  # 反序列化的事件列表

    for e in serialized_events:  # 遍历每个序列化事件
        if "event" in e:  # 检查是否包含事件类型
            event = Event.from_parameters(e)  # 从参数创建事件
            if event:  # 如果事件创建成功
                deserialised.append(event)  # 添加到结果列表
            else:  # 如果事件创建失败
                structlogger.warning(
                    "event.deserialization.failed", 
                    rasa_event=copy.deepcopy(event)  # 记录警告信息
                )

    return deserialised  # 返回反序列化的事件列表


def deserialise_entities(entities: Union[Text, List[Any]]) -> List[Dict[Text, Any]]:
    """反序列化实体数据。
    
    将实体数据从字符串或列表格式转换为字典列表格式，
    用于处理从不同来源获取的实体信息。
    
    Args:
        entities: 实体数据，可以是JSON字符串或实体列表
        
    Returns:
        实体字典列表
    """
    if isinstance(entities, str):  # 如果是字符串格式
        entities = json.loads(entities)  # 解析JSON字符串

    return [e for e in entities if isinstance(e, dict)]  # 返回字典类型的实体


def format_message(
    text: Text, intent: Optional[Text], entities: Union[Text, List[Any]]
) -> Text:
    """使用NLU解析器信息生成带有内联实体注释的消息。
    
    此函数将用户消息、意图和实体信息组合成格式化的消息，
    用于训练数据的生成和显示。

    Args:
        text: 消息文本
        intent: 消息意图
        entities: 消息实体

    Returns:
        带有内联实体注释的消息，例如：
        `I am from [Berlin]{"entity": "city"}`
    """
    # 导入训练数据相关模块
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter
    from rasa.shared.nlu.training_data import entities_parser

    # 解析训练示例
    message_from_md = entities_parser.parse_training_example(text, intent)
    # 反序列化实体
    deserialised_entities = deserialise_entities(entities)
    # 生成格式化消息
    return TrainingDataWriter.generate_message(
        {"text": message_from_md.get(TEXT), "entities": deserialised_entities}
    )


def split_events(
    events: Iterable["Event"],
    event_type_to_split_on: Type["Event"],
    additional_splitting_conditions: Optional[Dict[Text, Any]] = None,
    include_splitting_event: bool = True,
) -> List[List["Event"]]:
    """根据事件类型和条件分割事件列表。
    
    此函数用于将事件列表按照特定的事件类型和条件进行分割，
    常用于将长对话分割成多个会话或场景。

    Examples:
        按照 `ActionExecuted` 事件类型和 `action_name` 为 'action_session_start' 的条件
        分割事件列表：

        >> _events = split_events(
                        events,
                        ActionExecuted,
                        {"action_name": "action_session_start"},
                        True
                     )

    Args:
        events: 要分割的事件列表
        event_type_to_split_on: 用于分割的事件类型
        additional_splitting_conditions: 额外的分割条件（事件属性）
        include_splitting_event: 是否在返回的事件中包含分割事件本身

    Returns:
        分割后的事件列表
    """
    sub_events = []  # 分割后的事件组列表
    current: List["Event"] = []  # 当前事件组

    def event_fulfills_splitting_condition(evt: "Event") -> bool:
        """检查事件是否满足分割条件。
        
        Args:
            evt: 要检查的事件
            
        Returns:
            是否满足分割条件
        """
        # 检查事件类型是否正确
        if not isinstance(evt, event_type_to_split_on):
            return False

        # 如果类型正确且没有其他条件
        if not additional_splitting_conditions:
            return True

        # 如果有其他条件，检查这些条件
        return all(
            getattr(evt, k, None) == v
            for k, v in additional_splitting_conditions.items()
        )

    for event in events:  # 遍历所有事件
        if event_fulfills_splitting_condition(event):  # 如果事件满足分割条件
            if current:  # 如果当前组不为空
                sub_events.append(current)  # 将当前组添加到结果中

            current = []  # 重置当前组
            if include_splitting_event:  # 如果包含分割事件
                current.append(event)  # 将分割事件添加到新组
        else:  # 如果事件不满足分割条件
            current.append(event)  # 将事件添加到当前组

    if current:  # 如果最后还有未处理的事件
        sub_events.append(current)  # 添加到结果中

    return sub_events  # 返回分割后的事件列表


def do_events_begin_with_session_start(events: List["Event"]) -> bool:
    """判断事件列表是否以会话开始序列开始。
    
    会话开始序列由两个事件组成：一个执行的 `action_session_start` 动作
    和一个记录的 `session_started` 事件。

    Args:
        events: 要检查的事件列表

    Returns:
        事件列表是否以会话开始序列开始
    """
    if len(events) < 2:  # 如果事件数量少于2个
        return False  # 不可能有会话开始序列

    first = events[0]   # 第一个事件
    second = events[1]  # 第二个事件

    # 我们不关心特定的元数据或时间戳。动作名称和事件类型
    # 足以进行此检查
    return (
        isinstance(first, ActionExecuted)  # 第一个事件是动作执行
        and first.action_name == ACTION_SESSION_START_NAME  # 动作名称是会话开始
        and isinstance(second, SessionStarted)  # 第二个事件是会话开始
    )


def remove_parse_data(event: Dict[Text, Any]) -> Dict[Text, Any]:
    """将事件详情减少到结构化日志记录所需的最小值。
    
    此函数用于优化日志记录，删除事件中不必要的解析数据，
    以减少日志文件的大小和提高性能。

    Args:
        event: 要减少的事件

    Returns:
        减少后的事件副本
    """
    reduced_event = copy.deepcopy(event)  # 深拷贝事件
    if "parse_data" in reduced_event:  # 如果存在解析数据
        del reduced_event["parse_data"]  # 删除解析数据
    return reduced_event  # 返回减少后的事件


# =============================================================================
# 事件基类和核心事件类型
# =============================================================================

E = TypeVar("E", bound="Event")  # 事件类型变量


class Event(ABC):
    """描述对话中的事件以及它们如何影响对话状态。
    
    这是所有事件类型的抽象基类，提供了事件的基本结构和行为。
    事件是对话中发生的一切的不可变表示，告诉 `DialogueStateTracker`
    如何在事件发生时更新其状态。
    """

    type_name = "event"  # 事件类型名称

    def __init__(
        self,
        timestamp: Optional[float] = None,  # 时间戳
        metadata: Optional[Dict[Text, Any]] = None,  # 元数据
    ) -> None:
        """初始化事件。
        
        Args:
            timestamp: 事件时间戳，默认为当前时间
            metadata: 事件元数据，默认为空字典
        """
        self.timestamp = timestamp or time.time()  # 设置时间戳
        self.metadata = metadata or {}  # 设置元数据

    def __ne__(self, other: Any) -> bool:
        """不等于操作符。
        
        虽然不是严格必要的，但为了避免 x==y 和 x!=y 同时为 True
        """
        return not (self == other)

    @abc.abstractmethod
    def as_story_string(self) -> Optional[Text]:
        """返回事件的故事字符串表示。
        
        此方法用于将事件转换为故事格式的字符串，
        用于训练数据的生成和显示。

        Returns:
            事件的文本表示或 None
        """
        # 每个子类都应该实现此方法
        raise NotImplementedError

    @staticmethod
    def from_story_string(
        event_name: Text,  # 事件名称
        parameters: Dict[Text, Any],  # 事件参数
        default: Optional[Type["Event"]] = None,  # 默认事件类型
    ) -> Optional[List["Event"]]:
        """从故事字符串创建事件。
        
        Args:
            event_name: 事件类型名称
            parameters: 事件参数字典
            default: 默认事件类型
            
        Returns:
            创建的事件列表或 None
        """
        event_class = Event.resolve_by_type(event_name, default)  # 解析事件类型

        if not event_class:  # 如果无法解析事件类型
            return None

        return event_class._from_story_string(parameters)  # 从故事字符串创建事件

    @staticmethod
    def from_parameters(
        parameters: Dict[Text, Any],  # 参数字典
        default: Optional[Type["Event"]] = None  # 默认事件类型
    ) -> Optional["Event"]:
        """从参数字典创建事件。
        
        Args:
            parameters: 包含事件信息的参数字典
            default: 默认事件类型
            
        Returns:
            创建的事件或 None
        """
        event_name = parameters.get("event")  # 获取事件名称
        if event_name is None:  # 如果没有事件名称
            return None

        event_class: Optional[Type[Event]] = Event.resolve_by_type(event_name, default)  # 解析事件类型
        if not event_class:  # 如果无法解析事件类型
            return None

        return event_class._from_parameters(parameters)  # 从参数创建事件

    @classmethod
    def _from_story_string(
        cls: Type[E], parameters: Dict[Text, Any]
    ) -> Optional[List[E]]:
        """将解析的故事行转换为事件。
        
        此方法由子类实现，用于从故事格式的字符串创建事件对象。
        
        Args:
            cls: 事件类
            parameters: 参数字典
            
        Returns:
            事件列表
        """
        return [cls(parameters.get("timestamp"), parameters.get("metadata"))]  # 创建事件实例

    def as_dict(self) -> Dict[Text, Any]:
        """将事件转换为字典格式。
        
        Returns:
            事件的字典表示
        """
        d = {"event": self.type_name, "timestamp": self.timestamp}  # 基本字段

        if self.metadata:  # 如果有元数据
            d["metadata"] = self.metadata  # 添加元数据

        return d  # 返回字典

    def fingerprint(self) -> Text:
        """返回事件的唯一哈希值，在Python运行之间保持稳定。
        
        此方法用于生成事件的唯一标识符，用于去重和比较。
        
        Returns:
            事件的指纹
        """
        data = self.as_dict()  # 获取字典表示
        del data["timestamp"]  # 删除时间戳（时间戳不应该影响指纹）
        return rasa.shared.utils.io.get_dictionary_fingerprint(data)  # 生成指纹

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> Optional["Event"]:
        """将参数字典转换为单个事件。
        
        默认情况下使用与故事行转换相同的实现。但子类可能
        决定以不同的方式处理参数，特别是当解析的参数
        不是来自故事文件时。
        
        Args:
            cls: 事件类
            parameters: 参数字典
            
        Returns:
            创建的事件或 None
        """
        result = cls._from_story_string(parameters)  # 从故事字符串创建事件
        if len(result) > 1:  # 如果创建了多个事件
            logger.warning(
                f"Event from parameters called with parameters "
                f"for multiple events. This is not supported, "
                f"only the first event will be returned. "
                f"Parameters: {parameters}"
            )
        return result[0] if result else None  # 返回第一个事件或 None

    @staticmethod
    def resolve_by_type(
        type_name: Text, default: Optional[Type["Event"]] = None
    ) -> Optional[Type["Event"]]:
        """根据类型名称返回事件类。
        
        Args:
            type_name: 事件类型名称
            default: 默认事件类型
            
        Returns:
            对应的事件类或 None
        """
        for cls in rasa.shared.utils.common.all_subclasses(Event):  # 遍历所有事件子类
            if cls.type_name == type_name:  # 如果类型名称匹配
                return cls  # 返回对应的事件类
        if type_name == "topic":  # 如果是旧的主题事件
            return None  # 向后兼容，支持旧的 TopicSet 事件
        elif default is not None:  # 如果有默认类型
            return default  # 返回默认类型
        else:  # 如果无法找到对应类型
            raise ValueError(f"Unknown event name '{type_name}'.")  # 抛出异常

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。
        
        此方法由子类实现，用于更新跟踪器的状态。

        Args:
            tracker: 当前对话状态跟踪器
        """
        pass  # 默认实现为空

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。
        
        每个子类都应该实现此方法以支持事件比较。
        """
        # 每个子类都应该实现此方法
        raise NotImplementedError()

    def __str__(self) -> Text:
        """返回事件的文本表示。
        
        Returns:
            事件的字符串表示
        """
        return f"{self.__class__.__name__}()"  # 返回类名


# =============================================================================
# 事件混入类 - 提供通用行为
# =============================================================================

class AlwaysEqualEventMixin(Event, ABC):
    """用于没有额外属性的事件的通用行为去重类。
    
    此类提供始终相等的比较行为，适用于不需要复杂比较逻辑的事件。
    """

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。
        
        对于此类事件，只要类型相同就认为相等。
        """
        if not isinstance(other, self.__class__):  # 如果类型不同
            return NotImplemented  # 返回 NotImplemented

        return True  # 类型相同则相等


class SkipEventInMDStoryMixin(Event, ABC):
    """跳过在 Markdown 故事中可视化的事件。
    
    此类用于那些不应该在故事文件中显示的事件，
    如内部状态变化事件。
    """

    def as_story_string(self) -> None:
        """返回事件的故事字符串表示。
        
        此类事件不应该出现在故事中，因此返回 None。
        
        Returns:
            None，因为此事件不应出现在故事内部
        """
        return  # 返回 None


# =============================================================================
# 用户消息事件类
# =============================================================================

class UserUttered(Event):
    """用户对机器人说了什么。
    
    作为副作用，将在 `Tracker` 中创建一个新的 `Turn`。
    这是对话中最重要的事件类型之一，记录了用户的输入。
    """

    type_name = "user"  # 事件类型名称

    def __init__(
        self,
        text: Optional[Text] = None,  # 用户消息文本
        intent: Optional[Dict] = None,  # 意图预测
        entities: Optional[List[Dict]] = None,  # 提取的实体
        parse_data: Optional["NLUPredictionData"] = None,  # 详细的NLU解析结果
        timestamp: Optional[float] = None,  # 时间戳
        input_channel: Optional[Text] = None,  # 输入通道
        message_id: Optional[Text] = None,  # 消息ID
        metadata: Optional[Dict] = None,  # 元数据
        use_text_for_featurization: Optional[bool] = None,  # 是否使用文本进行特征化
    ) -> None:
        """创建传入用户消息的事件。

        Args:
            text: 用户消息文本
            intent: 用户消息的意图预测
            entities: 提取的实体
            parse_data: 消息的详细NLU解析结果
            timestamp: 事件创建时间
            metadata: 附加事件元数据
            input_channel: 用户发送消息的通道
            message_id: 消息的唯一ID
            use_text_for_featurization: 如果使用消息文本预测下一个动作则为 `True`，
                如果使用消息意图则为 `False`
        """
        self.text = text  # 设置文本
        self.intent = intent if intent else {}  # 设置意图，默认为空字典
        self.entities = entities if entities else []  # 设置实体，默认为空列表
        self.input_channel = input_channel  # 设置输入通道
        self.message_id = message_id  # 设置消息ID

        super().__init__(timestamp, metadata)  # 调用父类构造函数

        # 特征化设置由策略在预测时使用 `DefinePrevUserUtteredFeaturization` 事件设置
        self.use_text_for_featurization = use_text_for_featurization
        # 定义此用户话语应如何进行特征化
        if self.text and not self.intent_name:  # 如果有文本但没有意图名称
            # 在训练期间发生
            self.use_text_for_featurization = True
        elif self.intent_name and not self.text:  # 如果有意图名称但没有文本
            # 在训练期间发生
            self.use_text_for_featurization = False

        # 构建解析数据字典
        self.parse_data: "NLUPredictionData" = {
            INTENT: self.intent,  # type: ignore[misc]  # 意图信息
            # 复制实体，以便对 `self.entities` 的更改不会影响
            # `self.parse_data`，因此不会持久化
            ENTITIES: self.entities.copy(),  # 实体列表的副本
            TEXT: self.text,  # 文本
            "message_id": self.message_id,  # 消息ID
            "metadata": self.metadata,  # 元数据
        }
        if parse_data:  # 如果提供了解析数据
            self.parse_data.update(**parse_data)  # 更新解析数据

    @staticmethod
    def _from_parse_data(
        text: Text,  # 文本
        parse_data: "NLUPredictionData",  # 解析数据
        timestamp: Optional[float] = None,  # 时间戳
        input_channel: Optional[Text] = None,  # 输入通道
        message_id: Optional[Text] = None,  # 消息ID
        metadata: Optional[Dict] = None,  # 元数据
    ) -> "UserUttered":
        """从解析数据创建用户话语事件。
        
        Args:
            text: 用户消息文本
            parse_data: NLU解析数据
            timestamp: 时间戳
            input_channel: 输入通道
            message_id: 消息ID
            metadata: 元数据
            
        Returns:
            用户话语事件
        """
        return UserUttered(
            text,  # 文本
            parse_data.get(INTENT),  # 从解析数据获取意图
            parse_data.get(ENTITIES, []),  # 从解析数据获取实体
            parse_data,  # 解析数据
            timestamp,  # 时间戳
            input_channel,  # 输入通道
            message_id,  # 消息ID
            metadata,  # 元数据
        )

    def __hash__(self) -> int:
        """返回对象的唯一哈希值。"""
        return hash(json.dumps(self.as_sub_state()))

    @property
    def intent_name(self) -> Optional[Text]:
        """返回意图名称，如果没有意图则返回 `None`。"""
        return self.intent.get(INTENT_NAME_KEY)

    @property
    def full_retrieval_intent_name(self) -> Optional[Text]:
        """返回完整检索意图名称，如果没有检索意图则返回 `None`。"""
        return self.intent.get(FULL_RETRIEVAL_INTENT_NAME_KEY)

    # Note that this means two UserUttered events with the same text, intent
    # and entities but _different_ timestamps will be considered equal.
    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, UserUttered):
            return NotImplemented

        return (
            self.text,
            self.intent_name,
            [
                jsonpickle.encode(sorted(ent)) for ent in self.entities
            ],  # TODO: test? Or fix in regex_message_handler?
        ) == (
            other.text,
            other.intent_name,
            [jsonpickle.encode(sorted(ent)) for ent in other.entities],
        )

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        entities = ""
        if self.entities:
            entities_list = [
                f"{entity[ENTITY_ATTRIBUTE_VALUE]} "
                f"(Type: {entity[ENTITY_ATTRIBUTE_TYPE]}, "
                f"Role: {entity.get(ENTITY_ATTRIBUTE_ROLE)}, "
                f"Group: {entity.get(ENTITY_ATTRIBUTE_GROUP)})"
                for entity in self.entities
            ]
            entities = f", entities: {', '.join(entities_list)}"

        return (
            f"UserUttered(text: {self.text}, intent: {self.intent_name}"
            f"{entities}"
            f", use_text_for_featurization: {self.use_text_for_featurization})"
        )

    @staticmethod
    def empty() -> "UserUttered":
        return UserUttered(None)

    def is_empty(self) -> bool:
        return not self.text and not self.intent_name and not self.entities

    def as_dict(self) -> Dict[Text, Any]:
        _dict = super().as_dict()
        _dict.update(
            {
                "text": self.text,
                "parse_data": self.parse_data,
                "input_channel": getattr(self, "input_channel", None),
                "message_id": getattr(self, "message_id", None),
                "metadata": self.metadata,
            }
        )
        return _dict

    def as_sub_state(self) -> Dict[Text, Union[None, Text, List[Optional[Text]]]]:
        """Turns a UserUttered event into features.

        The substate contains information about entities, intent and text of the
        `UserUttered` event.

        Returns:
            a dictionary with intent name, text and entities
        """
        entities = [entity.get(ENTITY_ATTRIBUTE_TYPE) for entity in self.entities]
        entities.extend(
            (
                f"{entity.get(ENTITY_ATTRIBUTE_TYPE)}{ENTITY_LABEL_SEPARATOR}"
                f"{entity.get(ENTITY_ATTRIBUTE_ROLE)}"
            )
            for entity in self.entities
            if ENTITY_ATTRIBUTE_ROLE in entity
        )
        entities.extend(
            (
                f"{entity.get(ENTITY_ATTRIBUTE_TYPE)}{ENTITY_LABEL_SEPARATOR}"
                f"{entity.get(ENTITY_ATTRIBUTE_GROUP)}"
            )
            for entity in self.entities
            if ENTITY_ATTRIBUTE_GROUP in entity
        )

        out: Dict[Text, Union[None, Text, List[Optional[Text]]]] = {}
        # During training we expect either intent_name or text to be set.
        # During prediction both will be set.
        if self.text and (
            self.use_text_for_featurization or self.use_text_for_featurization is None
        ):
            out[TEXT] = self.text
        if self.intent_name and not self.use_text_for_featurization:
            out[INTENT] = self.intent_name
        # don't add entities for e2e utterances
        if entities and not self.use_text_for_featurization:
            out[ENTITIES] = entities

        return out

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["UserUttered"]]:
        try:
            return [
                cls._from_parse_data(
                    parameters.get("text"),
                    parameters.get("parse_data"),
                    parameters.get("timestamp"),
                    parameters.get("input_channel"),
                    parameters.get("message_id"),
                    parameters.get("metadata"),
                )
            ]
        except KeyError as e:
            raise ValueError(f"Failed to parse bot uttered event. {e}")

    def _entity_string(self) -> Text:
        if self.entities:
            return json.dumps(
                {
                    entity[ENTITY_ATTRIBUTE_TYPE]: entity[ENTITY_ATTRIBUTE_VALUE]
                    for entity in self.entities
                },
                ensure_ascii=False,
            )
        return ""

    def as_story_string(self, e2e: bool = False) -> Text:
        """Return event as string for Markdown training format.

        Args:
            e2e: `True` if the the event should be printed in the format for
                end-to-end conversation tests.

        Returns:
            Event as string.
        """
        if self.use_text_for_featurization and not e2e:
            raise UnsupportedFeatureException(
                f"Printing end-to-end user utterances is not supported in the "
                f"Markdown training format. Please use the YAML training data format "
                f"instead. Please see {DOCS_URL_TRAINING_DATA} for more information."
            )

        if e2e:
            text_with_entities = format_message(
                self.text or "", self.intent_name, self.entities
            )

            intent_prefix = f"{self.intent_name}: " if self.intent_name else ""
            return f"{intent_prefix}{text_with_entities}"

        return f"{self.intent_name or ''}{self._entity_string()}"

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to tracker. See docstring of `Event`."""
        tracker.latest_message = self
        tracker.clear_followup_action()

    @staticmethod
    def create_external(
        intent_name: Text,
        entity_list: Optional[List[Dict[Text, Any]]] = None,
        input_channel: Optional[Text] = None,
    ) -> "UserUttered":
        return UserUttered(
            text=f"{EXTERNAL_MESSAGE_PREFIX}{intent_name}",
            intent={INTENT_NAME_KEY: intent_name},
            metadata={IS_EXTERNAL: True},
            entities=entity_list or [],
            input_channel=input_channel,
        )


class DefinePrevUserUtteredFeaturization(SkipEventInMDStoryMixin):
    """Stores information whether action was predicted based on text or intent."""

    type_name = "user_featurization"

    def __init__(
        self,
        use_text_for_featurization: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event.

        Args:
            use_text_for_featurization: `True` if message text was used to predict
                action. `False` if intent was used.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        super().__init__(timestamp, metadata)
        self.use_text_for_featurization = use_text_for_featurization

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return f"DefinePrevUserUtteredFeaturization({self.use_text_for_featurization})"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.use_text_for_featurization)

    @classmethod
    def _from_parameters(
        cls, parameters: Dict[Text, Any]
    ) -> "DefinePrevUserUtteredFeaturization":
        return DefinePrevUserUtteredFeaturization(
            parameters.get(USE_TEXT_FOR_FEATURIZATION),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({USE_TEXT_FOR_FEATURIZATION: self.use_text_for_featurization})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state.

        Args:
            tracker: The current conversation state.
        """
        if tracker.latest_action_name != ACTION_LISTEN_NAME:
            # featurization belong only to the last user message
            # a user message is always followed by action listen
            return

        if not tracker.latest_message:
            return

        # update previous user message's featurization based on this event
        tracker.latest_message.use_text_for_featurization = (
            self.use_text_for_featurization
        )

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, DefinePrevUserUtteredFeaturization):
            return NotImplemented

        return self.use_text_for_featurization == other.use_text_for_featurization


class EntitiesAdded(SkipEventInMDStoryMixin):
    """Event that is used to add extracted entities to the tracker state."""

    type_name = "entities"

    def __init__(
        self,
        entities: List[Dict[Text, Any]],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Initializes event.

        Args:
            entities: Entities extracted from previous user message. This can either
                be done by NLU components or end-to-end policy predictions.
            timestamp: the timestamp
            metadata: some optional metadata
        """
        super().__init__(timestamp, metadata)
        self.entities = entities

    def __str__(self) -> Text:
        """Returns the string representation of the event."""
        entity_str = [e[ENTITY_ATTRIBUTE_TYPE] for e in self.entities]
        return f"{self.__class__.__name__}({entity_str})"

    def __hash__(self) -> int:
        """Returns the hash value of the event."""
        return hash(json.dumps(self.entities))

    def __eq__(self, other: Any) -> bool:
        """Compares this event with another event."""
        if not isinstance(other, EntitiesAdded):
            return NotImplemented

        return self.entities == other.entities

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "EntitiesAdded":
        return EntitiesAdded(
            parameters.get(ENTITIES),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Converts the event into a dict.

        Returns:
            A dict that represents this event.
        """
        d = super().as_dict()
        d.update({ENTITIES: self.entities})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state.

        Args:
            tracker: The current conversation state.
        """
        if tracker.latest_action_name != ACTION_LISTEN_NAME:
            # entities belong only to the last user message
            # a user message always comes after action listen
            return

        if not tracker.latest_message:
            return

        for entity in self.entities:
            if entity not in tracker.latest_message.entities:
                tracker.latest_message.entities.append(entity)


class BotUttered(SkipEventInMDStoryMixin):
    """机器人对用户说了什么。

    This class is not used in the story training as it is contained in the

    ``ActionExecuted`` class. An entry is made in the ``Tracker``.
    """

    type_name = "bot"

    def __init__(
        self,
        text: Optional[Text] = None,
        data: Optional[Dict] = None,
        metadata: Optional[Dict[Text, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Creates event for a bot response.

        Args:
            text: Plain text which bot responded with.
            data: Additional data for more complex utterances (e.g. buttons).
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.text = text
        self.data = data or {}
        super().__init__(timestamp, metadata)

    def __members(self) -> Tuple[Optional[Text], Text, Text]:
        data_no_nones = {k: v for k, v in self.data.items() if v is not None}
        meta_no_nones = {k: v for k, v in self.metadata.items() if v is not None}
        return (
            self.text,
            jsonpickle.encode(data_no_nones),
            jsonpickle.encode(meta_no_nones),
        )

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.__members())

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, BotUttered):
            return NotImplemented

        return self.__members() == other.__members()

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return "BotUttered(text: {}, data: {}, metadata: {})".format(
            self.text, json.dumps(self.data), json.dumps(self.metadata)
        )

    def __repr__(self) -> Text:
        """返回事件的调试文本表示。"""
        return "BotUttered('{}', {}, {}, {})".format(
            self.text, json.dumps(self.data), json.dumps(self.metadata), self.timestamp
        )

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.latest_bot_utterance = self

    def message(self) -> Dict[Text, Any]:
        """Return the complete message as a dictionary."""
        m = self.data.copy()
        m["text"] = self.text
        m["timestamp"] = self.timestamp
        m.update(self.metadata)

        if m.get("image") == m.get("attachment"):
            # we need this as there is an oddity we introduced a while ago where
            # we automatically set the attachment to the image. to not break
            # any persisted events we kept that, but we need to make sure that
            # the message contains the image only once
            m["attachment"] = None

        return m

    @staticmethod
    def empty() -> "BotUttered":
        """Creates an empty bot utterance."""
        return BotUttered()

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({"text": self.text, "data": self.data, "metadata": self.metadata})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "BotUttered":
        try:
            return BotUttered(
                parameters.get("text"),
                parameters.get("data"),
                parameters.get("metadata"),
                parameters.get("timestamp"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse bot uttered event. {e}")


class SlotSet(Event):
    """用户已指定其对槽位值的偏好。

    Every slot has a name and a value. This event can be used to set a
    value for a slot on a conversation.

    As a side effect the `Tracker`'s slots will be updated so
    that `tracker.slots[key]=value`.
    """

    type_name = "slot"

    def __init__(
        self,
        key: Text,
        value: Optional[Any] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event to set slot.

        Args:
            key: Name of the slot which is set.
            value: Value to which slot is set.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.key = key
        self.value = value
        super().__init__(timestamp, metadata)

    def __repr__(self) -> Text:
        """Returns text representation of event."""
        return f"SlotSet(key: {self.key}, value: {self.value})"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash((self.key, jsonpickle.encode(self.value)))

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, SlotSet):
            return NotImplemented

        return (self.key, self.value) == (other.key, other.value)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        props = json.dumps({self.key: self.value}, ensure_ascii=False)
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["SlotSet"]]:

        slots = []
        for slot_key, slot_val in parameters.items():
            slots.append(SlotSet(slot_key, slot_val))

        if slots:
            return slots
        else:
            return None

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({"name": self.key, "value": self.value})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "SlotSet":
        try:
            return SlotSet(
                parameters.get("name"),
                parameters.get("value"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse set slot event. {e}")

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._set_slot(self.key, self.value)


class Restarted(AlwaysEqualEventMixin):
    """对话应该重新开始并清除历史记录。

    Instead of deleting all events, this event can be used to reset the
    trackers state (e.g. ignoring any past user messages & resetting all
    the slots).
    """

    type_name = "restart"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124312)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Resets the tracker and triggers a followup `ActionSessionStart`."""
        tracker._reset()
        tracker.trigger_followup_action(ACTION_SESSION_START_NAME)


class UserUtteranceReverted(AlwaysEqualEventMixin):
    """机器人撤销最近用户消息之前的所有内容。

    The bot will revert all events after the latest `UserUttered`, this
    also means that the last event on the tracker is usually `action_listen`
    and the bot is waiting for a new user message.
    """

    type_name = "rewind"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124315)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._reset()
        tracker.replay_events()


class AllSlotsReset(AlwaysEqualEventMixin):
    """所有槽位都重置为其初始值。

    If you want to keep the dialogue history and only want to reset the
    slots, you can use this event to set all the slots to their initial
    values.
    """

    type_name = "reset_slots"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124316)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._reset_slots()


class ReminderScheduled(Event):
    """在给定时间安排用户意图的异步触发。

    The triggered intent can include entities if needed.
    """

    type_name = "reminder"

    def __init__(
        self,
        intent: Text,
        trigger_date_time: datetime,
        entities: Optional[List[Dict]] = None,
        name: Optional[Text] = None,
        kill_on_user_message: bool = True,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates the reminder.

        Args:
            intent: Name of the intent to be triggered.
            trigger_date_time: Date at which the execution of the action
                should be triggered (either utc or with tz).
            name: ID of the reminder. If there are multiple reminders with
                 the same id only the last will be run.
            entities: Entities that should be supplied together with the
                 triggered intent.
            kill_on_user_message: ``True`` means a user message before the
                 trigger date will abort the reminder.
            timestamp: Creation date of the event.
            metadata: Optional event metadata.
        """
        self.intent = intent
        self.entities = entities
        self.trigger_date_time = trigger_date_time
        self.kill_on_user_message = kill_on_user_message
        self.name = name if name is not None else str(uuid.uuid1())
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(
            (
                self.intent,
                self.entities,
                self.trigger_date_time.isoformat(),
                self.kill_on_user_message,
                self.name,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, ReminderScheduled):
            return NotImplemented

        return self.name == other.name

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return (
            f"ReminderScheduled(intent: {self.intent}, "
            f"trigger_date: {self.trigger_date_time}, "
            f"entities: {self.entities}, name: {self.name})"
        )

    def scheduled_job_name(self, sender_id: Text) -> Text:
        return (
            f"[{hash(self.name)},{hash(self.intent)},{hash(str(self.entities))}]"
            f"{ACTION_NAME_SENDER_ID_CONNECTOR_STR}"
            f"{sender_id}"
        )

    def _properties(self) -> Dict[Text, Any]:
        return {
            "intent": self.intent,
            "date_time": self.trigger_date_time.isoformat(),
            "entities": self.entities,
            "name": self.name,
            "kill_on_user_msg": self.kill_on_user_message,
        }

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        props = json.dumps(self._properties())
        return f"{self.type_name}{props}"

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update(self._properties())
        return d

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ReminderScheduled"]]:

        trigger_date_time = parser.parse(parameters.get("date_time"))

        return [
            ReminderScheduled(
                parameters.get("intent"),
                trigger_date_time,
                parameters.get("entities"),
                name=parameters.get("name"),
                kill_on_user_message=parameters.get("kill_on_user_msg", True),
                timestamp=parameters.get("timestamp"),
                metadata=parameters.get("metadata"),
            )
        ]


class ReminderCancelled(Event):
    """取消某些任务。"""

    type_name = "cancel_reminder"

    def __init__(
        self,
        name: Optional[Text] = None,
        intent: Optional[Text] = None,
        entities: Optional[List[Dict]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates a ReminderCancelled event.

        If all arguments are `None`, this will cancel all reminders.
        are to be cancelled. If no arguments are supplied, this will cancel all
        reminders.

        Args:
            name: Name of the reminder to be cancelled.
            intent: Intent name that is to be used to identify the reminders to be
                cancelled.
            entities: Entities that are to be used to identify the reminders to be
                cancelled.
            timestamp: Optional timestamp.
            metadata: Optional event metadata.
        """
        self.name = name
        self.intent = intent
        self.entities = entities
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash((self.name, self.intent, str(self.entities)))

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, ReminderCancelled):
            return NotImplemented

        return hash(self) == hash(other)

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return (
            f"ReminderCancelled(name: {self.name}, intent: {self.intent}, "
            f"entities: {self.entities})"
        )

    def cancels_job_with_name(self, job_name: Text, sender_id: Text) -> bool:
        """Determines if this event should cancel the job with the given name.

        Args:
            job_name: Name of the job to be tested.
            sender_id: The `sender_id` of the tracker.

        Returns:
            `True`, if this `ReminderCancelled` event should cancel the job with the
            given name, and `False` otherwise.
        """
        match = re.match(
            rf"^\[([\d\-]*),([\d\-]*),([\d\-]*)\]"
            rf"({re.escape(ACTION_NAME_SENDER_ID_CONNECTOR_STR)}"
            rf"{re.escape(sender_id)})",
            job_name,
        )
        if not match:
            return False
        name_hash, intent_hash, entities_hash = match.group(1, 2, 3)

        # Cancel everything unless names/intents/entities are given to
        # narrow it down.
        return (
            ((not self.name) or self._matches_name_hash(name_hash))
            and ((not self.intent) or self._matches_intent_hash(intent_hash))
            and ((not self.entities) or self._matches_entities_hash(entities_hash))
        )

    def _matches_name_hash(self, name_hash: Text) -> bool:
        return str(hash(self.name)) == name_hash

    def _matches_intent_hash(self, intent_hash: Text) -> bool:
        return str(hash(self.intent)) == intent_hash

    def _matches_entities_hash(self, entities_hash: Text) -> bool:
        return str(hash(str(self.entities))) == entities_hash

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        props = json.dumps(
            {"name": self.name, "intent": self.intent, "entities": self.entities}
        )
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ReminderCancelled"]]:
        return [
            ReminderCancelled(
                parameters.get("name"),
                parameters.get("intent"),
                parameters.get("entities"),
                timestamp=parameters.get("timestamp"),
                metadata=parameters.get("metadata"),
            )
        ]


class ActionReverted(AlwaysEqualEventMixin):
    """机器人撤销其最后一个动作。

    The bot reverts everything until before the most recent action.
    This includes the action itself, as well as any events that
    action created, like set slot events - the bot will now
    predict a new action using the state before the most recent
    action.
    """

    type_name = "undo"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124318)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._reset()
        tracker.replay_events()


class StoryExported(Event):
    """故事应该转储到文件。"""

    type_name = "export"

    def __init__(
        self,
        path: Optional[Text] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event about story exporting.

        Args:
            path: Path to which story was exported to.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.path = path
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124319)

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["StoryExported"]]:
        return [
            StoryExported(
                parameters.get("path"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        if self.path:
            tracker.export_stories_to_file(self.path)

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, StoryExported):
            return NotImplemented

        return self.path == other.path


class FollowupAction(Event):
    """将后续动作加入队列。"""

    type_name = "followup"

    def __init__(
        self,
        name: Text,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates an event which forces the model to run a certain action next.

        Args:
            name: Name of the action to run.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.action_name = name
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.action_name)

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, FollowupAction):
            return NotImplemented

        return self.action_name == other.action_name

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return f"FollowupAction(action: {self.action_name})"

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        props = json.dumps({"name": self.action_name})
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["FollowupAction"]]:

        return [
            FollowupAction(
                parameters.get("name"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({"name": self.action_name})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.trigger_followup_action(self.action_name)


class ConversationPaused(AlwaysEqualEventMixin):
    """忽略来自用户的消息，让人类接管。

    As a side effect the `Tracker`'s `paused` attribute will
    be set to `True`.
    """

    type_name = "pause"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124313)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return str(self)

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._paused = True


class ConversationResumed(AlwaysEqualEventMixin):
    """机器人接管对话。

    Inverse of `PauseConversation`. As a side effect the `Tracker`'s
    `paused` attribute will be set to `False`.
    """

    type_name = "resume"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124314)

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker._paused = False


class ActionExecuted(Event):
    """操作描述已执行的动作及其结果。

    It comprises an action and a list of events. operations will be appended
    to the latest `Turn`` in `Tracker.turns`.
    """

    type_name = "action"

    def __init__(
        self,
        action_name: Optional[Text] = None,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
        action_text: Optional[Text] = None,
        hide_rule_turn: bool = False,
    ) -> None:
        """Creates event for a successful event execution.

        Args:
            action_name: Name of the action which was executed. `None` if it was an
                end-to-end prediction.
            policy: Policy which predicted action.
            confidence: Confidence with which policy predicted action.
            timestamp: When the event was created.
            metadata: Additional event metadata.
            action_text: In case it's an end-to-end action prediction, the text which
                was predicted.
            hide_rule_turn: If `True`, this action should be hidden in the dialogue
                history created for ML-based policies.
        """
        self.action_name = action_name
        self.policy = policy
        self.confidence = confidence
        self.unpredictable = False
        self.action_text = action_text
        self.hide_rule_turn = hide_rule_turn

        if self.action_name is None and self.action_text is None:
            raise ValueError(
                "Both the name of the action and the end-to-end "
                "predicted text are missing. "
                "The `ActionExecuted` event cannot be initialised."
            )

        super().__init__(timestamp, metadata)

    def __members__(self) -> Tuple[Optional[Text], Optional[Text], Text]:
        meta_no_nones = {k: v for k, v in self.metadata.items() if v is not None}
        return (self.action_name, self.action_text, jsonpickle.encode(meta_no_nones))

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return "ActionExecuted(action: {}, policy: {}, confidence: {})".format(
            self.action_name, self.policy, self.confidence
        )

    def __str__(self) -> Text:
        """Returns event as human readable string."""
        return str(self.action_name) or str(self.action_text)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.__members__())

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, ActionExecuted):
            return NotImplemented

        return self.__members__() == other.__members__()

    def as_story_string(self) -> Optional[Text]:
        """Returns event in Markdown format."""
        if self.action_text:
            raise UnsupportedFeatureException(
                f"Printing end-to-end bot utterances is not supported in the "
                f"Markdown training format. Please use the YAML training data format "
                f"instead. Please see {DOCS_URL_TRAINING_DATA} for more information."
            )

        return self.action_name

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ActionExecuted"]]:
        return [
            ActionExecuted(
                parameters.get("name"),
                parameters.get("policy"),
                parameters.get("confidence"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
                parameters.get("action_text"),
                parameters.get("hide_rule_turn", False),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update(
            {
                "name": self.action_name,
                "policy": self.policy,
                "confidence": self.confidence,
                "action_text": self.action_text,
                "hide_rule_turn": self.hide_rule_turn,
            }
        )
        return d

    def as_sub_state(self) -> Dict[Text, Text]:
        """Turns ActionExecuted into a dictionary containing action name or action text.

        One action cannot have both set at the same time

        Returns:
            a dictionary containing action name or action text with the corresponding
            key.
        """
        if self.action_name:
            return {ACTION_NAME: self.action_name}
        else:
            # FIXME: we should define the type better here, and require either
            #        `action_name` or `action_text`
            return {ACTION_TEXT: cast(Text, self.action_text)}

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.set_latest_action(self.as_sub_state())
        tracker.clear_followup_action()


class AgentUttered(SkipEventInMDStoryMixin):
    """代理对用户说了什么。

    This class is not used in the story training as it is contained in the
    ``ActionExecuted`` class. An entry is made in the ``Tracker``.
    """

    type_name = "agent"

    def __init__(
        self,
        text: Optional[Text] = None,
        data: Optional[Any] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """See docstring of `BotUttered`."""
        self.text = text
        self.data = data
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash((self.text, jsonpickle.encode(self.data)))

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, AgentUttered):
            return NotImplemented

        return (self.text, jsonpickle.encode(self.data)) == (
            other.text,
            jsonpickle.encode(other.data),
        )

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return "AgentUttered(text: {}, data: {})".format(
            self.text, json.dumps(self.data)
        )

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({"text": self.text, "data": self.data})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "AgentUttered":
        try:
            return AgentUttered(
                parameters.get("text"),
                parameters.get("data"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse agent uttered event. {e}")


class ActiveLoop(Event):
    """如果给出 `name`：激活名为 `name` 的循环，否则停用活动循环。"""

    type_name = "active_loop"

    def __init__(
        self,
        name: Optional[Text],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event for active loop.

        Args:
            name: Name of activated loop or `None` if current loop is deactivated.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.name = name
        super().__init__(timestamp, metadata)

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return f"Loop({self.name})"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, ActiveLoop):
            return NotImplemented

        return self.name == other.name

    def as_story_string(self) -> Text:
        """返回事件的故事字符串表示。"""
        props = json.dumps({LOOP_NAME: self.name})
        return f"{ActiveLoop.type_name}{props}"

    @classmethod
    def _from_story_string(cls, parameters: Dict[Text, Any]) -> List["ActiveLoop"]:
        """Called to convert a parsed story line into an event."""
        return [
            ActiveLoop(
                parameters.get(LOOP_NAME),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({LOOP_NAME: self.name})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.change_loop_to(self.name)


class LegacyForm(ActiveLoop):
    """Legacy handler of old `Form` events.

    The `ActiveLoop` event used to be called `Form`. This class is there to handle old
    legacy events which were stored with the old type name `form`.
    """

    type_name = "form"

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        # Dump old `Form` events as `ActiveLoop` events instead of keeping the old
        # event type.
        d["event"] = ActiveLoop.type_name
        return d

    def fingerprint(self) -> Text:
        """Returns the hash of the event."""
        d = self.as_dict()
        # Revert event name to legacy subclass name to avoid different event types
        # having the same fingerprint.
        d["event"] = self.type_name
        del d["timestamp"]
        return rasa.shared.utils.io.get_dictionary_fingerprint(d)


class LoopInterrupted(SkipEventInMDStoryMixin):
    """由 FormPolicy 和 RulePolicy 添加的事件。

    Notifies form action whether or not to validate the user input.
    """

    type_name = "loop_interrupted"

    def __init__(
        self,
        is_interrupted: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Event to notify that loop was interrupted.

        This e.g. happens when a user is within a form, and is de-railing the
        form-filling by asking FAQs.

        Args:
            is_interrupted: `True` if the loop execution was interrupted, and ML
                policies had to take over the last prediction.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        super().__init__(timestamp, metadata)
        self.is_interrupted = is_interrupted

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return f"{LoopInterrupted.__name__}({self.is_interrupted})"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.is_interrupted)

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, LoopInterrupted):
            return NotImplemented

        return self.is_interrupted == other.is_interrupted

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "LoopInterrupted":
        return LoopInterrupted(
            parameters.get(LOOP_INTERRUPTED, False),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update({LOOP_INTERRUPTED: self.is_interrupted})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.interrupt_loop(self.is_interrupted)


class LegacyFormValidation(LoopInterrupted):
    """Legacy handler of old `FormValidation` events.

    The `LoopInterrupted` event used to be called `FormValidation`. This class is there
    to handle old legacy events which were stored with the old type name
    `form_validation`.
    """

    type_name = "form_validation"

    def __init__(
        self,
        validate: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """See parent class docstring."""
        # `validate = True` is the same as `interrupted = False`
        super().__init__(not validate, timestamp, metadata)

    @classmethod
    def _from_parameters(cls, parameters: Dict) -> "LoopInterrupted":
        return LoopInterrupted(
            # `validate = True` means `is_interrupted = False`
            not parameters.get("validate", True),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        # Dump old `Form` events as `ActiveLoop` events instead of keeping the old
        # event type.
        d["event"] = LoopInterrupted.type_name
        return d

    def fingerprint(self) -> Text:
        """Returns hash of the event."""
        d = self.as_dict()
        # Revert event name to legacy subclass name to avoid different event types
        # having the same fingerprint.
        d["event"] = self.type_name
        del d["timestamp"]
        return rasa.shared.utils.io.get_dictionary_fingerprint(d)


class ActionExecutionRejected(SkipEventInMDStoryMixin):
    """通知 Core 动作执行已被拒绝。"""

    type_name = "action_execution_rejected"

    def __init__(
        self,
        action_name: Text,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event.

        Args:
            action_name: Action which was rejected.
            policy: Policy which predicted the rejected action.
            confidence: Confidence with which the reject action was predicted.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.action_name = action_name
        self.policy = policy
        self.confidence = confidence
        super().__init__(timestamp, metadata)

    def __str__(self) -> Text:
        """返回事件的文本表示。"""
        return (
            "ActionExecutionRejected("
            "action: {}, policy: {}, confidence: {})"
            "".format(self.action_name, self.policy, self.confidence)
        )

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(self.action_name)

    def __eq__(self, other: Any) -> bool:
        """比较对象与另一个对象。"""
        if not isinstance(other, ActionExecutionRejected):
            return NotImplemented

        return self.action_name == other.action_name

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "ActionExecutionRejected":
        return ActionExecutionRejected(
            parameters.get("name"),
            parameters.get("policy"),
            parameters.get("confidence"),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """返回序列化的事件。"""
        d = super().as_dict()
        d.update(
            {
                "name": self.action_name,
                "policy": self.policy,
                "confidence": self.confidence,
            }
        )
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        tracker.reject_action(self.action_name)


class SessionStarted(AlwaysEqualEventMixin):
    """标记新对话会话的开始。"""

    type_name = "session_started"

    def __hash__(self) -> int:
        """返回事件的唯一哈希值。"""
        return hash(32143124320)

    def as_story_string(self) -> None:
        """Skips representing event in stories."""
        logger.warning(
            f"'{self.type_name}' events cannot be serialised as story strings."
        )

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """将事件应用到当前对话状态。"""
        # noinspection PyProtectedMember
        tracker._reset()
