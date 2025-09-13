# 导入深拷贝模块，用于创建对象的深拷贝
import copy
# 导入JSON处理模块
import json
# 导入日志模块，用于记录日志信息
import logging
# 导入类型注解相关的模块
from typing import (
    List,        # 列表类型
    Text,        # 文本类型
    Optional,    # 可选类型
    Dict,        # 字典类型
    Any,         # 任意类型
    TYPE_CHECKING,  # 类型检查标志
    Tuple,       # 元组类型
    Set,         # 集合类型
    cast,        # 类型转换函数
)

# 导入异步HTTP客户端
import aiohttp
# 导入Rasa核心模块
import rasa.core
# 导入动作相关常量
from rasa.core.actions.constants import DEFAULT_SELECTIVE_DOMAIN, SELECTIVE_DOMAIN
# 导入核心常量
from rasa.core.constants import (
    DEFAULT_REQUEST_TIMEOUT,                    # 默认请求超时时间
    COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,    # 压缩动作服务器请求环境变量名
    DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,     # 默认压缩动作服务器请求
)
# 导入策略预测类
from rasa.core.policies.policy import PolicyPrediction
# 导入NLU常量
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,           # 响应选择器默认意图
    RESPONSE_SELECTOR_PROPERTY_NAME,            # 响应选择器属性名
    RESPONSE_SELECTOR_PREDICTION_KEY,           # 响应选择器预测键
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,         # 响应选择器发声动作键
)
# 导入插件管理器
from rasa.plugin import plugin_manager
# 导入共享常量
from rasa.shared.constants import (
    DOCS_BASE_URL,                              # 文档基础URL
    DEFAULT_NLU_FALLBACK_INTENT_NAME,           # 默认NLU回退意图名称
    UTTER_PREFIX,                               # 发声前缀
)
# 导入核心事件模块
from rasa.shared.core import events
# 导入核心常量
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,                   # 用户意图超出范围
    ACTION_LISTEN_NAME,                         # 监听动作名称
    ACTION_RESTART_NAME,                        # 重启动作名称
    ACTION_SESSION_START_NAME,                  # 会话开始动作名称
    ACTION_DEFAULT_FALLBACK_NAME,               # 默认回退动作名称
    ACTION_DEACTIVATE_LOOP_NAME,                # 停用循环动作名称
    ACTION_REVERT_FALLBACK_EVENTS_NAME,         # 回退事件恢复动作名称
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,        # 默认询问确认动作名称
    ACTION_DEFAULT_ASK_REPHRASE_NAME,           # 默认询问重述动作名称
    ACTION_UNLIKELY_INTENT_NAME,                # 不太可能的意图动作名称
    ACTION_BACK_NAME,                           # 返回动作名称
    REQUESTED_SLOT,                             # 请求的槽位
    ACTION_EXTRACT_SLOTS,                       # 提取槽位动作
    DEFAULT_SLOT_NAMES,                         # 默认槽位名称
    MAPPING_CONDITIONS,                         # 映射条件
    ACTIVE_LOOP,                                # 活跃循环
    ACTION_VALIDATE_SLOT_MAPPINGS,              # 验证槽位映射动作
    MAPPING_TYPE,                               # 映射类型
    SlotMappingType,                            # 槽位映射类型
)
# 导入领域类
from rasa.shared.core.domain import Domain
# 导入核心事件类
from rasa.shared.core.events import (
    UserUtteranceReverted,                      # 用户发声恢复事件
    UserUttered,                                # 用户发声事件
    ActionExecuted,                             # 动作执行事件
    Event,                                      # 事件基类
    BotUttered,                                 # 机器人发声事件
    SlotSet,                                    # 槽位设置事件
    ActiveLoop,                                 # 活跃循环事件
    Restarted,                                  # 重启事件
    SessionStarted,                             # 会话开始事件
)
# 导入槽位映射类
from rasa.shared.core.slot_mappings import SlotMapping
# 导入槽位类
from rasa.shared.core.slots import ListSlot
# 导入对话状态跟踪器
from rasa.shared.core.trackers import DialogueStateTracker
# 导入异常类
from rasa.shared.exceptions import RasaException
# 导入NLU常量
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,                            # 意图名称键
    INTENT_RANKING_KEY,                         # 意图排名键
    ENTITY_ATTRIBUTE_TYPE,                      # 实体属性类型
    ENTITY_ATTRIBUTE_ROLE,                      # 实体属性角色
    ENTITY_ATTRIBUTE_GROUP,                     # 实体属性组
)
# 导入事件模式
from rasa.shared.utils.schemas.events import EVENTS_SCHEMA
# 导入工具模块
import rasa.shared.utils.io
# 导入通用工具函数
from rasa.utils.common import get_bool_env_variable
# 导入端点配置和客户端响应错误
from rasa.utils.endpoints import EndpointConfig, ClientResponseError

# 类型检查时的导入
if TYPE_CHECKING:
    from rasa.core.nlg import NaturalLanguageGenerator  # 自然语言生成器
    from rasa.core.channels.channel import OutputChannel  # 输出通道
    from rasa.shared.core.events import IntentPrediction  # 意图预测

# 创建日志记录器
logger = logging.getLogger(__name__)


def default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]:
    """列出默认动作。
    
    Args:
        action_endpoint: 可选的端点配置，用于运行自定义动作
        
    Returns:
        默认动作列表
    """
    from rasa.core.actions.two_stage_fallback import TwoStageFallbackAction

    return [
        ActionListen(),                                    # 监听动作
        ActionRestart(),                                   # 重启动作
        ActionSessionStart(),                              # 会话开始动作
        ActionDefaultFallback(),                           # 默认回退动作
        ActionDeactivateLoop(),                            # 停用循环动作
        ActionRevertFallbackEvents(),                      # 回退事件恢复动作
        ActionDefaultAskAffirmation(),                     # 默认询问确认动作
        ActionDefaultAskRephrase(),                        # 默认询问重述动作
        TwoStageFallbackAction(action_endpoint),           # 两阶段回退动作
        ActionUnlikelyIntent(),                            # 不太可能的意图动作
        ActionBack(),                                      # 返回动作
        ActionExtractSlots(action_endpoint),               # 提取槽位动作
    ]


def action_for_index(
    index: int, domain: Domain, action_endpoint: Optional[EndpointConfig]
) -> "Action":
    """根据动作在可用动作列表中的索引获取动作。

    Args:
        index: 动作的索引。这通常被`Policy`使用，因为它们预测动作索引而不是名称。
        domain: 当前模型的`Domain`。领域包含用户提供的动作+默认动作。
        action_endpoint: 可用于运行`custom_actions`（例如使用`rasa-sdk`）。

    Returns:
        实例化的`Action`，如果在给定索引处没有找到`Action`则返回`None`。
    """
    # 检查索引是否在有效范围内
    if domain.num_actions <= index or index < 0:
        raise IndexError(
            f"Cannot access action at index {index}. "
            f"Domain has {domain.num_actions} actions."
        )

    # 根据动作名称或文本获取动作
    return action_for_name_or_text(
        domain.action_names_or_texts[index], domain, action_endpoint
    )


def is_retrieval_action(action_name: Text, retrieval_intents: List[Text]) -> bool:
    """检查动作名称是否为检索动作。

    检索动作的名称在相应的检索意图名称前添加了额外的`utter_`前缀。

    Args:
        action_name: 动作的名称。
        retrieval_intents: 在NLU训练数据中定义的检索意图列表。

    Returns:
        如果解析的意图名称存在于检索意图列表中则返回`True`，否则返回`False`。
    """
    return (
        ActionRetrieveResponse.intent_name_from_action(action_name) in retrieval_intents
    )


def action_for_name_or_text(
    action_name_or_text: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]
) -> "Action":
    """根据动作名称或文本检索动作（如果是端到端动作）。

    Args:
        action_name_or_text: 动作的名称。
        domain: 当前模型领域。
        action_endpoint: 执行自定义动作的端点。

    Raises:
        ActionNotFoundException: 如果动作不在当前领域中。

    Returns:
        实例化的动作。
    """
    # 检查动作是否在领域中
    if action_name_or_text not in domain.action_names_or_texts:
        domain.raise_action_not_found_exception(action_name_or_text)

    # 获取默认动作字典
    defaults = {a.name(): a for a in default_actions(action_endpoint)}

    # 如果是默认动作且不在用户动作和表单中，返回默认动作
    if (
        action_name_or_text in defaults
        and action_name_or_text not in domain.user_actions_and_forms
    ):
        return defaults[action_name_or_text]

    # 如果是检索动作，返回检索响应动作
    if action_name_or_text.startswith(UTTER_PREFIX) and is_retrieval_action(
        action_name_or_text, domain.retrieval_intents
    ):
        return ActionRetrieveResponse(action_name_or_text)

    # 如果是端到端动作，返回端到端响应动作
    if action_name_or_text in domain.action_texts:
        return ActionEndToEndResponse(action_name_or_text)

    # 如果是发声动作，返回机器人响应动作
    if action_name_or_text.startswith(UTTER_PREFIX):
        return ActionBotResponse(action_name_or_text)

    # 检查是否为表单动作
    is_form = action_name_or_text in domain.form_names
    # 用户可以通过定义与表单同名的动作来覆盖表单
    user_overrode_form_action = is_form and action_name_or_text in domain.user_actions
    if is_form and not user_overrode_form_action:
        from rasa.core.actions.forms import FormAction

        return FormAction(action_name_or_text, action_endpoint)

    # 返回远程动作
    return RemoteAction(action_name_or_text, action_endpoint)


def create_bot_utterance(message: Dict[Text, Any]) -> BotUttered:
    """从消息创建BotUttered事件。
    
    Args:
        message: 包含消息数据的字典
        
    Returns:
        BotUttered事件对象
    """
    bot_message = BotUttered(
        text=message.pop("text", None),                    # 提取文本内容
        data={
            "elements": message.pop("elements", None),     # 提取元素
            "quick_replies": message.pop("quick_replies", None),  # 提取快速回复
            "buttons": message.pop("buttons", None),       # 提取按钮
            # 为了向后兼容，如果没有其他附件，我们需要将图像设置为附件
            # （`.get`是有意的 - 不使用`pop`，因为我们仍然需要图像属性在下一行设置）
            "attachment": message.pop("attachment", None) or message.get("image", None),
            "image": message.pop("image", None),           # 提取图像
            "custom": message.pop("custom", None),         # 提取自定义数据
        },
        metadata=message,                                  # 剩余的消息作为元数据
    )
    return bot_message


class Action:
    """响应对话状态要采取的下一个动作。"""

    def name(self) -> Text:
        """此简单动作的唯一标识符。
        
        Returns:
            动作名称
        """
        raise NotImplementedError

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """执行此动作的副作用。

        Args:
            nlg: 用于响应生成的``nlg``
            output_channel: 发送结果消息的``output_channel``
            tracker: 当前用户的状态跟踪器。您可以使用``tracker.get_slot(slot_name)``
                访问槽位值，最新的用户消息是``tracker.latest_message.text``
            domain: 机器人的领域

        Returns:
            :class:`rasa.core.events.Event`实例列表
        """
        raise NotImplementedError

    def __str__(self) -> Text:
        """返回表单的文本表示。
        
        Returns:
            动作的字符串表示
        """
        return f"{self.__class__.__name__}('{self.name()}')"

    def event_for_successful_execution(
        self, prediction: PolicyPrediction
    ) -> ActionExecuted:
        """此动作成功执行时应记录的事件。

        Args:
            prediction: 导致此事件执行的预测。

        Returns:
            应记录到跟踪器的事件。
        """
        return ActionExecuted(
            self.name(),                        # 动作名称
            prediction.policy_name,             # 策略名称
            prediction.max_confidence,          # 最大置信度
            hide_rule_turn=prediction.hide_rule_turn,  # 是否隐藏规则轮次
            metadata=prediction.action_metadata,  # 动作元数据
        )


class ActionBotResponse(Action):
    """一个动作，其唯一效果是在运行时发声响应。"""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False) -> None:
        """创建动作。

        Args:
            name: 动作的名称。
            silent_fail: 如果为`True`，则在未为此动作定义响应时静默失败。
        """
        self.utter_action = name                # 发声动作名称
        self.silent_fail = silent_fail          # 是否静默失败

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """简单的运行实现，发声（希望已定义的）响应。"""
        kwargs = {
            "domain_responses": domain.responses,  # 领域响应
        }

        # 生成消息
        message = await nlg.generate(
            self.utter_action,
            tracker,
            output_channel.name(),
            **kwargs,
        )
        # 如果消息生成失败
        if message is None:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for response '{}'."
                    "".format(self.utter_action)
                )
            return []
        # 设置发声动作
        message["utter_action"] = self.utter_action

        return [create_bot_utterance(message)]

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            动作名称
        """
        return self.utter_action


class ActionEndToEndResponse(Action):
    """向用户发声端到端响应的动作。"""

    def __init__(self, action_text: Text) -> None:
        """创建动作。

        Args:
            action_text: 端到端机器人响应的文本。
        """
        self.action_text = action_text          # 动作文本

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            动作名称（对于端到端动作，返回动作文本）
        """
        # 对于端到端动作，没有动作的标签（即名称）。
        # 我们通过返回机器人发送给用户的文本来伪造一个名称。
        return self.action_text

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作（完整文档字符串请参见父类）。"""
        message = {"text": self.action_text}    # 创建消息
        return [create_bot_utterance(message)]

    def event_for_successful_execution(
        self, prediction: PolicyPrediction
    ) -> ActionExecuted:
        """此动作成功执行时应记录的事件。

        Args:
            prediction: 导致此事件执行的预测。

        Returns:
            应记录到跟踪器的事件。
        """
        return ActionExecuted(
            policy=prediction.policy_name,      # 策略名称
            confidence=prediction.max_confidence,  # 置信度
            action_text=self.action_text,       # 动作文本
            hide_rule_turn=prediction.hide_rule_turn,  # 是否隐藏规则轮次
            metadata=prediction.action_metadata,  # 动作元数据
        )


class ActionRetrieveResponse(ActionBotResponse):
    """查询响应选择器以获取适当响应的动作。"""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False) -> None:
        """创建动作。请参见父类的文档字符串。"""
        super().__init__(name, silent_fail)
        self.action_name = name                 # 动作名称
        self.silent_fail = silent_fail          # 是否静默失败

    @staticmethod
    def intent_name_from_action(action_name: Text) -> Text:
        """从动作名称解析意图名称。
        
        Args:
            action_name: 动作名称
            
        Returns:
            意图名称
        """
        return action_name.split(UTTER_PREFIX)[1]

    def get_full_retrieval_name(
        self, tracker: "DialogueStateTracker"
    ) -> Optional[Text]:
        """返回动作的完整检索名称。

        从响应选择器提取检索意图并返回完整的动作发声名称。

        Args:
            tracker: 包含过去对话事件的跟踪器。

        Returns:
            如果最后一个用户发声包含响应选择器输出，则返回动作的完整检索名称，否则返回`None`。
        """
        latest_message = tracker.latest_message

        if latest_message is None:
            return None

        # 检查是否有响应选择器属性
        if RESPONSE_SELECTOR_PROPERTY_NAME not in latest_message.parse_data:
            return None

        response_selector_properties = latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME  # type: ignore[literal-required]
        ]

        # 检查是否有匹配的意图
        if (
            self.intent_name_from_action(self.action_name)
            in response_selector_properties
        ):
            query_key = self.intent_name_from_action(self.action_name)
        elif RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector_properties:
            query_key = RESPONSE_SELECTOR_DEFAULT_INTENT
        else:
            return None

        # 获取选中的响应
        selected = response_selector_properties[query_key]
        full_retrieval_utter_action = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_UTTER_ACTION_KEY
        ]
        return full_retrieval_utter_action

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """查询适当的响应并使用该响应创建机器人发声。"""
        latest_message = tracker.latest_message

        if latest_message is None:
            return []

        # 获取响应选择器属性
        response_selector_properties = latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME  # type: ignore[literal-required]
        ]

        # 确定查询键
        if (
            self.intent_name_from_action(self.action_name)
            in response_selector_properties
        ):
            query_key = self.intent_name_from_action(self.action_name)
        elif RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector_properties:
            query_key = RESPONSE_SELECTOR_DEFAULT_INTENT
        else:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for response action '{}'."
                    "".format(self.action_name)
                )
            return []

        logger.debug(f"Picking response from selector of type {query_key}")
        selected = response_selector_properties[query_key]

        # 用从响应选择器输出检索的完整发声动作
        # 覆盖ActionBotResponse的发声动作
        self.utter_action = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_UTTER_ACTION_KEY
        ]

        return await super().run(output_channel, nlg, tracker, domain)

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            动作名称
        """
        return self.action_name


class ActionBack(ActionBotResponse):
    """通过两个用户发声恢复跟踪器状态。"""

    def name(self) -> Text:
        """返回返回动作名称。
        
        Returns:
            返回动作名称
        """
        return ACTION_BACK_NAME

    def __init__(self) -> None:
        """初始化返回动作。"""
        super().__init__("utter_back", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        # 只有在响应可用时才发声
        evts = await super().run(output_channel, nlg, tracker, domain)

        # 添加两个用户发声恢复事件
        return evts + [UserUtteranceReverted(), UserUtteranceReverted()]


class ActionListen(Action):
    """任何轮次中的第一个动作 - 机器人等待用户消息。

    机器人应该停止采取进一步动作并等待用户说些什么。
    """

    def name(self) -> Text:
        """返回监听动作名称。
        
        Returns:
            监听动作名称
        """
        return ACTION_LISTEN_NAME

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        return []                               # 监听动作不产生事件


class ActionRestart(ActionBotResponse):
    """将跟踪器重置为其初始状态。

    如果可用，则发声重启响应。
    """

    def name(self) -> Text:
        """返回重启动作名称。
        
        Returns:
            重启动作名称
        """
        return ACTION_RESTART_NAME

    def __init__(self) -> None:
        """初始化重启动作。"""
        super().__init__("utter_restart", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        # 只有在响应可用时才发声
        evts = await super().run(output_channel, nlg, tracker, domain)

        # 添加重启事件
        return evts + [Restarted()]


class ActionSessionStart(Action):
    """应用对话会话开始。

    从前一个会话中获取所有`SlotSet`事件并将它们应用到新会话中。
    """

    def name(self) -> Text:
        """返回会话开始动作名称。
        
        Returns:
            会话开始动作名称
        """
        return ACTION_SESSION_START_NAME

    @staticmethod
    def _slot_set_events_from_tracker(
        tracker: "DialogueStateTracker",
    ) -> List["SlotSet"]:
        """从跟踪器获取SlotSet事件并携带键、值和元数据。
        
        Args:
            tracker: 对话状态跟踪器
            
        Returns:
            SlotSet事件列表
        """
        return [
            SlotSet(key=event.key, value=event.value, metadata=event.metadata)
            for event in tracker.applied_events()
            if isinstance(event, SlotSet)
        ]

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        _events: List[Event] = [SessionStarted()]  # 会话开始事件

        # 如果配置了槽位继承，则继承槽位
        if domain.session_config.carry_over_slots:
            _events.extend(self._slot_set_events_from_tracker(tracker))

        # 添加监听动作执行事件
        _events.append(ActionExecuted(ACTION_LISTEN_NAME))

        return _events


class ActionDefaultFallback(ActionBotResponse):
    """执行回退动作并返回到对话的前一个状态。"""

    def name(self) -> Text:
        """返回默认回退动作名称。
        
        Returns:
            默认回退动作名称
        """
        return ACTION_DEFAULT_FALLBACK_NAME

    def __init__(self) -> None:
        """初始化默认回退动作。"""
        super().__init__("utter_default", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        # 只有在响应可用时才发声
        evts = await super().run(output_channel, nlg, tracker, domain)

        # 添加用户话语回退事件
        return evts + [UserUtteranceReverted()]


class ActionDeactivateLoop(Action):
    """停用活动循环。"""

    def name(self) -> Text:
        """返回停用循环动作名称。
        
        Returns:
            停用循环动作名称
        """
        return ACTION_DEACTIVATE_LOOP_NAME

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        # 停用活动循环并清除请求槽位
        return [ActiveLoop(None), SlotSet(REQUESTED_SLOT, None)]


class RemoteAction(Action):
    """运行动作服务器动作。"""
    
    def __init__(self, name: Text, action_endpoint: Optional[EndpointConfig]) -> None:
        """初始化远程动作。

        Args:
            name: 动作名称
            action_endpoint: 动作端点配置
        """
        self._name = name                        # 动作名称
        self.action_endpoint = action_endpoint  # 动作端点配置

    def _action_call_format(
        self,
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> Dict[Text, Any]:
        """创建发送到动作服务器的请求JSON。
        
        Args:
            tracker: 对话状态跟踪器
            domain: 领域对象
            
        Returns:
            请求字典
        """
        from rasa.shared.core.trackers import EventVerbosity

        # 获取跟踪器的完整状态
        tracker_state = tracker.current_state(EventVerbosity.ALL)

        # 构建基本结果
        result = {
            "next_action": self._name,           # 下一个动作名称
            "sender_id": tracker.sender_id,      # 发送者ID
            "tracker": tracker_state,            # 跟踪器状态
            "version": rasa.__version__,         # Rasa版本
        }

        # 如果未启用选择性域或动作明确需要域，则添加域信息
        if (
            not self._is_selective_domain_enabled()
            or domain.does_custom_action_explicitly_need_domain(self.name())
        ):
            result["domain"] = domain.as_dict()

        return result

    def _is_selective_domain_enabled(self) -> bool:
        """检查是否启用了选择性域。
        
        Returns:
            是否启用选择性域
        """
        if self.action_endpoint is None:
            return False
        return bool(
            self.action_endpoint.kwargs.get(SELECTIVE_DOMAIN, DEFAULT_SELECTIVE_DOMAIN)
        )

    @staticmethod
    def action_response_format_spec() -> Dict[Text, Any]:
        """动作端点的预期响应模式。

        用于验证从动作端点返回的响应。
        
        Returns:
            响应模式字典
        """
        schema = {
            "type": "object",
            "properties": {
                "events": EVENTS_SCHEMA,                    # 事件模式
                "responses": {"type": "array", "items": {"type": "object"}},  # 响应数组
            },
        }
        return schema

    def _validate_action_result(self, result: Dict[Text, Any]) -> bool:
        """验证动作结果。
        
        Args:
            result: 动作结果字典
            
        Returns:
            验证是否成功
        """
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            # 使用模式验证结果
            validate(result, self.action_response_format_spec())
            return True
        except ValidationError as e:
            # 添加详细的错误信息
            e.message += (
                f". Failed to validate Action server response from API, "
                f"make sure your response from the Action endpoint is valid. "
                f"For more information about the format visit "
                f"{DOCS_BASE_URL}/custom-actions"
            )
            raise e

    @staticmethod
    async def _utter_responses(
        responses: List[Dict[Text, Any]],       # 响应列表
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
    ) -> List[BotUttered]:
        """使用动作端点生成的响应并发声。
        
        Args:
            responses: 响应列表
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            
        Returns:
            机器人发声事件列表
        """
        bot_messages = []
        for response in responses:
            # 获取生成的响应
            generated_response = response.pop("response", None)
            if generated_response:
                # 生成响应草稿
                draft = await nlg.generate(
                    generated_response, tracker, output_channel.name(), **response
                )
                if not draft:
                    continue
                # 添加发声动作信息
                draft["utter_action"] = generated_response
            else:
                draft = {}

            # 处理按钮
            buttons = response.pop("buttons", []) or []
            if buttons:
                draft.setdefault("buttons", [])
                draft["buttons"].extend(buttons)

            # 避免用空值覆盖`draft`值
            response = {k: v for k, v in response.items() if v}
            draft.update(response)
            bot_messages.append(create_bot_utterance(draft))

        return bot_messages

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        # 构建请求JSON体
        json_body = self._action_call_format(tracker, domain)
        if not self.action_endpoint:
            raise RasaException(
                f"Failed to execute custom action '{self.name()}' "
                f"because no endpoint is configured to run this "
                f"custom action. Please take a look at "
                f"the docs and set an endpoint configuration via the "
                f"--endpoints flag. "
                f"{DOCS_BASE_URL}/custom-actions"
            )

        try:
            logger.debug(
                "Calling action endpoint to run action '{}'.".format(self.name())
            )

            # 检查是否应该压缩请求
            should_compress = get_bool_env_variable(
                COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,
                DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,
            )

            # 使用插件管理器处理前缀剥离
            modified_json = plugin_manager().hook.prefix_stripping_for_custom_actions(
                json_body=json_body
            )
            # 发送请求到动作端点
            response: Any = await self.action_endpoint.request(
                json=modified_json if modified_json else json_body,
                method="post",
                timeout=DEFAULT_REQUEST_TIMEOUT,
                compress=should_compress,
            )
            # 如果修改了JSON，则使用插件管理器处理响应前缀
            if modified_json:
                plugin_manager().hook.prefixing_custom_actions_response(
                    json_body=json_body, response=response
                )
            # 验证动作结果
            self._validate_action_result(response)

            # 获取事件和响应
            events_json = response.get("events", [])
            responses = response.get("responses", [])
            # 处理响应并生成机器人消息
            bot_messages = await self._utter_responses(
                responses, output_channel, nlg, tracker
            )

            # 反序列化事件
            evts = events.deserialise_events(events_json)
            return cast(List[Event], bot_messages) + evts

        except ClientResponseError as e:
            # 处理400状态码（动作执行被拒绝）
            if e.status == 400:
                response_data = json.loads(e.text)
                exception = ActionExecutionRejection(
                    response_data["action_name"], response_data.get("error")
                )
                logger.error(exception.message)
                raise exception
            else:
                raise RasaException(
                    f"Failed to execute custom action '{self.name()}'"
                ) from e

        except aiohttp.ClientConnectionError as e:
            # 处理连接错误
            logger.error(
                f"Failed to run custom action '{self.name()}'. Couldn't connect "
                f"to the server at '{self.action_endpoint.url}'. "
                f"Is the server running? "
                f"Error: {e}"
            )
            raise RasaException(
                f"Failed to execute custom action '{self.name()}'. Couldn't connect "
                f"to the server at '{self.action_endpoint.url}."
            )

        except aiohttp.ClientError as e:
            # 不是所有错误都有状态属性，但如果有的话记录会很有帮助

            # noinspection PyUnresolvedReferences
            status = getattr(e, "status", None)
            raise RasaException(
                "Failed to run custom action '{}'. Action server "
                "responded with a non 200 status code of {}. "
                "Make sure your action server properly runs actions "
                "and returns a 200 once the action is executed. "
                "Error: {}".format(self.name(), status, e)
            )

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            动作名称
        """
        return self._name


class ActionExecutionRejection(RasaException):
    """抛出此异常将允许其他策略预测不同的动作。"""

    def __init__(self, action_name: Text, message: Optional[Text] = None) -> None:
        """创建新的动作执行拒绝异常。
        
        Args:
            action_name: 动作名称
            message: 可选的消息
        """
        self.action_name = action_name                    # 动作名称
        self.message = message or "Custom action '{}' rejected to run".format(
            action_name
        )
        super(ActionExecutionRejection, self).__init__()

    def __str__(self) -> Text:
        """返回异常消息。
        
        Returns:
            异常消息
        """
        return self.message


class ActionRevertFallbackEvents(Action):
    """回退在`TwoStageFallbackPolicy`期间完成的事件。

    这会回退在`TwoStageFallbackPolicy`回退期间完成的用户消息和机器人发声。
    通过这样做，不需要为不同的路径编写自定义故事，只需要为愉快路径编写。
    此功能已弃用，一旦`TwoStageFallbackPolicy`被移除就可以删除。
    """

    def name(self) -> Text:
        """返回回退事件回退动作名称。
        
        Returns:
            回退事件回退动作名称
        """
        return ACTION_REVERT_FALLBACK_EVENTS_NAME

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        from rasa.core.policies.two_stage_fallback import has_user_rephrased

        # 用户重新表述
        if has_user_rephrased(tracker):
            return _revert_successful_rephrasing(tracker)
        # 用户确认
        elif has_user_affirmed(tracker):
            return _revert_affirmation_events(tracker)
        else:
            return []


class ActionUnlikelyIntent(Action):
    """表示NLU预测的意图是意外的动作。

    此动作可以由`UnexpecTEDIntentPolicy`预测。
    """

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            动作名称
        """
        return ACTION_UNLIKELY_INTENT_NAME

    async def run(
        self,
        output_channel: "OutputChannel",        # 输出通道
        nlg: "NaturalLanguageGenerator",        # 自然语言生成器
        tracker: "DialogueStateTracker",        # 对话状态跟踪器
        domain: "Domain",                       # 领域对象
    ) -> List[Event]:
        """运行动作。完整文档字符串请参见父类。"""
        return []                               # 意外意图动作不产生事件


def has_user_affirmed(tracker: "DialogueStateTracker") -> bool:
    """指示最后执行的动作是否为`action_default_ask_affirmation`。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        是否最后执行了确认询问动作
    """
    return tracker.last_executed_action_has(ACTION_DEFAULT_ASK_AFFIRMATION_NAME)


def _revert_affirmation_events(tracker: "DialogueStateTracker") -> List[Event]:
    """回退确认事件。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        回退事件列表
    """
    revert_events = _revert_single_affirmation_events()

    # 用户确认重新表述的意图
    rephrased_intent = tracker.last_executed_action_has(
        name=ACTION_DEFAULT_ASK_REPHRASE_NAME, skip=1
    )
    if rephrased_intent:
        # 添加重新表述事件回退
        revert_events += _revert_rephrasing_events()

    # 获取最后一个用户事件
    last_user_event = tracker.get_last_event_for(UserUttered)
    if not last_user_event:
        raise TypeError("Cannot find last event to revert to.")

    # 深拷贝最后一个用户事件
    last_user_event = copy.deepcopy(last_user_event)
    # FIXME: `parse_data`的更好类型注解需要更大的重构（例如切换到dataclass）
    # 设置意图置信度为1.0
    last_user_event.parse_data["intent"]["confidence"] = 1.0  # type: ignore[typeddict-item]  # noqa: E501

    return revert_events + [last_user_event]


def _revert_single_affirmation_events() -> List[Event]:
    """回退单个确认事件。
    
    Returns:
        回退事件列表
    """
    return [
        UserUtteranceReverted(),  # 回退确认和请求
        # 回退原始意图（稍后需要重新添加）
        UserUtteranceReverted(),
        # 添加监听动作意图
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
    ]


def _revert_successful_rephrasing(tracker: "DialogueStateTracker") -> List[Event]:
    """回退成功的重新表述。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        回退事件列表
    """
    last_user_event = tracker.get_last_event_for(UserUttered)
    if not last_user_event:
        raise TypeError("Cannot find last event to revert to.")

    # 深拷贝最后一个用户事件
    last_user_event = copy.deepcopy(last_user_event)
    return _revert_rephrasing_events() + [last_user_event]


def _revert_rephrasing_events() -> List[Event]:
    """回退重新表述事件。
    
    Returns:
        回退事件列表
    """
    return [
        UserUtteranceReverted(),  # 移除重新表述
        # 移除反馈和重新表述请求
        UserUtteranceReverted(),
        # 移除确认请求和错误意图
        UserUtteranceReverted(),
        # 用监听动作替换动作
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
    ]


class ActionDefaultAskAffirmation(Action):
    """Default implementation which asks the user to affirm his intent.

    It is suggested to overwrite this default action with a custom action
    to have more meaningful prompts for the affirmations. E.g. have a
    description of the intent instead of its identifier name.
    """

    def name(self) -> Text:
        return ACTION_DEFAULT_ASK_AFFIRMATION_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        latest_message = tracker.latest_message
        if latest_message is None:
            raise TypeError(
                "Cannot find last user message for detecting fallback affirmation."
            )

        intent_to_affirm = latest_message.intent.get(INTENT_NAME_KEY)

        # FIXME: better type annotation for `parse_data` would require
        # a larger refactoring (e.g. switch to dataclass)
        intent_ranking = cast(
            List["IntentPrediction"],
            latest_message.parse_data.get(INTENT_RANKING_KEY) or [],
        )
        if (
            intent_to_affirm == DEFAULT_NLU_FALLBACK_INTENT_NAME
            and len(intent_ranking) > 1
        ):
            intent_to_affirm = intent_ranking[1][INTENT_NAME_KEY]  # type: ignore[literal-required] # noqa: E501

        affirmation_message = f"Did you mean '{intent_to_affirm}'?"

        message = {
            "text": affirmation_message,
            "buttons": [
                {"title": "Yes", "payload": f"/{intent_to_affirm}"},
                {"title": "No", "payload": f"/{USER_INTENT_OUT_OF_SCOPE}"},
            ],
            "utter_action": self.name(),
        }

        return [create_bot_utterance(message)]


class ActionDefaultAskRephrase(ActionBotResponse):
    """Default implementation which asks the user to rephrase his intent."""

    def name(self) -> Text:
        """Returns action default ask rephrase name."""
        return ACTION_DEFAULT_ASK_REPHRASE_NAME

    def __init__(self) -> None:
        """Initializes action default ask rephrase."""
        super().__init__("utter_ask_rephrase", silent_fail=True)


class ActionExtractSlots(Action):
    """Default action that runs after each user turn.

    Action is executed automatically in MessageProcessor.handle_message(...)
    before the next predicted action is run.

    Set slots to extracted values from user message
    according to assigned slot mappings.
    """

    def __init__(self, action_endpoint: Optional[EndpointConfig]) -> None:
        """Initializes default action extract slots."""
        self._action_endpoint = action_endpoint

    def name(self) -> Text:
        """Returns action_extract_slots name."""
        return ACTION_EXTRACT_SLOTS

    @staticmethod
    def _matches_mapping_conditions(
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker", slot_name: Text
    ) -> bool:
        slot_mapping_conditions = mapping.get(MAPPING_CONDITIONS)

        if not slot_mapping_conditions:
            return True

        if (
            tracker.is_active_loop_rejected
            and tracker.get_slot(REQUESTED_SLOT) == slot_name
        ):
            return False

        # check if found mapping conditions matches form
        for condition in slot_mapping_conditions:
            active_loop = condition.get(ACTIVE_LOOP)

            if active_loop and active_loop == tracker.active_loop_name:
                condition_requested_slot = condition.get(REQUESTED_SLOT)
                if not condition_requested_slot:
                    return True
                if condition_requested_slot == tracker.get_slot(REQUESTED_SLOT):
                    return True

            if active_loop is None and tracker.active_loop_name is None:
                return True

        return False

    @staticmethod
    def _verify_mapping_conditions(
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker", slot_name: Text
    ) -> bool:
        if mapping.get(MAPPING_CONDITIONS) and mapping[MAPPING_TYPE] != str(
            SlotMappingType.FROM_TRIGGER_INTENT
        ):
            if not ActionExtractSlots._matches_mapping_conditions(
                mapping, tracker, slot_name
            ):
                return False

        return True

    async def _run_custom_action(
        self,
        custom_action: Text,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        slot_events: List[Event] = []
        remote_action = RemoteAction(custom_action, self._action_endpoint)
        disallowed_types = set()

        try:
            custom_events = await remote_action.run(
                output_channel, nlg, tracker, domain
            )
            for event in custom_events:
                if isinstance(event, SlotSet):
                    slot_events.append(event)
                elif isinstance(event, BotUttered):
                    slot_events.append(event)
                else:
                    disallowed_types.add(event.type_name)
        except (RasaException, ClientResponseError) as e:
            logger.warning(
                f"Failed to execute custom action '{custom_action}' "
                f"as a result of error '{str(e)}'. The default action "
                f"'{self.name()}' failed to fill slots with custom "
                f"mappings."
            )

        for type_name in disallowed_types:
            logger.info(
                f"Running custom action '{custom_action}' has resulted "
                f"in an event of type '{type_name}'. This is "
                f"disallowed and the tracker will not be "
                f"updated with this event."
            )

        return slot_events

    async def _execute_custom_action(
        self,
        mapping: Dict[Text, Any],
        executed_custom_actions: Set[Text],
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> Tuple[List[Event], Set[Text]]:
        custom_action = mapping.get("action")

        if not custom_action or custom_action in executed_custom_actions:
            return [], executed_custom_actions

        slot_events = await self._run_custom_action(
            custom_action, output_channel, nlg, tracker, domain
        )

        executed_custom_actions.add(custom_action)

        return slot_events, executed_custom_actions

    async def _execute_validation_action(
        self,
        extraction_events: List[Event],
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        slot_events: List[SlotSet] = [
            event for event in extraction_events if isinstance(event, SlotSet)
        ]

        slot_candidates = "\n".join([e.key for e in slot_events])
        logger.debug(f"Validating extracted slots: {slot_candidates}")

        if ACTION_VALIDATE_SLOT_MAPPINGS not in domain.user_actions:
            return cast(List[Event], slot_events)

        _tracker = DialogueStateTracker.from_events(
            tracker.sender_id,
            tracker.events_after_latest_restart() + cast(List[Event], slot_events),
            slots=domain.slots,
        )
        validate_events = await self._run_custom_action(
            ACTION_VALIDATE_SLOT_MAPPINGS, output_channel, nlg, _tracker, domain
        )
        validated_slot_names = [
            event.key for event in validate_events if isinstance(event, SlotSet)
        ]

        # If the custom action doesn't return a SlotSet event for an extracted slot
        # candidate we assume that it was valid. The custom action has to return a
        # SlotSet(slot_name, None) event to mark a Slot as invalid.
        return validate_events + [
            event for event in slot_events if event.key not in validated_slot_names
        ]

    def _fails_unique_entity_mapping_check(
        self,
        slot_name: Text,
        mapping: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> bool:
        from rasa.core.actions.forms import FormAction

        if mapping[MAPPING_TYPE] != str(SlotMappingType.FROM_ENTITY):
            return False

        form_name = tracker.active_loop_name

        if not form_name:
            return False

        if tracker.get_slot(REQUESTED_SLOT) == slot_name:
            return False

        form = FormAction(form_name, self._action_endpoint)

        if slot_name not in form.required_slots(domain):
            return False

        if form.entity_mapping_is_unique(mapping, domain):
            return False

        return True

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        slot_events: List[Event] = []
        executed_custom_actions: Set[Text] = set()

        user_slots = [
            slot for slot in domain.slots if slot.name not in DEFAULT_SLOT_NAMES
        ]

        for slot in user_slots:
            for mapping in slot.mappings:
                mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))

                if not SlotMapping.check_mapping_validity(
                    slot_name=slot.name,
                    mapping_type=mapping_type,
                    mapping=mapping,
                    domain=domain,
                ):
                    continue

                intent_is_desired = SlotMapping.intent_is_desired(
                    mapping, tracker, domain
                )

                if not intent_is_desired:
                    continue

                if not ActionExtractSlots._verify_mapping_conditions(
                    mapping, tracker, slot.name
                ):
                    continue

                if self._fails_unique_entity_mapping_check(
                    slot.name, mapping, tracker, domain
                ):
                    continue

                if mapping_type.is_predefined_type():
                    value = extract_slot_value_from_predefined_mapping(
                        mapping_type, mapping, tracker
                    )
                else:
                    value = None

                if value:
                    if not isinstance(slot, ListSlot):
                        value = value[-1]

                    if value is not None or tracker.get_slot(slot.name) is not None:
                        slot_events.append(SlotSet(slot.name, value))
                        break

                should_fill_custom_slot = mapping_type == SlotMappingType.CUSTOM

                if should_fill_custom_slot:
                    (
                        custom_evts,
                        executed_custom_actions,
                    ) = await self._execute_custom_action(
                        mapping,
                        executed_custom_actions,
                        output_channel,
                        nlg,
                        tracker,
                        domain,
                    )
                    slot_events.extend(custom_evts)

        validated_events = await self._execute_validation_action(
            slot_events, output_channel, nlg, tracker, domain
        )
        return validated_events


def extract_slot_value_from_predefined_mapping(
    mapping_type: SlotMappingType,
    mapping: Dict[Text, Any],
    tracker: "DialogueStateTracker",
) -> List[Any]:
    """Extracts slot value if slot has an applicable predefined mapping."""
    should_fill_entity_slot = (
        mapping_type == SlotMappingType.FROM_ENTITY
        and SlotMapping.entity_is_desired(mapping, tracker)
    )

    should_fill_intent_slot = mapping_type == SlotMappingType.FROM_INTENT

    should_fill_text_slot = mapping_type == SlotMappingType.FROM_TEXT

    active_loops_in_mapping_conditions = [
        active_loop.get(ACTIVE_LOOP)
        for active_loop in mapping.get(MAPPING_CONDITIONS, [])
    ]

    trigger_mapping_condition_met = True

    if tracker.active_loop_name is None:
        trigger_mapping_condition_met = False
    elif (
        active_loops_in_mapping_conditions
        and tracker.active_loop_name is not None
        and (tracker.active_loop_name not in active_loops_in_mapping_conditions)
    ):
        trigger_mapping_condition_met = False

    should_fill_trigger_slot = (
        mapping_type == SlotMappingType.FROM_TRIGGER_INTENT
        and trigger_mapping_condition_met
    )

    value: List[Any] = []
    if should_fill_entity_slot:
        value = list(
            tracker.get_latest_entity_values(
                mapping.get(ENTITY_ATTRIBUTE_TYPE),
                mapping.get(ENTITY_ATTRIBUTE_ROLE),
                mapping.get(ENTITY_ATTRIBUTE_GROUP),
            )
        )
    elif should_fill_intent_slot or should_fill_trigger_slot:
        value = [mapping.get("value")]
    elif should_fill_text_slot:
        value = [
            tracker.latest_message.text if tracker.latest_message is not None else None
        ]

    return value
