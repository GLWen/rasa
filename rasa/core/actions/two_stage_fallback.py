# =============================================================================
# Rasa Core Two Stage Fallback 两阶段回退动作模块
# 本模块实现了两阶段回退机制，用于处理 NLU 预测置信度低的情况
# =============================================================================

# 导入标准库模块
import copy  # 深拷贝功能
import time  # 时间处理
from typing import List, Text, Optional  # 类型注解

# 导入 Rasa 核心模块
from rasa.core.actions import action  # 动作处理模块
from rasa.core.actions.loops import LoopAction  # 循环动作基类
from rasa.core.channels import OutputChannel  # 输出通道接口
from rasa.shared.core.domain import Domain  # 领域模型

# 导入事件相关类
from rasa.shared.core.events import (
    Event,                    # 事件基类
    UserUtteranceReverted,   # 用户话语回退事件
    ActionExecuted,          # 动作执行事件
    UserUttered,             # 用户话语事件
    ActiveLoop,              # 活跃循环事件
)

# 导入其他核心模块
from rasa.core.nlg import NaturalLanguageGenerator  # 自然语言生成器
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity  # 对话状态跟踪器和事件详细程度
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME  # 默认 NLU 回退意图名称

# 导入动作和意图常量
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,           # 用户意图超出范围
    ACTION_LISTEN_NAME,                 # 监听动作名称
    ACTION_DEFAULT_FALLBACK_NAME,       # 默认回退动作名称
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME, # 默认询问确认动作名称
    ACTION_DEFAULT_ASK_REPHRASE_NAME,   # 默认询问重述动作名称
    ACTION_TWO_STAGE_FALLBACK_NAME,     # 两阶段回退动作名称
)

# 导入 NLU 常量
from rasa.shared.nlu.constants import INTENT, PREDICTED_CONFIDENCE_KEY  # 意图和预测置信度键名
from rasa.utils.endpoints import EndpointConfig  # 端点配置


class TwoStageFallbackAction(LoopAction):
    """两阶段回退动作类。
    
    两阶段回退动作是 Rasa 中用于处理 NLU 预测置信度低的情况的循环动作。
    它实现了两阶段回退机制：
    1. 第一阶段：询问用户确认（affirmation）
    2. 第二阶段：询问用户重述（rephrase）
    """

    def __init__(self, action_endpoint: Optional[EndpointConfig] = None) -> None:
        """初始化两阶段回退动作。
        
        Args:
            action_endpoint: 动作端点配置，用于执行自定义动作
        """
        self._action_endpoint = action_endpoint  # 存储动作端点配置

    def name(self) -> Text:
        """返回动作名称。
        
        Returns:
            两阶段回退动作的名称
        """
        return ACTION_TWO_STAGE_FALLBACK_NAME

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """执行两阶段回退的主要逻辑。
        
        Args:
            output_channel: 输出通道，用于发送消息给用户
            nlg: 自然语言生成器，用于生成响应
            tracker: 对话状态跟踪器，跟踪对话状态
            domain: 领域模型，包含对话配置
            events_so_far: 到目前为止的事件列表
            
        Returns:
            执行回退逻辑产生的事件列表
        """
        # 检查用户是否应该确认
        if _user_should_affirm(tracker, events_so_far):
            # 如果应该确认，则询问用户确认
            return await self._ask_affirm(output_channel, nlg, tracker, domain)

        # 否则询问用户重述
        return await self._ask_rephrase(output_channel, nlg, tracker, domain)

    async def _ask_affirm(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        """询问用户确认。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            询问确认产生的事件列表
        """
        # 创建询问确认动作
        affirm_action = action.action_for_name_or_text(
            ACTION_DEFAULT_ASK_AFFIRMATION_NAME, domain, self._action_endpoint
        )

        # 执行询问确认动作
        return await affirm_action.run(output_channel, nlg, tracker, domain)

    async def _ask_rephrase(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        """询问用户重述。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            询问重述产生的事件列表
        """
        # 创建询问重述动作
        rephrase = action.action_for_name_or_text(
            ACTION_DEFAULT_ASK_REPHRASE_NAME, domain, self._action_endpoint
        )

        # 执行询问重述动作
        return await rephrase.run(output_channel, nlg, tracker, domain)

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        """检查两阶段回退是否完成。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            如果回退完成则返回 True，否则返回 False
        """
        # 检查用户是否已澄清（意图不是回退或超出范围）
        _user_clarified = _last_intent_name(tracker) not in [
            DEFAULT_NLU_FALLBACK_INTENT_NAME,  # 默认 NLU 回退意图
            USER_INTENT_OUT_OF_SCOPE,          # 用户意图超出范围
        ]
        
        # 回退完成的条件：用户已澄清 OR 连续两次回退 OR 第二次确认失败
        return (
            _user_clarified                    # 用户已澄清
            or _two_fallbacks_in_a_row(tracker)  # 连续两次回退
            or _second_affirmation_failed(tracker)  # 第二次确认失败
        )

    async def deactivate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """停用两阶段回退。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            停用回退产生的事件列表
        """
        # 如果连续两次回退或第二次确认失败，则放弃
        if _two_fallbacks_in_a_row(tracker) or _second_affirmation_failed(tracker):
            return await self._give_up(output_channel, nlg, tracker, domain)

        # 回退回退事件
        reverted_event: List[Event] = [UserUtteranceReverted()]  # 创建用户话语回退事件
        return reverted_event + _message_clarification(tracker)  # 返回回退事件和消息澄清

    async def _give_up(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Event]:
        """放弃回退，执行默认回退动作。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            默认回退动作产生的事件列表
        """
        # 创建默认回退动作
        fallback = action.action_for_name_or_text(
            ACTION_DEFAULT_FALLBACK_NAME, domain, self._action_endpoint
        )

        # 执行默认回退动作
        return await fallback.run(output_channel, nlg, tracker, domain)


def _last_intent_name(tracker: DialogueStateTracker) -> Optional[Text]:
    """获取最后一个用户消息的意图名称。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        最后一个用户消息的意图名称，如果没有则返回 None
    """
    last_message = tracker.latest_message  # 获取最新消息
    if not last_message:
        return None

    return last_message.intent_name  # 返回意图名称


def _two_fallbacks_in_a_row(tracker: DialogueStateTracker) -> bool:
    """检查是否连续两次回退。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        如果连续两次回退则返回 True，否则返回 False
    """
    return _last_n_intent_names(tracker, 2) == [
        DEFAULT_NLU_FALLBACK_INTENT_NAME,  # 默认 NLU 回退意图
        DEFAULT_NLU_FALLBACK_INTENT_NAME,  # 默认 NLU 回退意图
    ]


def _last_n_intent_names(
    tracker: DialogueStateTracker, number_of_last_intent_names: int
) -> List[Optional[Text]]:
    """获取最后 N 个用户消息的意图名称。
    
    Args:
        tracker: 对话状态跟踪器
        number_of_last_intent_names: 要获取的意图名称数量
        
    Returns:
        最后 N 个用户消息的意图名称列表
    """
    intent_names: List[Optional[Text]] = []  # 意图名称列表
    
    # 遍历最后 N 个消息
    for i in range(number_of_last_intent_names):
        message = tracker.get_last_event_for(
            (UserUttered, UserUtteranceReverted),  # 用户话语或回退事件
            skip=i,  # 跳过前 i 个事件
            event_verbosity=EventVerbosity.AFTER_RESTART,  # 重启后的事件详细程度
        )
        if isinstance(message, UserUttered):  # 如果是用户话语事件
            intent_names.append(message.intent.get("name"))  # 添加意图名称

    return intent_names


def _user_should_affirm(
    tracker: DialogueStateTracker, events_so_far: List[Event]
) -> bool:
    """检查用户是否应该确认。
    
    Args:
        tracker: 对话状态跟踪器
        events_so_far: 到目前为止的事件列表
        
    Returns:
        如果用户应该确认则返回 True，否则返回 False
    """
    # 检查回退是否刚刚被激活
    fallback_was_just_activated = any(
        isinstance(event, ActiveLoop) for event in events_so_far
    )
    if fallback_was_just_activated:
        return True

    # 检查最后一个意图是否是回退意图
    return _last_intent_name(tracker) == DEFAULT_NLU_FALLBACK_INTENT_NAME


def _second_affirmation_failed(tracker: DialogueStateTracker) -> bool:
    """检查第二次确认是否失败。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        如果第二次确认失败则返回 True，否则返回 False
    """
    return _last_n_intent_names(tracker, 3) == [
        USER_INTENT_OUT_OF_SCOPE,          # 用户意图超出范围
        DEFAULT_NLU_FALLBACK_INTENT_NAME,  # 默认 NLU 回退意图
        USER_INTENT_OUT_OF_SCOPE,          # 用户意图超出范围
    ]


def _message_clarification(tracker: DialogueStateTracker) -> List[Event]:
    """创建消息澄清事件。
    
    Args:
        tracker: 对话状态跟踪器
        
    Returns:
        消息澄清事件列表
        
    Raises:
        TypeError: 如果最新消息不在跟踪器上
    """
    latest_message = tracker.latest_message  # 获取最新消息
    if not latest_message:
        raise TypeError(
            "Cannot issue message clarification because "
            "latest message is not on tracker."
        )

    # 深拷贝最新消息
    clarification = copy.deepcopy(latest_message)
    # 设置意图预测置信度为 1.0
    clarification.parse_data[INTENT][PREDICTED_CONFIDENCE_KEY] = 1.0  # type: ignore[literal-required]  # noqa E501
    # 更新时间戳
    clarification.timestamp = time.time()
    # 返回动作执行事件和澄清消息
    return [ActionExecuted(ACTION_LISTEN_NAME), clarification]
