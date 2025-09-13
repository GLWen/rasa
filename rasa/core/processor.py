# 标准库导入
import copy  # 深拷贝和浅拷贝操作
import logging  # 日志记录模块
import structlog  # 结构化日志记录
import os  # 操作系统接口
from pathlib import Path  # 路径处理
import tarfile  # tar文件处理
import time  # 时间相关操作
from types import LambdaType  # Lambda类型
from typing import Any, Dict, List, Optional, Text, Tuple, Union  # 类型提示

# Rasa 内部模块导入
from rasa.core.http_interpreter import RasaNLUHttpInterpreter  # HTTP NLU解释器
from rasa.engine import loader  # 模型加载器
from rasa.engine.constants import PLACEHOLDER_MESSAGE, PLACEHOLDER_TRACKER  # 引擎常量
from rasa.engine.runner.dask import DaskGraphRunner  # Dask图运行器
from rasa.engine.storage.local_model_storage import LocalModelStorage  # 本地模型存储
from rasa.engine.storage.storage import ModelMetadata  # 模型元数据
from rasa.model import get_latest_model  # 获取最新模型
from rasa.plugin import plugin_manager  # 插件管理器
from rasa.shared.data import TrainingType  # 训练类型
import rasa.shared.utils.io  # 共享IO工具
import rasa.core.actions.action  # 动作模块
from rasa.core import jobs  # 任务调度
from rasa.core.actions.action import Action  # 动作基类
from rasa.core.channels.channel import (  # 通道相关类
    CollectingOutputChannel,  # 收集输出通道
    OutputChannel,  # 输出通道基类
    UserMessage,  # 用户消息类
)
import rasa.core.utils  # 核心工具
from rasa.core.policies.policy import PolicyPrediction  # 策略预测
from rasa.engine.runner.interface import GraphRunner  # 图运行器接口
from rasa.exceptions import ActionLimitReached, ModelNotFound  # 异常类
from rasa.shared.core.constants import (  # 核心常量
    USER_INTENT_RESTART,  # 用户重启意图
    ACTION_LISTEN_NAME,  # 监听动作名称
    ACTION_SESSION_START_NAME,  # 会话开始动作名称
    FOLLOWUP_ACTION,  # 后续动作
    SESSION_START_METADATA_SLOT,  # 会话开始元数据槽
    ACTION_EXTRACT_SLOTS,  # 提取槽位动作
)
from rasa.shared.core.events import (  # 核心事件类
    ActionExecutionRejected,  # 动作执行被拒绝
    BotUttered,  # 机器人话语
    Event,  # 事件基类
    ReminderCancelled,  # 提醒取消
    ReminderScheduled,  # 提醒安排
    SlotSet,  # 槽位设置
    UserUttered,  # 用户话语
    ActionExecuted,  # 动作执行
)
from rasa.shared.constants import (  # 共享常量
    ASSISTANT_ID_KEY,  # 助手ID键
    DOCS_URL_DOMAINS,  # 域文档URL
    DEFAULT_SENDER_ID,  # 默认发送者ID
    DOCS_URL_POLICIES,  # 策略文档URL
    UTTER_PREFIX,  # 话语前缀
)
from rasa.core.nlg import NaturalLanguageGenerator  # 自然语言生成器
from rasa.core.lock_store import LockStore  # 锁存储
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name  # 临时目录工具
import rasa.core.tracker_store  # 跟踪器存储
import rasa.core.actions.action  # 动作模块
import rasa.shared.core.trackers  # 共享跟踪器
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity  # 对话状态跟踪器和事件详细程度
from rasa.shared.nlu.constants import (  # NLU常量
    ENTITIES,  # 实体
    INTENT,  # 意图
    INTENT_NAME_KEY,  # 意图名称键
    INTENT_RESPONSE_KEY,  # 意图响应键
    PREDICTED_CONFIDENCE_KEY,  # 预测置信度键
    FULL_RETRIEVAL_INTENT_NAME_KEY,  # 完整检索意图名称键
    RESPONSE_SELECTOR,  # 响应选择器
    RESPONSE,  # 响应
    TEXT,  # 文本
)
from rasa.utils.endpoints import EndpointConfig  # 端点配置

# 日志记录器
logger = logging.getLogger(__name__)  # 标准日志记录器
structlogger = structlog.get_logger()  # 结构化日志记录器

# 最大预测次数配置
MAX_NUMBER_OF_PREDICTIONS = int(os.environ.get("MAX_NUMBER_OF_PREDICTIONS", "10"))  # 从环境变量获取最大预测次数，默认为10


class MessageProcessor:
    """消息处理器是与机器人模型通信的接口。"""

    def __init__(
        self,
        model_path: Union[Text, Path],
        tracker_store: rasa.core.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: NaturalLanguageGenerator,
        action_endpoint: Optional[EndpointConfig] = None,
        max_number_of_predictions: int = MAX_NUMBER_OF_PREDICTIONS,
        on_circuit_break: Optional[LambdaType] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ) -> None:
        """初始化消息处理器。"""
        self.nlg = generator  # 自然语言生成器
        self.tracker_store = tracker_store  # 跟踪器存储
        self.lock_store = lock_store  # 锁存储
        self.max_number_of_predictions = max_number_of_predictions  # 最大预测次数
        self.on_circuit_break = on_circuit_break  # 熔断回调函数
        self.action_endpoint = action_endpoint  # 动作端点配置
        self.model_filename, self.model_metadata, self.graph_runner = self._load_model(  # 加载模型并获取元数据
            model_path
        )

        if self.model_metadata.assistant_id is None:  # 如果模型元数据中没有助手ID
            rasa.shared.utils.io.raise_warning(  # 发出警告
                f"The model metadata does not contain a value for the "
                f"'{ASSISTANT_ID_KEY}' attribute. Check that 'config.yml' "
                f"file contains a value for the '{ASSISTANT_ID_KEY}' key "
                f"and re-train the model. Failure to do so will result in "
                f"streaming events without a unique assistant identifier.",
                UserWarning,
            )

        self.model_path = Path(model_path)  # 模型路径
        self.domain = self.model_metadata.domain  # 对话域
        self.http_interpreter = http_interpreter  # HTTP解释器

    @staticmethod
    def _load_model(
        model_path: Union[Text, Path]
    ) -> Tuple[Text, ModelMetadata, GraphRunner]:
        """使用图模型加载器从给定路径解包模型。"""
        try:
            if os.path.isfile(model_path):  # 如果路径是文件
                model_tar = model_path  # 直接使用该文件
            else:
                model_file_path = get_latest_model(model_path)  # 获取最新模型文件
                if not model_file_path:  # 如果没有找到模型文件
                    raise ModelNotFound(f"No model found at path '{model_path}'.")  # 抛出模型未找到异常
                model_tar = model_file_path  # 使用找到的模型文件
        except TypeError:  # 捕获类型错误
            raise ModelNotFound(f"Model {model_path} can not be loaded.")  # 抛出模型无法加载异常

        logger.info(f"Loading model {model_tar}...")  # 记录加载模型的日志
        with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:  # 创建临时目录
            try:
                metadata, runner = loader.load_predict_graph_runner(  # 加载预测图运行器
                    Path(temporary_directory),  # 临时目录路径
                    Path(model_tar),  # 模型tar文件路径
                    LocalModelStorage,  # 本地模型存储类
                    DaskGraphRunner,  # Dask图运行器类
                )
                return os.path.basename(model_tar), metadata, runner  # 返回模型文件名、元数据和运行器
            except tarfile.ReadError:  # 捕获tar文件读取错误
                raise ModelNotFound(f"Model {model_path} can not be loaded.")  # 抛出模型无法加载异常

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """使用此处理器处理单个消息。"""
        # TODO 如有必要，预处理消息
        tracker = await self.log_message(message, should_save_tracker=False)  # 记录消息但不保存跟踪器

        if self.model_metadata.training_type == TrainingType.NLU:  # 如果只是NLU模型
            await self.save_tracker(tracker)  # 保存跟踪器
            rasa.shared.utils.io.raise_warning(  # 发出警告
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None  # 返回None，因为没有核心模型

        tracker = await self.run_action_extract_slots(message.output_channel, tracker)  # 运行槽位提取动作

        await self._run_prediction_loop(message.output_channel, tracker)  # 运行预测循环

        await self.run_anonymization_pipeline(tracker)  # 运行匿名化管道

        await self.save_tracker(tracker)  # 保存跟踪器

        # todo 返回消息
        if isinstance(message.output_channel, CollectingOutputChannel):  # 如果是收集输出通道
            return message.output_channel.messages  # 返回收集的消息

        return None  # 返回None

    async def run_action_extract_slots(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> DialogueStateTracker:
        """运行动作以提取槽位并相应地更新跟踪器。

        Args:
            output_channel: 与传入用户消息关联的输出通道。
            tracker: 表示对话状态的跟踪器。

        Returns:
            给定的（已更新的）跟踪器
        """
        action_extract_slots = rasa.core.actions.action.action_for_name_or_text(  # 获取槽位提取动作
            ACTION_EXTRACT_SLOTS, self.domain, self.action_endpoint
        )
        extraction_events = await action_extract_slots.run(  # 运行槽位提取动作
            output_channel, self.nlg, tracker, self.domain
        )

        await self._send_bot_messages(extraction_events, tracker, output_channel)  # 发送机器人消息

        tracker.update_with_events(extraction_events, self.domain)  # 用提取事件更新跟踪器

        structlogger.debug(  # 记录调试日志
            "processor.extract.slots",
            action_extract_slot=ACTION_EXTRACT_SLOTS,
            len_extraction_events=len(extraction_events),
            rasa_events=copy.deepcopy(extraction_events),
        )

        return tracker  # 返回更新后的跟踪器

    async def run_anonymization_pipeline(self, tracker: DialogueStateTracker) -> None:
        """在跟踪器的新事件上运行匿名化管道。

        Args:
            tracker: 表示对话状态的跟踪器。
        """
        anonymization_pipeline = plugin_manager().hook.get_anonymization_pipeline()  # 获取匿名化管道
        if anonymization_pipeline is None:  # 如果没有匿名化管道
            return None  # 直接返回

        old_tracker = await self.tracker_store.retrieve(tracker.sender_id)  # 获取旧的跟踪器
        new_events = rasa.shared.core.trackers.TrackerEventDiffEngine.event_difference(  # 计算事件差异
            old_tracker, tracker
        )

        for event in new_events:  # 遍历新事件
            body = {"sender_id": tracker.sender_id}  # 创建事件体
            body.update(event.as_dict())  # 更新事件字典
            anonymization_pipeline.run(body)  # 运行匿名化管道

    async def predict_next_for_sender_id(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """为给定的发送者ID预测下一个动作。

        Args:
            sender_id: 对话ID。

        Returns:
            下一个动作的预测。如果没有加载域或策略则返回`None`。
        """
        tracker = await self.fetch_tracker_and_update_session(sender_id)  # 获取并更新会话的跟踪器
        result = self.predict_next_with_tracker(tracker)  # 使用跟踪器预测下一个动作

        # 保存跟踪器状态以从此状态继续对话
        await self.save_tracker(tracker)  # 保存跟踪器

        return result  # 返回预测结果

    def predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART,
    ) -> Optional[Dict[Text, Any]]:
        """为给定的对话状态预测下一个动作。

        Args:
            tracker: 表示对话状态的跟踪器。
            verbosity: 返回对话状态的详细程度。

        Returns:
            下一个动作的预测。如果没有加载域或策略则返回`None`。
        """
        if self.model_metadata.training_type == TrainingType.NLU:  # 如果只是NLU模型
            rasa.shared.utils.io.raise_warning(  # 发出警告
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None  # 返回None

        prediction = self._predict_next_with_tracker(tracker)  # 使用跟踪器预测下一个动作

        scores = [  # 创建分数列表
            {"action": a, "score": p}  # 动作和分数字典
            for a, p in zip(self.domain.action_names_or_texts, prediction.probabilities)  # 遍历动作名称和概率
        ]
        return {  # 返回预测结果字典
            "scores": scores,  # 分数列表
            "policy": prediction.policy_name,  # 策略名称
            "confidence": prediction.max_confidence,  # 最大置信度
            "tracker": tracker.current_state(verbosity),  # 跟踪器当前状态
        }

    async def _update_tracker_session(
        self,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        metadata: Optional[Dict] = None,
    ) -> None:
        """检查`tracker`中的当前会话，如果过期则更新它。

        如果最新的跟踪器会话已过期，或者跟踪器尚未包含任何事件（仅考虑最后重启后的事件），
        则运行'action_session_start'。

        Args:
            metadata: 与传入用户消息关联的客户端发送的数据。
            tracker: 要检查的跟踪器。
            output_channel: 自定义`ActionSessionStart`中潜在话语的输出通道。
        """
        if not tracker.applied_events() or self._has_session_expired(tracker):  # 如果没有应用事件或会话已过期
            logger.debug(  # 记录调试日志
                f"Starting a new session for conversation ID '{tracker.sender_id}'."
            )

            action_session_start = self._get_action(ACTION_SESSION_START_NAME)  # 获取会话开始动作

            if metadata:  # 如果有元数据
                tracker.update(  # 更新跟踪器
                    SlotSet(SESSION_START_METADATA_SLOT, metadata), self.domain  # 设置会话开始元数据槽
                )

            await self._run_action(  # 运行动作
                action=action_session_start,  # 会话开始动作
                tracker=tracker,  # 跟踪器
                output_channel=output_channel,  # 输出通道
                nlg=self.nlg,  # 自然语言生成器
                prediction=PolicyPrediction.for_action_name(  # 为动作名称创建策略预测
                    self.domain, ACTION_SESSION_START_NAME
                ),
            )

    async def fetch_tracker_and_update_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """获取`sender_id`的跟踪器并更新其对话会话。

        如果创建了新的跟踪器，则运行`action_session_start`。

        Args:
            metadata: 与传入用户消息关联的客户端发送的数据。
            output_channel: 与传入用户消息关联的输出通道。
            sender_id: 要获取跟踪器的对话ID。

        Returns:
              `sender_id`的跟踪器。
        """
        tracker = await self.get_tracker(sender_id)  # 获取跟踪器

        await self._update_tracker_session(tracker, output_channel, metadata)  # 更新跟踪器会话

        return tracker  # 返回跟踪器

    async def fetch_tracker_with_initial_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and runs a session start if it's a new
        tracker.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = await self.get_tracker(sender_id)

        # run session start only if the tracker is empty
        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_tracker(self, conversation_id: Text) -> DialogueStateTracker:
        """获取对话的跟踪器。

        与`fetch_tracker_and_update_session`不同，这不会在对话开始时添加任何
        `action_session_start`或`session_start`事件。

        Args:
            conversation_id: 应检索其历史的对话ID。

        Returns:
            对话的跟踪器。如果是新对话则创建空跟踪器。
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID  # 使用默认发送者ID如果未提供

        tracker = await self.tracker_store.get_or_create_tracker(  # 获取或创建跟踪器
            conversation_id, append_action_listen=False  # 不追加监听动作
        )
        tracker.model_id = self.model_metadata.model_id  # 设置模型ID
        if tracker.assistant_id is None:  # 如果跟踪器没有助手ID
            tracker.assistant_id = self.model_metadata.assistant_id  # 设置助手ID
        return tracker  # 返回跟踪器

    async def fetch_full_tracker_with_initial_session(
        self,
        conversation_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Get the full tracker for a conversation, including events after a restart.

        Args:
            conversation_id: The ID of the conversation for which the history should be
                retrieved.
            output_channel: Output channel associated with the incoming user message.
            metadata: Data sent from client associated with the incoming user message.

        Returns:
            Tracker for the conversation. Creates an empty tracker with a new session
            initialized in case it's a new conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.get_or_create_full_tracker(
            conversation_id, False
        )
        tracker.model_id = self.model_metadata.model_id

        if tracker.assistant_id is None:
            tracker.assistant_id = self.model_metadata.assistant_id

        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_trackers_for_all_conversation_sessions(
        self, conversation_id: Text
    ) -> List[DialogueStateTracker]:
        """Fetches all trackers for a conversation.

        Individual trackers are returned for each conversation session found
        for `conversation_id`.

        Args:
            conversation_id: The ID of the conversation for which the trackers should
                be retrieved.

        Returns:
            Trackers for the conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.retrieve_full_tracker(conversation_id)

        return rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)

    async def log_message(
        self, message: UserMessage, should_save_tracker: bool = True
    ) -> DialogueStateTracker:
        """在属于消息的conversation_id的跟踪器上记录`message`。

        如果`should_save_tracker`为`True`，可选择保存跟踪器。如果此方法返回的跟踪器
        用于进一步处理并在稍后阶段保存，则可以跳过跟踪器保存。
        """
        tracker = await self.fetch_tracker_and_update_session(  # 获取并更新会话的跟踪器
            message.sender_id, message.output_channel, message.metadata
        )

        # TODO 解析消息
        await self._handle_message_with_tracker(message, tracker)  # 使用跟踪器处理消息

        if should_save_tracker:  # 如果需要保存跟踪器
            await self.save_tracker(tracker)  # 保存跟踪器

        return tracker  # 返回跟踪器

    async def execute_action(
        self,
        sender_id: Text,
        action_name: Text,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> Optional[DialogueStateTracker]:
        """为对话执行一个动作。

        注意，这可能导致意外的机器人行为。最好使用意图在对话中执行某些行为
        （例如，通过使用`trigger_external_user_uttered`）。

        Args:
            sender_id: 对话的ID。
            action_name: 要执行的动作名称。
            output_channel: 用于机器人响应的输出通道。
            nlg: 响应生成器。
            prediction: 动作的预测。

        Returns:
            新的对话状态。注意，新状态也会被持久化。
        """
        # 我们为每个用户都有一个Tracker实例
        # 它维护对话状态
        tracker = await self.fetch_tracker_and_update_session(sender_id, output_channel)  # 获取并更新会话的跟踪器

        action = self._get_action(action_name)  # 获取动作
        await self._run_action(action, tracker, output_channel, nlg, prediction)  # 运行动作

        # 保存跟踪器状态以从此状态继续对话
        await self.save_tracker(tracker)  # 保存跟踪器

        return tracker  # 返回跟踪器

    def predict_next_with_tracker_if_should(
        self, tracker: DialogueStateTracker
    ) -> Tuple[rasa.core.actions.action.Action, PolicyPrediction]:
        """预测机器人在看到x后应该采取的下一个动作。

        这应该被更高级的策略重写以使用ML来预测动作。

        Returns:
             下一个动作的索引和策略的预测。

        Raises:
            ActionLimitReached 如果已达到预测动作的限制。
        """
        should_predict_another_action = self.should_predict_another_action(  # 检查是否应该预测另一个动作
            tracker.latest_action_name
        )

        if self.is_action_limit_reached(tracker, should_predict_another_action):  # 如果达到动作限制
            raise ActionLimitReached(  # 抛出动作限制达到异常
                "The limit of actions to predict has been reached."
            )

        prediction = self._predict_next_with_tracker(tracker)  # 使用跟踪器预测下一个动作

        action = rasa.core.actions.action.action_for_index(  # 根据索引获取动作
            prediction.max_confidence_index, self.domain, self.action_endpoint
        )

        logger.debug(  # 记录调试日志
            f"Predicted next action '{action.name()}' with confidence "
            f"{prediction.max_confidence:.2f}."
        )

        return action, prediction  # 返回动作和预测

    @staticmethod
    def _is_reminder(e: Event, name: Text) -> bool:
        """检查事件是否是具有指定名称的提醒。"""
        return isinstance(e, ReminderScheduled) and e.name == name  # 检查是否是提醒安排事件且名称匹配

    @staticmethod
    def _is_reminder_still_valid(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """检查提醒后对话是否已重启。"""
        for e in reversed(tracker.applied_events()):  # 遍历应用的事件（逆序）
            if MessageProcessor._is_reminder(e, reminder_event.name):  # 如果找到匹配的提醒
                return True  # 返回True
        return False  # 在应用事件中未找到 --> 已重启

    @staticmethod
    def _has_message_after_reminder(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """检查用户在提醒后是否发送了消息。"""
        for e in reversed(tracker.events):  # 遍历事件（逆序）
            if MessageProcessor._is_reminder(e, reminder_event.name):  # 如果找到匹配的提醒
                return False  # 返回False

            if isinstance(e, UserUttered) and e.text:  # 如果是用户话语且有文本
                return True  # 返回True

        return True  # 跟踪器可能已重启

    async def handle_reminder(
        self,
        reminder_event: ReminderScheduled,
        sender_id: Text,
        output_channel: OutputChannel,
    ) -> None:
        """处理异步触发的提醒。"""
        async with self.lock_store.lock(sender_id):  # 使用锁确保同一发送者的操作串行
            tracker = await self.fetch_tracker_and_update_session(  # 获取并更新会话的跟踪器
                sender_id, output_channel
            )

            if (  # 如果满足以下条件之一
                reminder_event.kill_on_user_message  # 提醒在用户消息时终止
                and self._has_message_after_reminder(tracker, reminder_event)  # 且提醒后有用户消息
                or not self._is_reminder_still_valid(tracker, reminder_event)  # 或提醒不再有效
            ):
                logger.debug(  # 记录调试日志
                    f"Canceled reminder because it is outdated ({reminder_event})."
                )
            else:  # 否则
                intent = reminder_event.intent  # 获取提醒的意图
                entities: Union[List[Dict], Dict] = reminder_event.entities or {}  # 获取提醒的实体
                await self.trigger_external_user_uttered(  # 触发外部用户话语
                    intent, entities, tracker, output_channel
                )

    async def trigger_external_user_uttered(
        self,
        intent_name: Text,
        entities: Optional[Union[List[Dict[Text, Any]], Dict[Text, Text]]],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """触发外部消息。

        触发外部消息（类似用户消息，但不可见；
        用于，例如，提醒或trigger_intent端点）。

        Args:
            intent_name: 要触发的意图名称。
            entities: 要传递的实体。
            tracker: 应添加事件的跟踪器。
            output_channel: 输出通道。
        """
        if isinstance(entities, list):  # 如果实体是列表
            entity_list = entities  # 直接使用
        elif isinstance(entities, dict):  # 如果实体是字典
            # 允许简写表示法 {"ent1": "val1", "ent2": "val2", ...}。
            # 如果没有给出'start'、'end'或'extractor'等属性，这很有用，
            # 例如，对于外部事件。
            entity_list = [  # 创建实体列表
                {"entity": ent, "value": val} for ent, val in entities.items()
            ]
        elif not entities:  # 如果没有实体
            entity_list = []  # 空列表
        else:  # 其他情况
            rasa.shared.utils.io.raise_warning(  # 发出警告
                f"Invalid entity specification: {entities}. Assuming no entities."
            )
            entity_list = []  # 空列表

        # 将新事件的输入通道设置为最新的输入通道，这样
        # 我们就不会丢失这个属性。
        input_channel = tracker.get_latest_input_channel()  # 获取最新输入通道

        tracker.update(  # 更新跟踪器
            UserUttered.create_external(intent_name, entity_list, input_channel),  # 创建外部用户话语事件
            self.domain,  # 对话域
        )

        tracker = await self.run_action_extract_slots(output_channel, tracker)  # 运行槽位提取动作

        await self._run_prediction_loop(output_channel, tracker)  # 运行预测循环
        # 保存跟踪器状态以从此状态继续对话
        await self.save_tracker(tracker)  # 保存跟踪器

    @staticmethod
    def _log_slots(tracker: DialogueStateTracker) -> None:
        """记录当前设置的槽位。"""
        # 记录当前设置的槽位
        slot_values = "\n".join(  # 连接槽位值
            [f"\t{s.name}: {s.value}" for s in tracker.slots.values()]  # 格式化槽位名称和值
        )
        if slot_values.strip():  # 如果有槽位值
            structlogger.debug(  # 记录调试日志
                "processor.slots.log", slot_values=copy.deepcopy(slot_values)
            )

    def _check_for_unseen_features(self, parse_data: Dict[Text, Any]) -> None:
        """如果NLU解析数据包含未识别的特征，则警告用户。

        检查NLU解析获取的意图和实体
        与域进行对比，并警告用户那些不匹配的。
        还考虑默认意图列表，这些意图是有效的但不需要
        在域中列出。

        Args:
            parse_data: 要对照域检查的消息解析数据。
        """
        if not self.domain or self.domain.is_empty():  # 如果没有域或域为空
            return  # 直接返回

        intent = parse_data["intent"][INTENT_NAME_KEY]  # 获取意图名称
        if intent and intent not in self.domain.intents:  # 如果有意图且不在域中
            rasa.shared.utils.io.raise_warning(  # 发出警告
                f"Parsed an intent '{intent}' "
                f"which is not defined in the domain. "
                f"Please make sure all intents are listed in the domain.",
                docs=DOCS_URL_DOMAINS,
            )

        entities = parse_data["entities"] or []  # 获取实体列表
        for element in entities:  # 遍历实体
            entity = element["entity"]  # 获取实体名称
            if entity and entity not in self.domain.entities:  # 如果有实体且不在域中
                rasa.shared.utils.io.raise_warning(  # 发出警告
                    f"Parsed an entity '{entity}' "
                    f"which is not defined in the domain. "
                    f"Please make sure all entities are listed in the domain.",
                    docs=DOCS_URL_DOMAINS,
                )

    def _get_action(
        self, action_name: Text
    ) -> Optional[rasa.core.actions.action.Action]:
        """根据动作名称或文本获取动作。"""
        return rasa.core.actions.action.action_for_name_or_text(  # 根据名称或文本获取动作
            action_name, self.domain, self.action_endpoint
        )

    async def parse_message(
        self,
        message: UserMessage,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """解释传递的消息。

        Args:
            message: 要处理的消息。
            tracker: 要使用的跟踪器。
            only_output_properties: 如果为`True`，将输出限制为
                Message.only_output_properties。

        Returns:
            从消息中提取的解析数据。
        """
        if self.http_interpreter:  # 如果有HTTP解释器
            parse_data = await self.http_interpreter.parse(message)  # 使用HTTP解释器解析消息
        else:  # 否则
            if tracker is None:  # 如果没有跟踪器
                tracker = DialogueStateTracker.from_events(message.sender_id, [])  # 创建空跟踪器
            parse_data = self._parse_message_with_graph(  # 使用图解析消息
                message, tracker, only_output_properties
            )

        self._update_full_retrieval_intent(parse_data)  # 更新完整检索意图
        structlogger.debug(  # 记录调试日志
            "processor.message.parse",
            parse_data_text=copy.deepcopy(parse_data["text"]),
            parse_data_intent=parse_data["intent"],
            parse_data_entities=copy.deepcopy(parse_data["entities"]),
        )

        self._check_for_unseen_features(parse_data)  # 检查未识别的特征

        return parse_data  # 返回解析数据

    def _update_full_retrieval_intent(self, parse_data: Dict[Text, Any]) -> None:
        """使用完整检索意图更新解析数据。

        Args:
            parse_data: 要更新的消息解析数据。
        """
        intent_name = parse_data.get(INTENT, {}).get(INTENT_NAME_KEY)  # 获取意图名称
        response_selector = parse_data.get(RESPONSE_SELECTOR, {})  # 获取响应选择器
        all_retrieval_intents = response_selector.get("all_retrieval_intents", [])  # 获取所有检索意图
        if intent_name and intent_name in all_retrieval_intents:  # 如果有意图名称且在检索意图中
            retrieval_intent = (  # 获取检索意图
                response_selector.get(intent_name, {})
                .get(RESPONSE, {})
                .get(INTENT_RESPONSE_KEY)
            )
            parse_data[INTENT][FULL_RETRIEVAL_INTENT_NAME_KEY] = retrieval_intent  # 设置完整检索意图名称

    def _parse_message_with_graph(
        self,
        message: UserMessage,
        tracker: DialogueStateTracker,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """使用图解释传递的消息。

        Arguments:
            message: 要处理的消息
            tracker: 要使用的跟踪器
            only_output_properties: 如果为`True`，将输出限制为
                Message.only_output_properties。

        Returns:
            从消息中提取的解析数据。
        """
        results = self.graph_runner.run(  # 运行图
            inputs={PLACEHOLDER_MESSAGE: [message], PLACEHOLDER_TRACKER: tracker},  # 输入消息和跟踪器
            targets=[self.model_metadata.nlu_target],  # 目标NLU
        )
        parsed_messages = results[self.model_metadata.nlu_target]  # 获取解析的消息
        parsed_message = parsed_messages[0]  # 获取第一个解析消息
        parse_data = {  # 创建解析数据字典
            TEXT: "",  # 文本
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},  # 意图
            ENTITIES: [],  # 实体
        }
        parse_data.update(  # 更新解析数据
            parsed_message.as_dict(only_output_properties=only_output_properties)
        )
        return parse_data  # 返回解析数据

    async def _handle_message_with_tracker(
        self, message: UserMessage, tracker: DialogueStateTracker
    ) -> None:
        """使用跟踪器处理消息。"""

        if message.parse_data:  # 如果消息有解析数据
            parse_data = message.parse_data  # 直接使用
        else:  # 否则
            parse_data = await self.parse_message(message, tracker)  # 解析消息

        # 永远不要直接改变跟踪器
        # - 而是将其事件传递给日志
        tracker.update(  # 更新跟踪器
            UserUttered(  # 创建用户话语事件
                message.text,  # 消息文本
                parse_data["intent"],  # 意图
                parse_data["entities"],  # 实体
                parse_data,  # 解析数据
                input_channel=message.input_channel,  # 输入通道
                message_id=message.message_id,  # 消息ID
                metadata=message.metadata,  # 元数据
            ),
            self.domain,  # 对话域
        )

        if parse_data["entities"]:  # 如果有实体
            self._log_slots(tracker)  # 记录槽位

        logger.debug(  # 记录调试日志
            f"Logged UserUtterance - tracker now has {len(tracker.events)} events."
        )

    @staticmethod
    def _should_handle_message(tracker: DialogueStateTracker) -> bool:
        """检查是否应该处理消息。"""
        return not tracker.is_paused() or (  # 如果跟踪器未暂停或
            tracker.latest_message is not None  # 有最新消息
            and tracker.latest_message.intent.get(INTENT_NAME_KEY)  # 且意图名称
            == USER_INTENT_RESTART  # 是用户重启意图
        )

    def is_action_limit_reached(
        self, tracker: DialogueStateTracker, should_predict_another_action: bool
    ) -> bool:
        """检查是否已达到最大预测次数。

        Args:
            tracker: DialogueStateTracker实例。
            should_predict_another_action: 最后执行的动作是否允许
            预测更多动作。

        Returns:
            如果已达到预测动作的限制则返回`True`。
        """
        reversed_events = list(tracker.events)[::-1]  # 获取逆序事件列表
        num_predicted_actions = 0  # 预测动作计数

        for e in reversed_events:  # 遍历逆序事件
            if isinstance(e, ActionExecuted):  # 如果是动作执行事件
                if e.action_name in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME):  # 如果是监听或会话开始动作
                    break  # 跳出循环
                num_predicted_actions += 1  # 增加计数

        return (  # 返回是否达到限制
            num_predicted_actions >= self.max_number_of_predictions  # 预测动作数大于等于最大预测数
            and should_predict_another_action  # 且应该预测另一个动作
        )

    async def _run_prediction_loop(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> None:
        """运行预测循环。"""
        # 继续采取策略决定的动作，直到它选择'监听'
        should_predict_another_action = True  # 是否应该预测另一个动作

        # 动作循环。预测动作直到我们遇到动作监听
        while should_predict_another_action and self._should_handle_message(tracker):  # 当应该预测另一个动作且应该处理消息时
            # 这实际上只是调用策略的同名方法
            try:
                action, prediction = self.predict_next_with_tracker_if_should(tracker)  # 预测下一个动作
            except ActionLimitReached:  # 如果达到动作限制
                logger.warning(  # 记录警告日志
                    "Circuit breaker tripped. Stopped predicting "
                    f"more actions for sender '{tracker.sender_id}'."
                )
                if self.on_circuit_break:  # 如果有熔断回调
                    # 调用注册的回调
                    self.on_circuit_break(tracker, output_channel, self.nlg)  # 调用熔断回调
                break  # 跳出循环

            if prediction.is_end_to_end_prediction:  # 如果是端到端预测
                logger.debug(  # 记录调试日志
                    f"An end-to-end prediction was made which has triggered the 2nd "
                    f"execution of the default action '{ACTION_EXTRACT_SLOTS}'."
                )
                tracker = await self.run_action_extract_slots(output_channel, tracker)  # 运行槽位提取动作

            should_predict_another_action = await self._run_action(  # 运行动作并获取是否应该预测另一个动作
                action, tracker, output_channel, self.nlg, prediction
            )

    @staticmethod
    def should_predict_another_action(action_name: Text) -> bool:
        """确定处理器是否应该预测另一个动作。

        Args:
            action_name: 最后执行的动作名称。

        Returns:
            如果`action_name`是`ACTION_LISTEN_NAME`或
            `ACTION_SESSION_START_NAME`则返回`False`，否则返回`True`。
        """
        return action_name not in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME)  # 如果动作名称不是监听或会话开始则返回True

    async def execute_side_effects(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """发送机器人消息、安排和取消在事件数组中记录的提醒。"""
        await self._send_bot_messages(events, tracker, output_channel)  # 发送机器人消息
        await self._schedule_reminders(events, tracker, output_channel)  # 安排提醒
        await self._cancel_reminders(events, tracker)  # 取消提醒

    @staticmethod
    async def _send_bot_messages(
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """发送在事件数组中记录的所有机器人消息。"""
        for e in events:  # 遍历事件
            if not isinstance(e, BotUttered):  # 如果不是机器人话语事件
                continue  # 跳过

            await output_channel.send_response(tracker.sender_id, e.message())  # 发送响应

    async def _schedule_reminders(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """使用调度器安排作业以触发传递的提醒。

        具有相同`id`属性的提醒将相互覆盖
        （即，只有其中一个最终会运行）。
        """
        for e in events:  # 遍历事件
            if not isinstance(e, ReminderScheduled):  # 如果不是提醒安排事件
                continue  # 跳过

            (await jobs.scheduler()).add_job(  # 添加作业
                self.handle_reminder,  # 处理提醒方法
                "date",  # 日期类型
                run_date=e.trigger_date_time,  # 运行日期
                args=[e, tracker.sender_id, output_channel],  # 参数
                id=e.name,  # 作业ID
                replace_existing=True,  # 替换现有作业
                name=e.scheduled_job_name(tracker.sender_id),  # 作业名称
            )

    @staticmethod
    async def _cancel_reminders(
        events: List[Event], tracker: DialogueStateTracker
    ) -> None:
        """取消与`ReminderCancelled`事件匹配的提醒。"""
        # 由ReminderCancelled事件指定的所有提醒将被取消
        for event in events:  # 遍历事件
            if isinstance(event, ReminderCancelled):  # 如果是提醒取消事件
                scheduler = await jobs.scheduler()  # 获取调度器
                for scheduled_job in scheduler.get_jobs():  # 遍历所有作业
                    if event.cancels_job_with_name(  # 如果事件取消具有指定名称的作业
                        scheduled_job.name, tracker.sender_id
                    ):
                        scheduler.remove_job(scheduled_job.id)  # 移除作业

    async def _run_action(
        self,
        action: rasa.core.actions.action.Action,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> bool:
        """运行动作。"""
        # 事件和返回值用于在采取动作后更新跟踪器状态
        try:
            # 使用临时跟踪器，因为我们可能需要在拒绝的情况下丢弃策略事件。
            temporary_tracker = tracker.copy()  # 复制跟踪器
            temporary_tracker.update_with_events(prediction.events, self.domain)  # 用预测事件更新临时跟踪器
            events = await action.run(  # 运行动作
                output_channel, nlg, temporary_tracker, self.domain
            )
        except rasa.core.actions.action.ActionExecutionRejection:  # 如果动作执行被拒绝
            events = [  # 创建动作执行被拒绝事件
                ActionExecutionRejected(
                    action.name(), prediction.policy_name, prediction.max_confidence
                )
            ]
            tracker.update(events[0])  # 更新跟踪器
            return self.should_predict_another_action(action.name())  # 返回是否应该预测另一个动作
        except Exception:  # 如果发生其他异常
            logger.exception(  # 记录异常日志
                f"Encountered an exception while running action '{action.name()}'."
                "Bot will continue, but the actions events are lost. "
                "Please check the logs of your action server for "
                "more information."
            )
            events = []  # 空事件列表

        self._log_action_on_tracker(tracker, action, events, prediction)  # 在跟踪器上记录动作

        if any(isinstance(e, UserUttered) for e in events):  # 如果事件中有用户话语
            logger.debug(  # 记录调试日志
                f"A `UserUttered` event was returned by executing "
                f"action '{action.name()}'. This will run the default action "
                f"'{ACTION_EXTRACT_SLOTS}'."
            )
            tracker = await self.run_action_extract_slots(output_channel, tracker)  # 运行槽位提取动作

        if action.name() != ACTION_LISTEN_NAME and not action.name().startswith(  # 如果动作不是监听且不是话语前缀
            UTTER_PREFIX
        ):
            self._log_slots(tracker)  # 记录槽位

        await self.execute_side_effects(events, tracker, output_channel)  # 执行副作用

        return self.should_predict_another_action(action.name())  # 返回是否应该预测另一个动作

    def _log_action_on_tracker(
        self,
        tracker: DialogueStateTracker,
        action: Action,
        events: Optional[List[Event]],
        prediction: PolicyPrediction,
    ) -> None:
        """在跟踪器上记录动作。"""
        # 确保即使懒惰的程序员在动作结束时忘记输入`return []`或run方法
        # 由于某种原因返回`None`，代码仍然可以工作。
        if events is None:  # 如果事件为None
            events = []  # 设置为空列表

        action_was_rejected_manually = any(  # 检查动作是否被手动拒绝
            isinstance(event, ActionExecutionRejected) for event in events
        )
        if not action_was_rejected_manually:  # 如果动作没有被手动拒绝
            structlogger.debug(  # 记录调试日志
                "processor.actions.policy_prediction",
                prediction_events=copy.deepcopy(prediction.events),
            )
            tracker.update_with_events(prediction.events, self.domain)  # 用预测事件更新跟踪器

            # 记录动作及其产生的事件
            tracker.update(action.event_for_successful_execution(prediction))  # 更新跟踪器

        structlogger.debug(  # 记录调试日志
            "processor.actions.log",
            action_name=action.name(),
            rasa_events=copy.deepcopy(events),
        )
        tracker.update_with_events(events, self.domain)  # 用事件更新跟踪器

    def _has_session_expired(self, tracker: DialogueStateTracker) -> bool:
        """确定`tracker`中的最新会话是否已过期。

        Args:
            tracker: 要检查的跟踪器。

        Returns:
            如果`tracker`中的会话已过期则返回`True`，否则返回`False`。
        """
        if not self.domain.session_config.are_sessions_enabled():  # 如果会话未启用
            # 如果会话被禁用，跟踪器永远不会过期
            return False

        user_uttered_event: Optional[UserUttered] = tracker.get_last_event_for(  # 获取最后一个用户话语事件
            UserUttered
        )

        if not user_uttered_event:  # 如果没有用户话语事件
            # 到目前为止没有用户事件，所以会话不应该被认为是过期的
            return False

        time_delta_in_seconds = time.time() - user_uttered_event.timestamp  # 计算时间差
        has_expired = (  # 检查是否过期
            time_delta_in_seconds / 60  # 转换为分钟
            > self.domain.session_config.session_expiration_time  # 大于会话过期时间
        )
        if has_expired:  # 如果已过期
            logger.debug(  # 记录调试日志
                f"The latest session for conversation ID '{tracker.sender_id}' has "
                f"expired."
            )

        return has_expired  # 返回是否过期

    async def save_tracker(self, tracker: DialogueStateTracker) -> None:
        """将给定的跟踪器保存到跟踪器存储。

        Args:
            tracker: 要保存的跟踪器。
        """
        await self.tracker_store.save(tracker)  # 保存跟踪器

    def _predict_next_with_tracker(
        self, tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        """从集成中收集预测并返回动作和预测。"""
        followup_action = tracker.followup_action  # 获取后续动作
        if followup_action:  # 如果有后续动作
            tracker.clear_followup_action()  # 清除后续动作
            if followup_action in self.domain.action_names_or_texts:  # 如果后续动作在域中
                prediction = PolicyPrediction.for_action_name(  # 为动作名称创建策略预测
                    self.domain, followup_action, FOLLOWUP_ACTION
                )
                return prediction  # 返回预测

            logger.error(  # 记录错误日志
                f"Trying to run unknown follow-up action '{followup_action}'. "
                "Instead of running that, Rasa Open Source will ignore the action "
                "and predict the next action."
            )

        target = self.model_metadata.core_target  # 获取核心目标
        if not target:  # 如果没有核心目标
            raise ValueError("Cannot predict next action if there is no core target.")  # 抛出异常

        results = self.graph_runner.run(  # 运行图
            inputs={PLACEHOLDER_TRACKER: tracker}, targets=[target]  # 输入跟踪器，目标核心
        )
        policy_prediction = results[target]  # 获取策略预测
        return policy_prediction  # 返回策略预测
