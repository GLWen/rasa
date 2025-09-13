# =============================================================================
# Rasa Core Forms 表单处理模块
# 本模块实现了 Rasa 中的表单（Form）功能，用于收集用户信息
# =============================================================================

# 导入标准库模块
import copy  # 深拷贝功能
from typing import Text, List, Optional, Union, Any, Dict, Set  # 类型注解
import itertools  # 迭代工具
import logging  # 日志记录
import structlog  # 结构化日志
import json  # JSON 处理

# 导入 Rasa 核心模块
from rasa.core.actions import action  # 动作处理模块
from rasa.core.actions.loops import LoopAction  # 循环动作基类
from rasa.core.channels import OutputChannel  # 输出通道接口
from rasa.shared.core.domain import Domain, KEY_SLOTS  # 领域模型和槽位键名
from rasa.shared.core.constants import SlotMappingType, SLOT_MAPPINGS, MAPPING_TYPE  # 槽位映射常量

# 导入动作相关类
from rasa.core.actions.action import ActionExecutionRejection, RemoteAction  # 动作执行异常和远程动作
from rasa.shared.core.constants import (
    ACTION_EXTRACT_SLOTS,  # 槽位提取动作名称
    ACTION_LISTEN_NAME,    # 监听动作名称
    REQUESTED_SLOT,        # 请求槽位键名
)
from rasa.shared.constants import UTTER_PREFIX  # 话语前缀常量

# 导入事件相关类
from rasa.shared.core.events import (
    Event,                    # 事件基类
    SlotSet,                 # 槽位设置事件
    ActionExecuted,          # 动作执行事件
    ActiveLoop,              # 活跃循环事件
    ActionExecutionRejected, # 动作执行拒绝事件
    Restarted,               # 重启事件
)

# 导入其他核心模块
from rasa.core.nlg import NaturalLanguageGenerator  # 自然语言生成器
from rasa.shared.core.slot_mappings import SlotMapping  # 槽位映射类
from rasa.shared.core.slots import ListSlot  # 列表槽位类型
from rasa.shared.core.trackers import DialogueStateTracker  # 对话状态跟踪器
from rasa.utils.endpoints import EndpointConfig  # 端点配置

# 初始化日志记录器
logger = logging.getLogger(__name__)  # 标准日志记录器
structlogger = structlog.get_logger()  # 结构化日志记录器


class FormAction(LoopAction):
    """表单动作类，实现并执行表单逻辑。
    
    表单动作是 Rasa 中用于收集用户信息的重要组件，它继承自 LoopAction，
    可以循环执行直到收集到所有必需的信息。
    """

    def __init__(
        self, form_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> None:
        """创建表单动作实例。

        Args:
            form_name: 表单名称，用于标识特定的表单
            action_endpoint: 执行自定义动作的端点配置
        """
        self._form_name = form_name  # 存储表单名称
        self.action_endpoint = action_endpoint  # 存储动作端点配置
        # 创建唯一实体映射需要领域信息，我们在初始化时没有
        # 将在第一次调用时创建
        self._unique_entity_mappings: Set[Text] = set()  # 唯一实体映射集合
        self._have_unique_entity_mappings_been_initialized = False  # 唯一实体映射是否已初始化标志

    def name(self) -> Text:
        """返回表单名称。
        
        Returns:
            表单的名称字符串
        """
        return self._form_name

    def required_slots(self, domain: Domain) -> List[Text]:
        """获取表单需要填充的必需槽位列表。

        Args:
            domain: 领域模型，包含槽位定义信息
            
        Returns:
            必需槽位名称的列表
        """
        return domain.required_slots_for_form(self.name())

    def from_entity(
        self,
        entity: Text,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Dict[Text, Any]:
        """创建从实体提取槽位值的映射字典。

        从以下条件提取槽位值：
        - 提取的实体
        - 条件限制：
            - intent: 如果指定，则用户意图必须匹配
            - not_intent: 如果指定，则用户意图不能匹配（用户意图不应该是这个意图）
            - role: 如果指定，则实体角色必须匹配
            - group: 如果指定，则实体组必须匹配

        Args:
            entity: 实体类型名称
            intent: 允许的意图列表（可选）
            not_intent: 不允许的意图列表（可选）
            role: 实体角色（可选）
            group: 实体组（可选）

        Returns:
            槽位映射字典
        """
        # 将意图参数转换为列表格式
        intent, not_intent = (
            SlotMapping.to_list(intent),      # 转换允许的意图为列表
            SlotMapping.to_list(not_intent),  # 转换不允许的意图为列表
        )

        # 返回槽位映射字典
        return {
            "type": str(SlotMappingType.FROM_ENTITY),  # 映射类型：从实体
            "entity": entity,                          # 实体名称
            "intent": intent,                          # 允许的意图列表
            "not_intent": not_intent,                  # 不允许的意图列表
            "role": role,                              # 实体角色
            "group": group,                            # 实体组
        }

    def get_mappings_for_slot(
        self, slot_to_fill: Text, domain: Domain
    ) -> List[Dict[Text, Any]]:
        """获取指定槽位的映射配置。

        如果映射为空，则将请求的槽位映射到同名的实体

        Args:
            slot_to_fill: 需要填充的槽位名称
            domain: 领域模型

        Returns:
            槽位映射配置列表

        Raises:
            TypeError: 当提供的槽位映射不兼容时抛出异常
        """
        # 从领域模型中获取槽位定义
        domain_slots = domain.as_dict().get(KEY_SLOTS, {})
        # 获取指定槽位的映射配置
        requested_slot_mappings = domain_slots.get(slot_to_fill, {}).get("mappings", [])

        # 检查提供的槽位映射是否有效
        for requested_slot_mapping in requested_slot_mappings:
            if (
                not isinstance(requested_slot_mapping, dict)  # 不是字典类型
                or requested_slot_mapping.get("type") is None  # 缺少类型字段
            ):
                raise TypeError("Provided incompatible slot mapping")  # 抛出类型错误异常

        return requested_slot_mappings

    def _create_unique_entity_mappings(self, domain: Domain) -> Set[Text]:
        """查找唯一设置槽位的 `from_entity` 类型映射。

        例如在以下表单中：
        some_form:
          departure_city:
            - type: from_entity
              entity: city
              role: from
            - type: from_entity
              entity: city
          arrival_city:
            - type: from_entity
              entity: city
              role: to
            - type: from_entity
              entity: city

        具有 `from` 角色的 `city` 实体唯一设置 `departure_city` 槽位，
        具有 `to` 角色的 `city` 实体唯一设置 `arrival_city` 槽位，
        因此对应的映射是唯一的。
        但是没有角色的 `city` 实体可以填充 `departure_city` 和 `arrival_city`，
        因此对应的映射不是唯一的。

        Args:
            domain: 领域模型

        Returns:
            唯一 `from_entity` 类型映射的 JSON 字符串集合
        """
        unique_entity_slot_mappings: Set[Text] = set()      # 唯一实体槽位映射集合
        duplicate_entity_slot_mappings: Set[Text] = set()   # 重复实体槽位映射集合
        domain_slots = domain.as_dict().get(KEY_SLOTS, {})  # 获取领域槽位定义
        
        # 遍历表单的所有必需槽位
        for slot in domain.required_slots_for_form(self.name()):
            # 遍历每个槽位的映射配置
            for slot_mapping in domain_slots.get(slot, {}).get(SLOT_MAPPINGS, []):
                # 只处理 from_entity 类型的映射
                if slot_mapping.get(MAPPING_TYPE) == str(SlotMappingType.FROM_ENTITY):
                    # 将映射转换为排序的 JSON 字符串
                    mapping_as_string = json.dumps(slot_mapping, sort_keys=True)
                    
                    # 检查映射是否已存在于唯一映射中
                    if mapping_as_string in unique_entity_slot_mappings:
                        # 如果已存在，则从唯一映射中移除，添加到重复映射中
                        unique_entity_slot_mappings.remove(mapping_as_string)
                        duplicate_entity_slot_mappings.add(mapping_as_string)
                    elif mapping_as_string not in duplicate_entity_slot_mappings:
                        # 如果不在重复映射中，则添加到唯一映射中
                        unique_entity_slot_mappings.add(mapping_as_string)

        return unique_entity_slot_mappings

    def entity_mapping_is_unique(
        self, slot_mapping: Dict[Text, Any], domain: Domain
    ) -> bool:
        """验证 from_entity 映射是否唯一。
        
        Args:
            slot_mapping: 要检查的槽位映射字典
            domain: 领域模型
            
        Returns:
            如果映射唯一则返回 True，否则返回 False
        """
        if not self._have_unique_entity_mappings_been_initialized:
            # 在第一次调用时创建唯一实体映射
            self._unique_entity_mappings = self._create_unique_entity_mappings(domain)
            self._have_unique_entity_mappings_been_initialized = True

        # 将映射转换为排序的 JSON 字符串进行比较
        mapping_as_string = json.dumps(slot_mapping, sort_keys=True)
        return mapping_as_string in self._unique_entity_mappings

    @staticmethod
    def get_entity_value_for_slot(
        name: Text,
        tracker: "DialogueStateTracker",
        slot_to_be_filled: Text,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Any:
        """为指定名称和可选角色、组提取实体值。

        Args:
            name: 感兴趣的实体类型（名称）
            tracker: 对话状态跟踪器
            slot_to_be_filled: 应该被此实体填充的槽位
            role: 感兴趣的实体角色（可选）
            group: 感兴趣的实体组（可选）

        Returns:
            实体值
        """
        # 使用列表来覆盖列表槽位类型的情况
        value = list(
            tracker.get_latest_entity_values(name, entity_group=group, entity_role=role)
        )

        # 如果目标槽位是列表类型，直接返回值列表
        if isinstance(tracker.slots.get(slot_to_be_filled), ListSlot):
            return value

        # 如果没有找到值，返回 None
        if len(value) == 0:
            return None

        # 如果只有一个值，返回单个值
        if len(value) == 1:
            return value[0]

        # 否则返回值列表
        return value

    def get_slot_to_fill(self, tracker: "DialogueStateTracker") -> Optional[str]:
        """获取下一个应该填充的槽位名称。

        当切换到另一个表单时，请求的槽位设置仍然来自
        前一个表单，必须被忽略。

        Args:
            tracker: 对话状态跟踪器

        Returns:
            槽位名称或 `None`
        """
        return (
            tracker.get_slot(REQUESTED_SLOT)  # 获取请求的槽位
            if tracker.active_loop_name == self.name()  # 如果当前活跃循环是此表单
            else None  # 否则返回 None
        )

    async def validate_slots(
        self,
        slot_candidates: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Union[SlotSet, Event]]:
        """验证提取的槽位。

        如果有可用的自定义动作来验证槽位，我们调用它来验证。
        否则不进行验证。

        Args:
            slot_candidates: 提取的槽位，是填充表单所需槽位的候选值
            tracker: 当前对话跟踪器
            domain: 当前模型领域
            output_channel: 可用于向用户发送消息的输出通道
            nlg: 用于响应生成的自然语言生成器

        Returns:
            验证事件，包括潜在机器人消息和已验证槽位的 `SlotSet` 事件，
            如果自定义表单验证动作存在于领域动作中。
            否则返回空列表，因为提取的槽位在跟踪器中已经有对应的 `SlotSet` 事件。
        """
        # 记录调试信息
        structlogger.debug(
            "forms.slots.validate", slot_candidates=copy.deepcopy(slot_candidates)
        )
        
        # 为每个槽位候选值创建 SlotSet 事件
        events: List[Union[SlotSet, Event]] = [
            SlotSet(slot_name, value) for slot_name, value in slot_candidates.items()
        ]

        # 构建验证动作名称
        validate_name = f"validate_{self.name()}"

        # 如果领域中没有自定义验证动作，返回空列表
        if validate_name not in domain.action_names_or_texts:
            return []

        # 创建临时跟踪器，只包含自上次用户话语以来添加的 SlotSet 事件
        _tracker = self._temporary_tracker(tracker, events, domain)

        # 创建远程验证动作并执行
        _action = RemoteAction(validate_name, self.action_endpoint)
        validate_events = await _action.run(output_channel, nlg, _tracker, domain)

        # 只返回自定义表单验证动作验证的 SlotSet 事件
        # 以避免为已经有效的槽位添加重复的 SlotSet 事件
        return validate_events

    def _temporary_tracker(
        self,
        current_tracker: DialogueStateTracker,
        additional_events: List[Event],
        domain: Domain,
    ) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            current_tracker.sender_id,
            current_tracker.events_after_latest_restart()
            # Insert SlotSet event to make sure REQUESTED_SLOT belongs to active form.
            + [SlotSet(REQUESTED_SLOT, self.get_slot_to_fill(current_tracker))]
            # Insert form execution event so that it's clearly distinguishable which
            # events were newly added.
            + [ActionExecuted(self.name())] + additional_events,
            slots=domain.slots,
        )

    def _user_rejected_manually(self, validation_events: List[Event]) -> bool:
        """Checks if user rejected the form execution during a slot_validation.

        Args:
            validation_events: Events returned by the custom slot_validation action

        Returns:
            True if the validation_events include an ActionExecutionRejected event,
            else False.
        """
        return any(
            isinstance(event, ActionExecutionRejected) for event in validation_events
        )

    @staticmethod
    def _get_events_since_last_user_uttered(
        tracker: "DialogueStateTracker",
    ) -> List[SlotSet]:
        # TODO: Better way to get this latest_message index is through an instance
        # variable, eg. tracker.latest_message_index
        index_from_end = next(
            (
                i
                for i, event in enumerate(reversed(tracker.events))
                if event == Restarted() or event == tracker.latest_message
            ),
            len(tracker.events) - 1,
        )
        index = len(tracker.events) - index_from_end - 1
        events_since_last_user_uttered = [
            event
            for event in itertools.islice(tracker.events, index, None)
            if isinstance(event, SlotSet)
        ]

        return events_since_last_user_uttered

    def _update_slot_values(
        self,
        event: SlotSet,
        tracker: "DialogueStateTracker",
        domain: Domain,
        slot_values: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        slot_values[event.key] = event.value

        return slot_values

    def _add_dynamic_slots_requested_by_dynamic_forms(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Set[Text]:
        required_slots = set(self.required_slots(domain))
        requested_slot = self.get_slot_to_fill(tracker)

        if requested_slot:
            required_slots.add(requested_slot)

        return required_slots

    def _get_slot_extractions(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Dict[Text, Any]:
        events_since_last_user_uttered = FormAction._get_events_since_last_user_uttered(
            tracker
        )
        slot_values: Dict[Text, Any] = {}

        required_slots = self._add_dynamic_slots_requested_by_dynamic_forms(
            tracker, domain
        )

        for event in events_since_last_user_uttered:
            if event.key not in required_slots:
                continue

            slot_values = self._update_slot_values(event, tracker, domain, slot_values)

        return slot_values

    async def validate(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Union[SlotSet, Event]]:
        """提取并验证请求槽位和其他槽位的值。

        Args:
            tracker: 对话状态跟踪器
            domain: 领域模型
            output_channel: 输出通道
            nlg: 自然语言生成器

        Returns:
            自定义表单验证动作创建的新验证事件

        Raises:
            ActionExecutionRejection: 如果没有提取到任何内容，则抛出异常以拒绝执行表单动作

        Note:
            子类可以重写此方法以添加自定义验证和拒绝逻辑。
        """
        # 获取提取的槽位值
        extracted_slot_values = self._get_slot_extractions(tracker, domain)

        # 验证槽位
        validation_events = await self.validate_slots(
            extracted_slot_values, tracker, domain, output_channel, nlg
        )

        # 检查是否有槽位被验证（忽略 REQUESTED_SLOT）
        some_slots_were_validated = any(
            isinstance(event, SlotSet) and not event.key == REQUESTED_SLOT
            for event in validation_events
            # 忽略 `REQUESTED_SLOT` 的 `SlotSet`，因为这不是用户需要填充的槽位
        )

        # 提取请求的槽位
        slot_to_fill = self.get_slot_to_fill(tracker)

        # 如果请求了槽位但没有提取到值，且没有槽位被验证，且用户没有手动拒绝
        if (
            slot_to_fill
            and not extracted_slot_values
            and not some_slots_were_validated
            and not self._user_rejected_manually(validation_events)
        ):
            # 拒绝执行表单动作
            # 如果请求了某个槽位但没有提取到任何内容
            # 这将允许其他策略预测另一个动作
            #
            # 如果用户手动拒绝，不要在这里抛出异常，以允许填充
            # 请求槽位之外的其他槽位
            #
            raise ActionExecutionRejection(
                self.name(),
                f"Failed to extract slot {slot_to_fill} with action {self.name()}",
            )
        return validation_events

    async def request_next_slot(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        events_so_far: List[Event],
    ) -> List[Union[SlotSet, Event]]:
        """请求下一个槽位，如果需要则生成响应，否则返回 `None`。
        
        Args:
            tracker: 对话状态跟踪器
            domain: 领域模型
            output_channel: 输出通道
            nlg: 自然语言生成器
            events_so_far: 到目前为止的事件列表
            
        Returns:
            请求槽位的事件列表
        """
        request_slot_events: List[Event] = []  # 请求槽位事件列表

        # 如果表单已完成，返回空槽位请求
        if await self.is_done(output_channel, nlg, tracker, domain, events_so_far):
            # 自定义槽位验证动作决定提前停止表单
            return [SlotSet(REQUESTED_SLOT, None)]

        # 查找当前要请求的槽位
        slot_to_request = next(
            (
                event.value
                for event in events_so_far
                if isinstance(event, SlotSet) and event.key == REQUESTED_SLOT
            ),
            None,
        )

        # 创建临时跟踪器
        temp_tracker = self._temporary_tracker(tracker, events_so_far, domain)

        # 如果没有指定槽位，查找下一个需要请求的槽位
        if not slot_to_request:
            slot_to_request = self._find_next_slot_to_request(temp_tracker, domain)
            request_slot_events.append(SlotSet(REQUESTED_SLOT, slot_to_request))

        # 如果有槽位需要请求，生成机器人消息
        if slot_to_request:
            bot_message_events = await self._ask_for_slot(
                domain, nlg, output_channel, slot_to_request, temp_tracker
            )
            return request_slot_events + bot_message_events

        # 没有更多需要填充的必需槽位
        return [SlotSet(REQUESTED_SLOT, None)]

    def _find_next_slot_to_request(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:
        """查找下一个需要请求的槽位。
        
        Args:
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            下一个需要请求的槽位名称，如果没有则返回 None
        """
        return next(
            (
                slot
                for slot in self.required_slots(domain)  # 遍历所有必需槽位
                if self._should_request_slot(tracker, slot)  # 检查是否应该请求该槽位
            ),
            None,  # 如果没有找到，返回 None
        )

    def _name_of_utterance(self, domain: Domain, slot_name: Text) -> Optional[Text]:
        """查找用于询问指定槽位的话语动作名称。
        
        Args:
            domain: 领域模型
            slot_name: 槽位名称
            
        Returns:
            找到的话语动作名称，如果没有找到则返回 None
        """
        # 定义搜索路径，按优先级排序
        search_path = [
            f"action_ask_{self._form_name}_{slot_name}",      # 表单特定的动作
            f"{UTTER_PREFIX}ask_{self._form_name}_{slot_name}", # 表单特定的话语
            f"action_ask_{slot_name}",                         # 通用动作
            f"{UTTER_PREFIX}ask_{slot_name}",                  # 通用话语
        ]

        # 查找第一个存在于领域中的动作
        found_actions = (
            action_name
            for action_name in search_path
            if action_name in domain.action_names_or_texts
        )

        return next(found_actions, None)

    async def _ask_for_slot(
        self,
        domain: Domain,
        nlg: NaturalLanguageGenerator,
        output_channel: OutputChannel,
        slot_name: Text,
        tracker: DialogueStateTracker,
    ) -> List[Event]:
        """询问指定槽位的信息。
        
        Args:
            domain: 领域模型
            nlg: 自然语言生成器
            output_channel: 输出通道
            slot_name: 槽位名称
            tracker: 对话状态跟踪器
            
        Returns:
            询问槽位的事件列表
        """
        logger.debug(f"Request next slot '{slot_name}'")  # 记录调试信息

        # 查找用于询问槽位的话语动作名称
        action_name_to_ask_for_next_slot = self._name_of_utterance(domain, slot_name)
        if not action_name_to_ask_for_next_slot:
            # 使用调试日志，因为用户可能作为自定义动作的一部分询问
            logger.debug(
                f"There was no action found to ask for slot '{slot_name}' "
                f"name to be filled."
            )
            return []

        # 创建动作实例并执行
        action_to_ask_for_next_slot = action.action_for_name_or_text(
            action_name_to_ask_for_next_slot, domain, self.action_endpoint
        )
        return await action_to_ask_for_next_slot.run(
            output_channel, nlg, tracker, domain
        )

    async def _validate_if_required(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """如果需要则返回 `self.validate(...)` 的事件列表。

        在以下情况下需要验证：
           - 表单处于活跃状态
           - 表单在 `action_listen` 之后被调用
           - 表单验证未被取消

        Args:
            tracker: 对话状态跟踪器
            domain: 领域模型
            output_channel: 输出通道
            nlg: 自然语言生成器

        Returns:
            验证事件列表
        """
        # 没有活跃循环意味着在激活期间被调用
        needs_validation = not tracker.active_loop or (
            tracker.latest_action_name == ACTION_LISTEN_NAME  # 最新动作是监听
            and not tracker.is_active_loop_interrupted  # 活跃循环未被中断
        )

        if needs_validation:
            # 记录验证需求
            structlogger.debug(
                "forms.validation.required",
                tracker_latest_message=copy.deepcopy(tracker.latest_message),
            )
            return await self.validate(tracker, domain, output_channel, nlg)
        else:
            # 需要确定要请求哪些槽位，尽管没有槽位需要实际验证
            # 这在从不愉快路径返回表单后发生
            return await self.validate_slots({}, tracker, domain, output_channel, nlg)

    @staticmethod
    def _should_request_slot(tracker: "DialogueStateTracker", slot_name: Text) -> bool:
        """检查表单动作是否应该请求给定的槽位。
        
        Args:
            tracker: 对话状态跟踪器
            slot_name: 槽位名称
            
        Returns:
            如果应该请求槽位则返回 True，否则返回 False
        """
        return tracker.get_slot(slot_name) is None  # 槽位值为空时才需要请求

    async def activate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """如果表单是第一次被调用，则激活表单。

        如果激活，运行 action_extract_slots 以使用触发意图的映射条件填充槽位。
        验证任何可以填充的必需槽位，并返回这些预填充槽位的提取和验证的 `SlotSet` 事件。

        Args:
            output_channel: 可用于向用户发送消息的输出通道
            nlg: 用于响应生成的自然语言生成器
            tracker: 用户的当前对话跟踪器
            domain: 当前模型领域

        Returns:
            来自激活的事件
        """
        logger.debug(f"Activated the form '{self.name()}'.")  # 记录表单激活
        # 收集激活前填充的必需槽位值
        prefilled_slots = {}

        # 创建槽位提取动作
        action_extract_slots = action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, domain, self.action_endpoint
        )

        logger.debug(
            f"Executing default action '{ACTION_EXTRACT_SLOTS}' at form activation."
        )

        # 执行槽位提取动作
        extraction_events = await action_extract_slots.run(
            output_channel, nlg, tracker, domain
        )

        # 记录提取事件
        events_as_str = "\n".join(str(e) for e in extraction_events)
        logger.debug(
            f"The execution of '{ACTION_EXTRACT_SLOTS}' resulted in "
            f"these events: {events_as_str}."
        )

        # 更新跟踪器
        tracker.update_with_events(extraction_events, domain)

        # 收集预填充的槽位
        for slot_name in self.required_slots(domain):
            if not self._should_request_slot(tracker, slot_name):
                prefilled_slots[slot_name] = tracker.get_slot(slot_name)

        if not prefilled_slots:
            logger.debug("No pre-filled required slots to validate.")
        else:
            structlogger.debug(
                "forms.validate.prefilled_slots",
                prefilled_slots=copy.deepcopy(prefilled_slots),
            )

        # 构建验证动作名称
        validate_name = f"validate_{self.name()}"

        # 如果没有验证动作，返回提取事件
        if validate_name not in domain.action_names_or_texts:
            logger.debug(
                f"There is no validation action '{validate_name}' "
                f"to execute at form activation."
            )
            return [event for event in extraction_events if isinstance(event, SlotSet)]

        logger.debug(
            f"Executing validation action '{validate_name}' at form activation."
        )

        # 执行验证动作
        validated_events = await self.validate_slots(
            prefilled_slots, tracker, domain, output_channel, nlg
        )

        # 获取已验证的槽位名称
        validated_slot_names = [
            event.key for event in validated_events if isinstance(event, SlotSet)
        ]

        # 返回验证事件和未验证的提取事件
        return validated_events + [
            event
            for event in extraction_events
            if isinstance(event, SlotSet) and event.key not in validated_slot_names
        ]

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """在激活后执行表单循环。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            执行表单循环的事件列表
        """
        events: List[Event] = []  # 事件列表
        """
        当槽位在表单激活时已经被验证时，不需要调用验证。
        events_so_far:
            - 当槽位未被验证时为空
            - 当已经验证时有 SlotSet 对象
            - 当事件未被验证时有 ActiveLoop 对象
        因此过滤事件以移除在表单激活时添加的 ActiveLoop 对象
        """
        # 过滤掉 ActiveLoop 事件
        filtered_events = [
            event for event in events_so_far if not isinstance(event, ActiveLoop)
        ]
        
        # 如果没有过滤的事件，进行验证
        if not filtered_events:
            events = await self._validate_if_required(
                tracker, domain, output_channel, nlg
            )

        # 如果用户没有手动拒绝，请求下一个槽位
        if not self._user_rejected_manually(events):
            events += await self.request_next_slot(
                tracker, domain, output_channel, nlg, events_so_far + events
            )

        return events

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        """检查循环是否可以终止。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            如果循环可以终止则返回 True，否则返回 False
        """
        # 如果有动作执行被拒绝的事件，不能终止
        if any(isinstance(event, ActionExecutionRejected) for event in events_so_far):
            return False

        # 自定义验证动作可以通过将请求的槽位设置为 `None` 或设置 `ActiveLoop(None)` 来
        # 决定提前终止循环。
        # 我们显式检查每个可能的终止事件的最后出现，而不是执行 `return event in events_so_far`
        # 以便可以覆盖之前返回的终止事件。
        return next(
            (
                event
                for event in reversed(events_so_far)  # 从后往前查找
                if isinstance(event, SlotSet) and event.key == REQUESTED_SLOT
            ),
            None,
        ) == SlotSet(REQUESTED_SLOT, None) or next(  # 检查请求槽位是否为 None
            (
                event
                for event in reversed(events_so_far)  # 从后往前查找
                if isinstance(event, ActiveLoop)
            ),
            None,
        ) == ActiveLoop(None)  # 检查活跃循环是否为 None

    async def deactivate(self, *args: Any, **kwargs: Any) -> List[Event]:
        """停用表单。
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            停用表单的事件列表（空列表）
        """
        logger.debug(f"Deactivating the form '{self.name()}'")  # 记录表单停用
        return []  # 返回空事件列表

    async def _activate_loop(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """激活循环。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            激活循环的事件列表
        """
        # 获取默认激活事件
        events = self._default_activation_events()

        # 创建临时跟踪器并更新事件
        temp_tracker = tracker.copy()
        temp_tracker.update_with_events(events, domain)
        
        # 激活表单并添加事件
        events += await self.activate(output_channel, nlg, temp_tracker, domain)

        return events
