import copy
import dataclasses
import itertools
import logging
import os
import time
from collections import deque
from enum import Enum
from typing import (
    Dict,
    Text,
    Any,
    Optional,
    Iterator,
    Generator,
    Type,
    TypeVar,
    List,
    Deque,
    Iterable,
    Union,
    FrozenSet,
    Tuple,
    TYPE_CHECKING,
    cast,
)

# =============================================================================
# 对话状态跟踪器模块 - 维护对话状态和事件历史
# =============================================================================
# 此模块包含 DialogueStateTracker 类，用于跟踪对话状态、
# 管理事件历史、处理槽位状态和循环管理。
# 跟踪器是 Rasa 对话系统的核心组件，负责维护对话的完整状态。

# =============================================================================
# 导入相关常量和工具模块
# =============================================================================
import rasa.shared.utils.io
from rasa.shared.constants import ASSISTANT_ID_KEY, DEFAULT_SENDER_ID  # 助手ID键和默认发送者ID
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,    # 实体值属性
    ENTITY_ATTRIBUTE_TYPE,     # 实体类型属性
    ENTITY_ATTRIBUTE_GROUP,    # 实体组属性
    ENTITY_ATTRIBUTE_ROLE,     # 实体角色属性
    ACTION_TEXT,               # 动作文本
    ACTION_NAME,               # 动作名称
    ENTITIES,                  # 实体
    METADATA_MODEL_ID,         # 元数据模型ID
)
from rasa.shared.core import events  # 事件模块
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,        # 监听动作名称
    LOOP_NAME,                 # 循环名称
    SHOULD_NOT_BE_SET,         # 不应设置
    PREVIOUS_ACTION,           # 前一动作
    ACTIVE_LOOP,               # 活动循环
    ACTION_SESSION_START_NAME, # 会话开始动作名称
    FOLLOWUP_ACTION,           # 后续动作
)
from rasa.shared.core.conversation import Dialogue  # 对话类
from rasa.shared.core.events import (
    UserUttered,                              # 用户话语事件
    ActionExecuted,                           # 动作执行事件
    Event,                                    # 事件基类
    Restarted,                                # 重启事件
    ActionReverted,                           # 动作撤销事件
    UserUtteranceReverted,                    # 用户话语撤销事件
    BotUttered,                               # 机器人话语事件
    ActiveLoop,                               # 活动循环事件
    SessionStarted,                           # 会话开始事件
    ActionExecutionRejected,                  # 动作执行拒绝事件
    DefinePrevUserUtteredFeaturization,       # 定义前一用户话语特征化事件
)
from rasa.shared.core.domain import Domain, State  # 域和状态
from rasa.shared.core.slots import AnySlot, Slot   # 槽位类

# =============================================================================
# 类型检查和数据结构定义
# =============================================================================
if TYPE_CHECKING:
    from rasa.shared.core.events import NLUPredictionData  # NLU预测数据
    from rasa.shared.core.training_data.structures import Story  # 故事结构
    from rasa.shared.core.training_data.story_writer.story_writer import StoryWriter  # 故事写入器

    EventTypeAlias = TypeVar("EventTypeAlias", bound=Event)  # 事件类型别名


@dataclasses.dataclass
class TrackerActiveLoop:
    """跟踪器活动循环的数据类。
    
    用于存储当前活动循环的状态信息。
    """

    name: Optional[Text]              # 循环名称
    is_interrupted: bool              # 是否被中断
    rejected: bool                    # 是否被拒绝
    trigger_message: Optional[Dict]   # 触发消息


logger = logging.getLogger(__name__)

# 与 State 相同，但将 Dict[...] 替换为 FrozenSet[Tuple[...]]
# 用于创建可哈希的状态表示
FrozenState = FrozenSet[Tuple[Text, FrozenSet[Tuple[Text, Tuple[Union[float, Text]]]]]]


# =============================================================================
# 事件详细程度枚举
# =============================================================================
class EventVerbosity(Enum):
    """过滤跟踪器转储中包含哪些事件的枚举。"""

    # 不包含任何事件
    NONE = 1

    # 包含所有对跟踪器状态有贡献的事件
    # 这些是重建跟踪器状态所需的全部事件
    APPLIED = 2

    # 包含更多事件，在这种情况下包括
    # 最近重启事件之后的所有内容。这还包括
    # 被撤销的话语和动作。
    AFTER_RESTART = 3

    # 包含每个记录的事件
    ALL = 4


# =============================================================================
# 任意槽位字典类
# =============================================================================
class AnySlotDict(dict):
    """一个槽位字典，通过按需创建槽位来假装每个槽位都存在。

    这只使用通用槽位类型！这意味着某些功能不会工作，
    例如正确地对槽位进行特征化。
    """

    def __missing__(self, key: Text) -> Slot:
        """当访问不存在的键时，创建一个新的 AnySlot。
        
        Args:
            key: 槽位名称
            
        Returns:
            新创建的 AnySlot 实例
        """
        value = self[key] = AnySlot(key, mappings=[])
        return value

    def __contains__(self, key: Any) -> bool:
        """总是返回 True，假装所有槽位都存在。
        
        Args:
            key: 要检查的键
            
        Returns:
            总是返回 True
        """
        return True


# =============================================================================
# 对话状态跟踪器类
# =============================================================================
class DialogueStateTracker:
    """维护对话的状态。

    max_event_history 字段只会给你这些最后的事件，
    它可以在 tracker_store 中设置。
    """

    @classmethod
    def from_dict(
        cls,
        sender_id: Text,
        events_as_dict: List[Dict[Text, Any]],
        slots: Optional[Iterable[Slot]] = None,
        max_event_history: Optional[int] = None,
    ) -> "DialogueStateTracker":
        """从转储创建跟踪器。

        转储应该是转储事件的数组。恢复跟踪器时，
        这些事件将被重放以重新创建状态。
        
        Args:
            sender_id: 发送者ID
            events_as_dict: 事件字典列表
            slots: 可选的槽位列表
            max_event_history: 最大事件历史数量
            
        Returns:
            创建的跟踪器实例
        """
        evts = events.deserialise_events(events_as_dict)

        return cls.from_events(sender_id, evts, slots, max_event_history)

    @classmethod
    def from_events(
        cls,
        sender_id: Text,
        evts: List[Event],
        slots: Optional[Iterable[Slot]] = None,
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
        domain: Optional[Domain] = None,
    ) -> "DialogueStateTracker":
        """从现有事件创建跟踪器。

        Args:
            sender_id: 对话的ID
            evts: 应应用于新跟踪器的现有事件
            slots: 可以设置的槽位
            max_event_history: 应存储的最大事件数量
            sender_source: 消息的文件源
            domain: 当前模型域

        Returns:
            根据给定事件更新状态的实例化跟踪器
        """
        tracker = cls(sender_id, slots, max_event_history, sender_source)

        for e in evts:
            tracker.update(e, domain)

        return tracker

    def __init__(
        self,
        sender_id: Text,
        slots: Optional[Iterable[Slot]],
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
        is_rule_tracker: bool = False,
    ) -> None:
        """初始化跟踪器。

        一组事件可以外部存储，我们将遍历所有事件
        来获取当前状态。跟踪器将表示我们在处理
        对话消息时捕获的所有信息。
        
        Args:
            sender_id: 发送者ID
            slots: 可选的槽位列表
            max_event_history: 最大事件历史数量
            sender_source: 发送者源
            is_rule_tracker: 是否为基于规则的跟踪器
        """
        # 要存储的最大事件数量
        self._max_event_history = max_event_history
        # 之前看到的事件列表
        self.events = self._create_events([])
        # 消息源的ID
        self.sender_id = sender_id
        # 在此域中可以填充的槽位
        if slots is not None:
            self.slots = {slot.name: copy.copy(slot) for slot in slots}
        else:
            self.slots = AnySlotDict()
        # 消息的文件源
        self.sender_source = sender_source
        # 跟踪器是否属于基于规则的数据
        self.is_rule_tracker = is_rule_tracker

        ###
        # 跟踪器的当前状态 - 必须通过处理所有事件
        # 来重新创建。这里只定义属性，值在 `reset()` 中设置
        ###
        # 如果跟踪器暂停，不应采取任何动作
        self._paused = False
        # 确定性地安排的下一个要执行的动作
        self.followup_action: Optional[Text] = ACTION_LISTEN_NAME
        self.latest_action: Optional[Dict[Text, Text]] = None
        # 存储用户发送的最新消息
        self.latest_message: Optional[UserUttered] = None
        self.latest_bot_utterance: Optional[BotUttered] = None
        self._reset()
        self.active_loop: Optional[TrackerActiveLoop] = None

        # 添加到所有事件的可选 model_id
        self.model_id: Optional[Text] = None
        self.assistant_id: Optional[Text] = None

    ###
    # Public tracker interface
    ###
    def current_state(
        self, event_verbosity: EventVerbosity = EventVerbosity.NONE
    ) -> Dict[Text, Any]:
        """返回当前跟踪器状态作为对象。
        
        Args:
            event_verbosity: 事件详细程度
            
        Returns:
            包含跟踪器状态的字典
        """
        events = self._events_for_verbosity(event_verbosity)
        events_as_dict = [e.as_dict() for e in events] if events is not None else None
        latest_event_time = None
        if len(self.events) > 0:
            latest_event_time = self.events[-1].timestamp

        return {
            "sender_id": self.sender_id,
            "slots": self.current_slot_values(),
            "latest_message": self._latest_message_data(),
            "latest_event_time": latest_event_time,
            FOLLOWUP_ACTION: self.followup_action,
            "paused": self.is_paused(),
            "events": events_as_dict,
            "latest_input_channel": self.get_latest_input_channel(),
            ACTIVE_LOOP: (
                dataclasses.asdict(self.active_loop) if self.active_loop else {}
            ),
            "latest_action": self.latest_action,
            "latest_action_name": self.latest_action_name,
        }

    def _events_for_verbosity(
        self, event_verbosity: EventVerbosity
    ) -> Optional[List[Event]]:
        if event_verbosity == EventVerbosity.ALL:
            return list(self.events)
        if event_verbosity == EventVerbosity.AFTER_RESTART:
            return self.events_after_latest_restart()
        if event_verbosity == EventVerbosity.APPLIED:
            return self.applied_events()

        return None

    def _latest_message_data(self) -> Optional["NLUPredictionData"]:
        if not self.latest_message:
            return None

        parse_data_with_nlu_state = self.latest_message.parse_data.copy()
        # Combine entities predicted by NLU with entities predicted by policies so that
        # users can access them together via `latest_message` (e.g. in custom actions)
        parse_data_with_nlu_state[ENTITIES] = self.latest_message.entities  # type: ignore[literal-required]  # noqa: E501

        return parse_data_with_nlu_state

    @staticmethod
    def freeze_current_state(state: State) -> FrozenState:
        """Convert State dict into a hashable format FrozenState.

        Args:
            state: The state which should be converted

        Return:
            hashable form of the state of type `FrozenState`
        """
        return frozenset(
            {
                key: frozenset(values.items())
                if isinstance(values, Dict)
                else frozenset(values)
                for key, values in state.items()
            }.items()
        )

    def past_states(
        self,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """Generates the past states of this tracker based on the history.

        Args:
            domain: The Domain.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states
        """
        return domain.states_for_tracker_history(
            self,
            omit_unset_slots=omit_unset_slots,
            ignore_rule_only_turns=ignore_rule_only_turns,
            rule_only_data=rule_only_data,
        )

    def change_loop_to(self, loop_name: Optional[Text]) -> None:
        """Set the currently active loop.

        Args:
            loop_name: The name of loop which should be marked as active.
        """
        if loop_name is not None:
            self.active_loop = TrackerActiveLoop(
                loop_name,
                False,
                False,
                self.latest_message.parse_data if self.latest_message else None,
            )
        else:
            self.active_loop = None

    def interrupt_loop(self, is_interrupted: bool) -> None:
        """Interrupt loop and mark that we entered an unhappy path in the conversation.

        Args:
            is_interrupted: `True` if the loop was run after an unhappy path.
        """
        if self.active_loop is not None:
            self.active_loop.is_interrupted = is_interrupted

    def reject_action(self, action_name: Text) -> None:
        """Notify active loop that it was rejected."""
        if self.active_loop is not None and action_name == self.active_loop_name:
            self.active_loop.rejected = True

    def set_latest_action(self, action: Dict[Text, Text]) -> None:
        """Sets latest action name or text.

        Resets loop validation and rejection parameters.

        Args:
            action: Serialized action event.
        """
        self.latest_action = action
        if self.active_loop is not None and self.active_loop_name:
            # reset form validation if some loop is active
            self.active_loop.is_interrupted = False

        if (
            self.active_loop is not None
            and action.get(ACTION_NAME) == self.active_loop_name
        ):
            # reset loop rejection if it was predicted again
            self.active_loop.rejected = False

    def current_slot_values(self) -> Dict[Text, Any]:
        """Return the currently set values of the slots."""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key: Text) -> Optional[Any]:
        """Retrieves the value of a slot."""
        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info(f"Tried to access non existent slot '{key}'")
            return None

    def get_latest_entity_values(
        self,
        entity_type: Text,
        entity_role: Optional[Text] = None,
        entity_group: Optional[Text] = None,
    ) -> Iterator[Text]:
        """Get entity values found for the passed entity type and optional role and
        group in latest message.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values(`"`my_entity_name`"`), None)`.
        If no entity is found `None` is the default result.

        Args:
            entity_type: the entity type of interest
            entity_role: optional entity role of interest
            entity_group: optional entity group of interest

        Returns:
            Entity values.
        """
        if self.latest_message is None:
            return iter([])

        return (
            cast(Text, x[ENTITY_ATTRIBUTE_VALUE])
            for x in self.latest_message.entities
            if x.get(ENTITY_ATTRIBUTE_TYPE) == entity_type
            and x.get(ENTITY_ATTRIBUTE_GROUP) == entity_group
            and x.get(ENTITY_ATTRIBUTE_ROLE) == entity_role
        )

    def get_latest_input_channel(self) -> Optional[Text]:
        """Get the name of the input_channel of the latest UserUttered event."""
        for e in reversed(self.events):
            if isinstance(e, UserUttered):
                return e.input_channel
        return None

    def is_paused(self) -> bool:
        """State whether the tracker is currently paused."""
        return self._paused

    def idx_after_latest_restart(self) -> int:
        """Return the idx of the most recent restart in the list of events.

        If the conversation has not been restarted, ``0`` is returned.
        """
        for i, event in enumerate(reversed(self.events)):
            if isinstance(event, Restarted):
                return len(self.events) - i

        return 0

    def events_after_latest_restart(self) -> List[Event]:
        """Return a list of events after the most recent restart."""
        return list(self.events)[self.idx_after_latest_restart() :]

    def init_copy(self) -> "DialogueStateTracker":
        """Creates a new state tracker with the same initial values."""
        return DialogueStateTracker(
            self.sender_id or DEFAULT_SENDER_ID,
            self.slots.values(),
            self._max_event_history,
            is_rule_tracker=self.is_rule_tracker,
        )

    def generate_all_prior_trackers(
        self,
    ) -> Generator[Tuple["DialogueStateTracker", bool], None, None]:
        """Returns a generator of the previous trackers of this tracker.

        Returns:
            The tuple with the tracker before each action,
            and the boolean flag representing whether this action should be hidden
            in the dialogue history created for ML-based policies.
        """
        tracker = self.init_copy()

        for event in self.applied_events():

            if isinstance(event, ActionExecuted):
                yield tracker, event.hide_rule_turn

            tracker.update(event)

        yield tracker, False

    def applied_events(self) -> List[Event]:
        """Returns all actions that should be applied - w/o reverted events.

        Returns:
            The events applied to the tracker.
        """
        loop_names = [
            event.name
            for event in self.events
            if isinstance(event, ActiveLoop) and event.name
        ]

        applied_events: List[Event] = []

        for event in self.events:
            if isinstance(event, (Restarted, SessionStarted)):
                applied_events = []
            elif isinstance(event, ActionReverted):
                self._undo_till_previous(ActionExecuted, applied_events)
            elif isinstance(event, UserUtteranceReverted):
                # Seeing a user uttered event automatically implies there was
                # a listen event right before it, so we'll first rewind the
                # user utterance, then get the action right before it (also removes
                # the `action_listen` action right before it).
                self._undo_till_previous(UserUttered, applied_events)
                self._undo_till_previous(ActionExecuted, applied_events)
            elif (
                isinstance(event, ActionExecuted)
                and event.action_name in loop_names
                and not self._first_loop_execution_or_unhappy_path(
                    event.action_name, applied_events
                )
            ):
                self._undo_till_previous_loop_execution(
                    event.action_name, applied_events
                )
            else:
                applied_events.append(event)

        return applied_events

    @staticmethod
    def _undo_till_previous(event_type: Type[Event], done_events: List[Event]) -> None:
        """Removes events from `done_events`.

        Removes events from `done_events` until the first occurrence `event_type`
        is found which is also removed.
        """
        # list gets modified - hence we need to copy events!
        for e in reversed(done_events[:]):
            del done_events[-1]
            if isinstance(e, event_type):
                break

    def _first_loop_execution_or_unhappy_path(
        self, loop_action_name: Text, applied_events: List[Event]
    ) -> bool:
        next_action: Optional[Text] = None

        for event in reversed(applied_events):
            # Stop looking for a previous loop execution if there is a loop deactivation
            # event because it means that the current loop is running for the first
            # time and previous loop events belong to different loops.
            if isinstance(event, ActiveLoop) and event.name is None:
                return True

            if self._is_within_unhappy_path(loop_action_name, event, next_action):
                return True

            if isinstance(event, ActionExecuted):
                # We found a previous execution of the loop and we are not within an
                # unhappy path.
                if event.action_name == loop_action_name:
                    return False

                # Remember the action as we need that to check whether we might be
                # within an unhappy path.
                next_action = event.action_name

        return True

    @staticmethod
    def _is_within_unhappy_path(
        loop_action_name: Text, event: Event, next_action_in_the_future: Optional[Text]
    ) -> bool:
        # When actual users are talking to the action has to return an
        # `ActionExecutionRejected` in order to enter an unhappy path.
        loop_was_rejected_previously = (
            isinstance(event, ActionExecutionRejected)
            and event.action_name == loop_action_name
        )
        # During the policy training there are no `ActionExecutionRejected` events
        # which let us see whether we are within an unhappy path. Hence, we check if a
        # different action was executed instead of the loop after last user utterance.
        other_action_after_latest_user_utterance = (
            isinstance(event, UserUttered)
            and next_action_in_the_future is not None
            and next_action_in_the_future != loop_action_name
        )

        return loop_was_rejected_previously or other_action_after_latest_user_utterance

    @staticmethod
    def _undo_till_previous_loop_execution(
        loop_action_name: Text, done_events: List[Event]
    ) -> None:
        offset = 0
        for e in reversed(done_events[:]):
            if isinstance(e, ActionExecuted) and e.action_name == loop_action_name:
                break

            if isinstance(
                e, (ActionExecuted, UserUttered, DefinePrevUserUtteredFeaturization)
            ):
                del done_events[-1 - offset]
            else:
                # Remember events which aren't unfeaturized to get the index right
                offset += 1

    def replay_events(self) -> None:
        """Update the tracker based on a list of events."""
        applied_events = self.applied_events()
        for event in applied_events:
            event.apply_to(self)

    def recreate_from_dialogue(self, dialogue: Dialogue) -> None:
        """Use a serialised `Dialogue` to update the trackers state.

        This uses the state as is persisted in a ``TrackerStore``. If the
        tracker is blank before calling this method, the final state will be
        identical to the tracker from which the dialogue was created.
        """
        if not isinstance(dialogue, Dialogue):
            raise ValueError(
                f"story {dialogue} is not of type Dialogue. "
                f"Have you deserialized it?"
            )

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self) -> "DialogueStateTracker":
        """Creates a duplicate of this tracker."""
        return self.travel_back_in_time(float("inf"))

    def travel_back_in_time(self, target_time: float) -> "DialogueStateTracker":
        """Creates a new tracker with a state at a specific timestamp.

        A new tracker will be created and all events previous to the
        passed time stamp will be replayed. Events that occur exactly
        at the target time will be included.
        """
        tracker = self.init_copy()

        for event in self.events:
            if event.timestamp <= target_time:
                tracker.update(event)
            else:
                break

        return tracker  # yields the final state

    def as_dialogue(self) -> Dialogue:
        """Return a ``Dialogue`` object containing all of the turns.

        This can be serialised and later used to recover the state
        of this tracker exactly.
        """
        return Dialogue(self.sender_id, list(self.events))

    def update(self, event: Event, domain: Optional[Domain] = None) -> None:
        """根据 ``Event`` 修改跟踪器的状态。
        
        Args:
            event: 要应用的事件
            domain: 可选的域对象
            
        Raises:
            ValueError: 当事件不是 Event 子类实例时
        """
        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("要记录的事件必须是 Event 子类的实例。")

        if self.model_id and METADATA_MODEL_ID not in event.metadata:
            event.metadata = {**event.metadata, METADATA_MODEL_ID: self.model_id}

        if self.assistant_id and ASSISTANT_ID_KEY not in event.metadata:
            event.metadata = {**event.metadata, ASSISTANT_ID_KEY: self.assistant_id}

        self.events.append(event)
        event.apply_to(self)

    def update_with_events(
        self,
        new_events: List[Event],
        domain: Optional[Domain],
        override_timestamp: bool = True,
    ) -> None:
        """Adds multiple events to the tracker.

        Args:
            new_events: Events to apply.
            domain: The current model's domain.
            override_timestamp: If `True` refresh all timestamps of the events. As the
                events are usually created at some earlier point, this makes sure that
                all new events come after any current tracker events.
        """
        for e in new_events:
            if override_timestamp:
                e.timestamp = time.time()
            self.update(e, domain)

    def as_story(self, include_source: bool = False) -> "Story":
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string.
        """
        from rasa.shared.core.training_data.structures import Story

        story_name = (
            f"{self.sender_id} ({self.sender_source})"
            if include_source
            else self.sender_id
        )
        return Story.from_events(list(self.events), story_name)

    def export_stories(
        self,
        writer: "StoryWriter",
        e2e: bool = False,
        include_source: bool = False,
        should_append_stories: bool = False,
    ) -> Text:
        """Dump the tracker as a story in the Rasa Core story format.

        Returns:
            The dumped tracker as a string.
        """
        story = self.as_story(include_source)
        return writer.dumps(
            story.story_steps, is_appendable=should_append_stories, is_test_story=e2e
        )

    def export_stories_to_file(self, export_path: Text = "debug_stories.yml") -> None:
        """Dump the tracker as a story to a file."""
        from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
            YAMLStoryWriter,
        )

        append = os.path.exists(export_path)

        rasa.shared.utils.io.write_text_file(
            self.export_stories(YAMLStoryWriter(), should_append_stories=append) + "\n",
            export_path,
            append=append,
        )

    def get_last_event_for(
        self,
        event_type: Union[Type["EventTypeAlias"], Tuple[Type["EventTypeAlias"], ...]],
        action_names_to_exclude: List[Text] = None,
        skip: int = 0,
        event_verbosity: EventVerbosity = EventVerbosity.APPLIED,
    ) -> Optional["EventTypeAlias"]:
        """Gets the last event of a given type which was actually applied.

        Args:
            event_type: The type of event you want to find.
            action_names_to_exclude: Events of type `ActionExecuted` which
                should be excluded from the results. Can be used to skip
                `action_listen` events.
            skip: Skips n possible results before return an event.
            event_verbosity: Which `EventVerbosity` should be used to search for events.

        Returns:
            event which matched the query or `None` if no event matched.
        """
        to_exclude = action_names_to_exclude or []

        def filter_function(e: Event) -> bool:
            has_instance = isinstance(e, event_type)
            excluded = isinstance(e, ActionExecuted) and e.action_name in to_exclude
            return has_instance and not excluded

        filtered = filter(
            filter_function, reversed(self._events_for_verbosity(event_verbosity) or [])
        )

        for i in range(skip):
            next(filtered, None)

        return next(filtered, None)

    def last_executed_action_has(self, name: Text, skip: int = 0) -> bool:
        """Returns whether last `ActionExecuted` event had a specific name.

        Args:
            name: Name of the event which should be matched.
            skip: Skips n possible results in between.

        Returns:
            `True` if last executed action had name `name`, otherwise `False`.
        """
        last: Optional[ActionExecuted] = self.get_last_event_for(
            ActionExecuted, action_names_to_exclude=[ACTION_LISTEN_NAME], skip=skip
        )
        return last is not None and last.action_name == name

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_to`` method modifies the tracker.
    ###
    def _reset(self) -> None:
        """重置跟踪器到初始状态 - 不过不会删除事件！
        
        重置所有状态变量到初始值，但保留事件历史。
        """
        self._reset_slots()
        self._paused = False
        self.latest_action = {}
        self.latest_message = UserUttered.empty()
        self.latest_bot_utterance = BotUttered.empty()
        self.followup_action = ACTION_LISTEN_NAME
        self.active_loop = None

    def _reset_slots(self) -> None:
        """将所有槽位设置为其初始值。
        
        遍历所有槽位并调用其 reset() 方法。
        """
        for slot in self.slots.values():
            slot.reset()

    def _set_slot(self, key: Text, value: Any) -> None:
        """如果槽位存在，则设置槽位的值。
        
        Args:
            key: 槽位名称
            value: 要设置的值
        """
        if key in self.slots:
            slot = self.slots[key]
            slot.value = value
        else:
            logger.error(
                f"尝试设置不存在的槽位 '{key}'。请确保您 "
                f"已将所有槽位添加到域文件中。"
            )

    def _create_events(self, evts: List[Event]) -> Deque[Event]:
        """创建事件队列。
        
        Args:
            evts: 事件列表
            
        Returns:
            事件队列
            
        Raises:
            ValueError: 当事件不是 Event 实例时
        """
        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("如果提供事件，必须是事件列表")
        return deque(evts, self._max_event_history)

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, type(other)):
            return other.events == self.events and self.sender_id == other.sender_id
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def trigger_followup_action(self, action: Text) -> None:
        """Triggers another action following the execution of the current."""
        self.followup_action = action

    def clear_followup_action(self) -> None:
        """Clears follow up action when it was executed."""
        self.followup_action = None

    @property
    def active_loop_name(self) -> Optional[Text]:
        """Get the name of the currently active loop.

        Returns: `None` if no active loop or the name of the currently active loop.
        """
        if not self.active_loop or self.active_loop.name == SHOULD_NOT_BE_SET:
            return None

        return self.active_loop.name

    @property
    def latest_action_name(self) -> Optional[Text]:
        """Get the name of the previously executed action or text of e2e action.

        Returns: name of the previously executed action or text of e2e action
        """
        if self.latest_action is None:
            return None

        return self.latest_action.get(ACTION_NAME) or self.latest_action.get(
            ACTION_TEXT
        )

    @property
    def is_active_loop_rejected(self) -> bool:
        """Return True if there is an active loop and it's rejected."""
        return self.active_loop is not None and self.active_loop.rejected

    @property
    def is_active_loop_interrupted(self) -> bool:
        """Return True if there is an active loop and it's interrupted."""
        return self.active_loop is not None and self.active_loop.is_interrupted

    def fingerprint(self) -> Text:
        """Returns a unique hash for the tracker which is stable across python runs.

        Returns:
            fingerprint of the tracker
        """
        data: Dict[Text, Any] = {"sender_id": self.sender_id}

        if self.slots:
            data.update(self.slots)

        if self.events:
            data["events"] = list(self.events)

        return rasa.shared.utils.io.get_dictionary_fingerprint(data)


# =============================================================================
# 跟踪器事件差异引擎
# =============================================================================
class TrackerEventDiffEngine:
    """计算两个跟踪器的事件差异。"""

    @staticmethod
    def event_difference(
        original: DialogueStateTracker, tracker: DialogueStateTracker
    ) -> List[Event]:
        """返回新跟踪器中不存在于原始跟踪器中的所有事件。

        Args:
            original: 原始跟踪器
            tracker: 包含当前对话会话事件的跟踪器
            
        Returns:
            差异事件列表
        """
        offset = len(original.events) if original else 0
        events = tracker.events
        return list(itertools.islice(events, offset, len(events)))


# =============================================================================
# 辅助函数
# =============================================================================
def get_active_loop_name(
    state: State,
) -> Optional[Text]:
    """获取当前活动循环的名称。

    Args:
        state: 应从中提取活动循环名称的状态

    Return:
        活动循环的名称或 None
    """
    if (
        not state.get(ACTIVE_LOOP)
        or state[ACTIVE_LOOP].get(LOOP_NAME) == SHOULD_NOT_BE_SET
    ):
        return None

    # FIXME: 更好的 `State` 类型注解需要
    # 更大的重构（例如切换到 dataclass）
    return cast(Optional[Text], state[ACTIVE_LOOP].get(LOOP_NAME))


def is_prev_action_listen_in_state(state: State) -> bool:
    """检查 action_listen 是否是前一个执行的动作。

    Args:
        state: 应执行检查的状态

    Return:
        表示 action_listen 是否为前一动作的布尔值
    """
    prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
    return prev_action_name == ACTION_LISTEN_NAME


def get_trackers_for_conversation_sessions(
    tracker: DialogueStateTracker,
) -> List[DialogueStateTracker]:
    """为按对话会话分割的 `tracker` 生成跟踪器。

    Args:
        tracker: 要分割的 `DialogueStateTracker` 实例

    Returns:
        按对话会话分割的跟踪器列表
    """
    split_conversations = events.split_events(
        tracker.events,
        ActionExecuted,
        {"action_name": ACTION_SESSION_START_NAME},
        include_splitting_event=True,
    )

    return [
        DialogueStateTracker.from_events(
            tracker.sender_id,
            evts,
            tracker.slots.values(),
            sender_source=tracker.sender_source,
            max_event_history=tracker._max_event_history,
        )
        for evts in split_conversations
    ]
