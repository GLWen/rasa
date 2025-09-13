# =============================================================================
# 训练数据生成器模块 - 从故事和规则生成跟踪器
# =============================================================================
# 此模块负责从故事图和规则中生成训练数据，包括跟踪器的创建、
# 数据增强、去重等功能。它是 Rasa Core 训练系统的核心组件。

# 标准库导入
from collections import defaultdict, namedtuple, deque  # 集合工具

import copy                    # 深拷贝功能
import logging                 # 日志记录
import random                  # 随机数生成
from contextlib import contextmanager  # 上下文管理器

# 第三方库导入
from tqdm import tqdm          # 进度条显示

# 类型提示导入
from typing import (
    Optional,                  # 可选类型
    List,                      # 列表类型
    Text,                      # 文本类型
    Set,                       # 集合类型
    Dict,                      # 字典类型
    Tuple,                     # 元组类型
    Deque,                     # 双端队列类型
    DefaultDict,               # 默认字典类型
    Any,                       # 任意类型
    Iterable,                  # 可迭代类型
    Generator,                 # 生成器类型
)

# Rasa 内部模块导入
from rasa.shared.constants import DOCS_URL_STORIES  # 故事文档URL
from rasa.shared.core.constants import SHOULD_NOT_BE_SET  # 不应设置常量
from rasa.shared.core.domain import Domain, State  # 域和状态
from rasa.shared.core.events import (
    ActionExecuted,            # 动作执行事件
    UserUttered,               # 用户话语事件
    ActionReverted,            # 动作撤销事件
    UserUtteranceReverted,     # 用户话语撤销事件
    Restarted,                 # 重启事件
    Event,                     # 事件基类
    SlotSet,                   # 槽位设置事件
    ActiveLoop,                # 活动循环事件
)
from rasa.shared.core.trackers import DialogueStateTracker, FrozenState  # 跟踪器和冻结状态
from rasa.shared.core.slots import Slot  # 槽位
from rasa.shared.core.training_data.structures import (
    StoryGraph,                # 故事图
    STORY_START,               # 故事开始
    StoryStep,                 # 故事步骤
    RuleStep,                  # 规则步骤
    GENERATED_CHECKPOINT_PREFIX,  # 生成检查点前缀
)
from rasa.shared.utils.io import is_logging_disabled  # 日志禁用检查
import rasa.shared.utils.io    # IO工具

# 日志记录器
logger = logging.getLogger(__name__)

# =============================================================================
# 配置和数据结构定义
# =============================================================================

# 提取器配置命名元组
ExtractorConfig = namedtuple(
    "ExtractorConfig",
    "remove_duplicates "              # 是否移除重复项
    "unique_last_num_states "         # 唯一最后状态数量
    "augmentation_factor "            # 增强因子
    "max_number_of_augmented_trackers "  # 最大增强跟踪器数量
    "tracker_limit "                  # 跟踪器限制
    "use_story_concatenation "        # 是否使用故事连接
    "rand",                           # 随机数生成器
)


# =============================================================================
# 带缓存状态的跟踪器类
# =============================================================================

class TrackerWithCachedStates(DialogueStateTracker):
    """带缓存状态的跟踪器包装器。
    
    此类扩展了基本的对话状态跟踪器，添加了状态缓存功能，
    用于提高训练数据生成过程中的性能。
    """

    def __init__(
        self,
        sender_id: Text,                    # 发送者ID
        slots: Optional[Iterable[Slot]],    # 槽位列表
        max_event_history: Optional[int] = None,  # 最大事件历史
        domain: Optional[Domain] = None,    # 域对象
        is_augmented: bool = False,         # 是否为增强数据
        is_rule_tracker: bool = False,      # 是否为规则跟踪器
    ) -> None:
        """初始化带缓存状态的跟踪器。
        
        Args:
            sender_id: 发送者标识符
            slots: 槽位列表
            max_event_history: 最大事件历史记录数
            domain: 对话域
            is_augmented: 是否为数据增强生成的跟踪器
            is_rule_tracker: 是否为规则跟踪器
        """
        super().__init__(
            sender_id, slots, max_event_history, is_rule_tracker=is_rule_tracker
        )
        self._states_for_hashing: Deque[FrozenState] = deque()  # 用于哈希的状态缓存
        self.domain = domain if domain is not None else Domain.empty()  # 域对象
        # T/F 属性用于过滤增强故事
        self.is_augmented = is_augmented  # 是否为增强数据
        self.__skip_states = False        # 是否跳过状态更新

    @classmethod
    def from_events(
        cls,
        sender_id: Text,                    # 发送者ID
        evts: List[Event],                 # 事件列表
        slots: Optional[Iterable[Slot]] = None,  # 槽位列表
        max_event_history: Optional[int] = None,  # 最大事件历史
        sender_source: Optional[Text] = None,     # 发送者来源
        domain: Optional[Domain] = None,          # 域对象
        is_rule_tracker: bool = False,            # 是否为规则跟踪器
    ) -> "TrackerWithCachedStates":
        """从给定事件初始化跟踪器。
        
        Args:
            sender_id: 发送者标识符
            evts: 事件列表
            slots: 槽位列表
            max_event_history: 最大事件历史记录数
            sender_source: 发送者来源
            domain: 对话域
            is_rule_tracker: 是否为规则跟踪器
            
        Returns:
            初始化的跟踪器
        """
        tracker = cls(
            sender_id, slots, max_event_history, domain, is_rule_tracker=is_rule_tracker
        )
        for e in evts:  # 遍历所有事件
            tracker.update(e)  # 更新跟踪器状态
        return tracker  # 返回跟踪器

    def past_states_for_hashing(
        self, domain: Domain, omit_unset_slots: bool = False
    ) -> Deque[FrozenState]:
        """基于历史记录生成并缓存跟踪器的过去状态。

        Args:
            domain: a :class:`rasa.shared.core.domain.Domain`
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A list of states
        """
        if domain != self.domain:
            raise ValueError(
                "TrackerWithCachedStates cannot be used with a domain "
                "that is different from the one it was created with."
            )

        if omit_unset_slots:
            # the tracker caches states with omit_unset_slots=False
            # Retrieving them from cache with omit_unset_slots=True is not possible as
            # this information is lost after a position in the event stream is turned
            # into a state
            states = super().past_states(domain, omit_unset_slots=omit_unset_slots)
            states_for_hashing = deque(self.freeze_current_state(s) for s in states)
        else:
            # if don't have it cached, we use the domain to calculate the states
            # from the events
            # note: we ignore omit_unset_slots here as the cache was generated
            # with the default value
            states_for_hashing = self._states_for_hashing
            if not states_for_hashing:
                states = super().past_states(domain)
                states_for_hashing = deque(self.freeze_current_state(s) for s in states)

            self._states_for_hashing = states_for_hashing

        return states_for_hashing

    @staticmethod
    def _unfreeze_states(frozen_states: Deque[FrozenState]) -> List[State]:
        return [
            {key: dict(value) for key, value in dict(frozen_state).items()}
            for frozen_state in frozen_states
        ]

    def past_states(
        self,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """基于历史记录生成跟踪器的过去状态。

        Args:
            domain: The Domain.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            a list of states
        """
        states_for_hashing = self.past_states_for_hashing(
            domain, omit_unset_slots=omit_unset_slots
        )
        return self._unfreeze_states(states_for_hashing)

    def clear_states(self) -> None:
        """重置状态。"""
        self._states_for_hashing = deque()

    def init_copy(self) -> "TrackerWithCachedStates":
        """创建具有相同初始值的新状态跟踪器。"""
        return type(self)(
            "",
            self.slots.values(),
            self._max_event_history,
            self.domain,
            self.is_augmented,
            self.is_rule_tracker,
        )

    @contextmanager
    def _skip_states_manager(self) -> Generator[None, None, None]:
        self.__skip_states = True
        try:
            yield
        finally:
            self.__skip_states = False

    def copy(
        self, sender_id: Text = "", sender_source: Text = ""
    ) -> "TrackerWithCachedStates":
        """创建此跟踪器的副本。

        A new tracker will be created and all events
        will be replayed.
        """
        # This is an optimization, we could use the original copy, but
        # the states would be lost and we would need to recalculate them

        tracker = self.init_copy()
        tracker.sender_id = sender_id
        tracker.sender_source = sender_source

        with tracker._skip_states_manager():
            for event in self.events:
                tracker.update(event)

        tracker._states_for_hashing = copy.copy(self._states_for_hashing)

        return tracker

    def _append_current_state(self) -> None:
        if self._states_for_hashing is None:
            self._states_for_hashing = self.past_states_for_hashing(self.domain)
        else:
            state = self.domain.get_active_state(self)
            frozen_state = self.freeze_current_state(state)
            self._states_for_hashing.append(frozen_state)

    def update(
        self,
        event: Event,
        domain: Optional[Domain] = None,
    ) -> None:
        """根据事件修改跟踪器的状态。"""
        # if `skip_states` is `True`, this function behaves exactly like the
        # normal update of the `DialogueStateTracker`
        if not self._states_for_hashing and not self.__skip_states:
            # rest of this function assumes we have the previous state
            # cached. let's make sure it is there.
            self._states_for_hashing = self.past_states_for_hashing(self.domain)

        super().update(event)

        if not self.__skip_states:
            if isinstance(event, ActionExecuted):
                pass
            elif isinstance(event, ActionReverted):
                self._states_for_hashing.pop()  # removes the state after the action
                self._states_for_hashing.pop()  # removes the state used for the action
            elif isinstance(event, UserUtteranceReverted):
                self.clear_states()
            elif isinstance(event, Restarted):
                self.clear_states()
            else:
                self._states_for_hashing.pop()

            self._append_current_state()


# =============================================================================
# 类型定义
# =============================================================================

# 跟踪器查找字典类型
TrackerLookupDict = DefaultDict[Text, List[TrackerWithCachedStates]]

# 跟踪器元组类型（当前跟踪器列表，结束跟踪器列表）
TrackersTuple = Tuple[List[TrackerWithCachedStates], List[TrackerWithCachedStates]]


# =============================================================================
# 训练数据生成器类
# =============================================================================

class TrainingDataGenerator:
    """从训练数据生成跟踪器。
    
    此类负责从故事图和规则中生成训练数据，包括跟踪器的创建、
    数据增强、去重等功能。它是 Rasa Core 训练系统的核心组件。
    """

    def __init__(
        self,
        story_graph: StoryGraph,
        domain: Domain,
        remove_duplicates: bool = True,
        unique_last_num_states: Optional[int] = None,
        augmentation_factor: int = 50,
        tracker_limit: Optional[int] = None,
        use_story_concatenation: bool = True,
        debug_plots: bool = False,
    ):
        """给定一组故事部分，生成所有可能的故事。

        The different story parts can end and start with checkpoints
        and this generator will match start and end checkpoints to
        connect complete stories. Afterwards, duplicate stories will be
        removed and the data is augmented (if augmentation is enabled).
        """
        self.story_graph = story_graph.with_cycles_removed()
        if debug_plots:
            self.story_graph.visualize("story_blocks_connections.html")

        self.domain = domain

        # 10倍因子是增强轮数的启发式方法
        max_number_of_augmented_trackers = augmentation_factor * 10

        self.config = ExtractorConfig(
            remove_duplicates=remove_duplicates,
            unique_last_num_states=unique_last_num_states,
            augmentation_factor=augmentation_factor,
            max_number_of_augmented_trackers=max_number_of_augmented_trackers,
            tracker_limit=tracker_limit,
            use_story_concatenation=use_story_concatenation,
            rand=random.Random(42),
        )
        # 所有已完成跟踪器的哈希特征化
        self.hashed_featurizations: Set[int] = set()

    @staticmethod
    def _phase_name(everything_reachable_is_reached: bool, phase: int) -> Text:
        if everything_reachable_is_reached:
            return f"augmentation round {phase}"
        else:
            return f"data generation round {phase}"

    def generate(self) -> List[TrackerWithCachedStates]:
        """从故事和规则生成跟踪器。

        Returns:
            The generated trackers.
        """
        return self.generate_story_trackers() + self._generate_rule_trackers()

    def generate_story_trackers(self) -> List[TrackerWithCachedStates]:
        """从故事生成跟踪器（排除规则跟踪器）。

        Returns:
            The generated story trackers.
        """
        steps = [
            step
            for step in self.story_graph.ordered_steps()
            if not isinstance(step, RuleStep)
        ]

        return self._generate(steps, is_rule_data=False)

    def _generate_rule_trackers(self) -> List[TrackerWithCachedStates]:
        steps = [
            step
            for step in self.story_graph.ordered_steps()
            if isinstance(step, RuleStep)
        ]

        return self._generate(steps, is_rule_data=True)

    def _generate(
        self, story_steps: List[StoryStep], is_rule_data: bool = False
    ) -> List[TrackerWithCachedStates]:
        if not story_steps:
            logger.debug(f"No {'rules' if is_rule_data else 'story blocks'} found.")
            return []

        if self.config.remove_duplicates and self.config.unique_last_num_states:
            logger.debug(
                "Generated trackers will be deduplicated "
                "based on their unique last {} states."
                "".format(self.config.unique_last_num_states)
            )
        self._mark_first_action_in_story_steps_as_unpredictable()

        active_trackers: DefaultDict[Text, List[TrackerWithCachedStates]] = defaultdict(
            list
        )

        init_tracker = TrackerWithCachedStates(
            "",
            self.domain.slots,
            max_event_history=self.config.tracker_limit,
            domain=self.domain,
            is_rule_tracker=is_rule_data,
        )
        active_trackers[STORY_START].append(init_tracker)

        # 发送到特征化器的跟踪器
        finished_trackers = []
        # 为增强单独保留故事结尾跟踪器
        story_end_trackers = []

        phase = 0  # 一个阶段是对所有故事步骤的一次遍历

        # 不对规则数据进行增强
        if not is_rule_data:
            min_num_aug_phases = 3 if self.config.augmentation_factor > 0 else 0
            logger.debug(f"Number of augmentation rounds is {min_num_aug_phases}")
        else:
            min_num_aug_phases = 0

        # 跟踪检查点粘合过程的占位符
        used_checkpoints: Set[Text] = set()
        previous_unused: Set[Text] = set()
        everything_reachable_is_reached = False

        # 我们将继续生成数据，直到达到所有
        # 似乎可到达的检查点。这是一个启发式方法，
        # 如果我们在一次迭代中没有达到任何新的检查点，我们
        # 假设我们已经达到所有并停止

        while not everything_reachable_is_reached or phase < min_num_aug_phases:
            phase_name = self._phase_name(everything_reachable_is_reached, phase)

            num_active_trackers = self._count_trackers(active_trackers)

            if num_active_trackers:
                logger.debug(
                    "Starting {} ... (with {} trackers)"
                    "".format(phase_name, num_active_trackers)
                )
            else:
                logger.debug(f"There are no trackers for {phase_name}")
                break

            # 跟踪此阶段未使用的检查点
            unused_checkpoints: Set[Text] = set()

            desc = f"Processed {'rules' if is_rule_data else 'story blocks'}"
            pbar = tqdm(story_steps, desc=desc, disable=is_logging_disabled())
            for step in pbar:
                incoming_trackers: List[TrackerWithCachedStates] = []
                for start in step.start_checkpoints:
                    if active_trackers[start.name]:
                        ts = start.filter_trackers(active_trackers[start.name])
                        incoming_trackers.extend(ts)
                        used_checkpoints.add(start.name)
                    elif start.name not in used_checkpoints:
                        # need to skip - there was no previous step that
                        # had this start checkpoint as an end checkpoint
                        # it will be processed in next phases
                        unused_checkpoints.add(start.name)
                if not incoming_trackers:
                    # 如果没有跟踪器，
                    # 我们可以跳过循环的其余部分
                    continue

                # 这些是到达此故事的跟踪器
                # 步骤并需要处理步骤的所有事件

                if self.config.remove_duplicates:
                    incoming_trackers, end_trackers = self._remove_duplicate_trackers(
                        incoming_trackers
                    )

                    # 将结束跟踪器附加到已完成的跟踪器
                    finished_trackers.extend(end_trackers)

                if everything_reachable_is_reached:
                    # 增强轮
                    incoming_trackers = self._subsample_trackers(
                        incoming_trackers, self.config.max_number_of_augmented_trackers
                    )

                # 更新进度条
                pbar.set_postfix({"# trackers": "{:d}".format(len(incoming_trackers))})

                trackers, end_trackers = self._process_step(step, incoming_trackers)

                # 将结束跟踪器添加到已完成的跟踪器
                finished_trackers.extend(end_trackers)

                # 用跟踪器更新我们的跟踪器字典
                # 处理了步骤的事件并且
                # 现在可以用于进一步的故事步骤
                # 以此步骤结束的检查点开始

                for end in step.end_checkpoints:
                    start_name = self._find_start_checkpoint_name(end.name)

                    active_trackers[start_name].extend(trackers)

                    if start_name in used_checkpoints:
                        # 将结束检查点添加为未使用
                        # 如果此检查点被处理为
                        # 之前的开始
                        unused_checkpoints.add(start_name)

                if not step.end_checkpoints:
                    unique_ends = self._remove_duplicate_story_end_trackers(trackers)
                    story_end_trackers.extend(unique_ends)

            num_finished = len(finished_trackers) + len(story_end_trackers)
            logger.debug(f"Finished phase ({num_finished} training samples found).")

            # 准备下一轮
            phase += 1

            if not everything_reachable_is_reached:
                # 检查我们是否达到了所有可以到达的节点
                # 如果这一轮我们至少达到了一个更多节点
                # 比上一个，我们假设仍然有
                # 一些东西需要达到，我们继续

                unused_checkpoints = self._add_unused_end_checkpoints(
                    set(active_trackers.keys()), unused_checkpoints, used_checkpoints
                )
                active_trackers = self._filter_active_trackers(
                    active_trackers, unused_checkpoints
                )
                num_active_trackers = self._count_trackers(active_trackers)

                everything_reachable_is_reached = (
                    unused_checkpoints == previous_unused or num_active_trackers == 0
                )
                previous_unused = unused_checkpoints

                if everything_reachable_is_reached:
                    # 应该只发生一次

                    previous_unused -= used_checkpoints
                    # 添加带有未使用检查点的跟踪器
                    # 到已完成的跟踪器
                    for start_name in previous_unused:
                        finished_trackers.extend(active_trackers[start_name])

                    logger.debug("Data generation rounds finished.")
                    logger.debug(
                        "Found {} unused checkpoints".format(len(previous_unused))
                    )
                    phase = 0
                else:
                    logger.debug(
                        "Found {} unused checkpoints "
                        "in current phase."
                        "".format(len(unused_checkpoints))
                    )
                    logger.debug(
                        "Found {} active trackers "
                        "for these checkpoints."
                        "".format(num_active_trackers)
                    )

            if everything_reachable_is_reached:
                # 增强轮, so we process only
                # 故事结尾检查点
                # 重置已使用的检查点
                used_checkpoints = set()

                # 为增强生成活动跟踪器
                active_trackers = self._create_start_trackers_for_augmentation(
                    story_end_trackers
                )

        finished_trackers.extend(story_end_trackers)
        self._issue_unused_checkpoint_notification(previous_unused)
        logger.debug("Found {} training trackers.".format(len(finished_trackers)))

        if self.config.augmentation_factor > 0:
            augmented_trackers, original_trackers = [], []
            for t in finished_trackers:
                if t.is_augmented:
                    augmented_trackers.append(t)
                else:
                    original_trackers.append(t)
            augmented_trackers = self._subsample_trackers(
                augmented_trackers, self.config.max_number_of_augmented_trackers
            )
            logger.debug(
                "Subsampled to {} augmented training trackers."
                "".format(len(augmented_trackers))
            )
            logger.debug(
                "There are {} original trackers.".format(len(original_trackers))
            )
            finished_trackers = original_trackers + augmented_trackers

        return finished_trackers

    @staticmethod
    def _count_trackers(active_trackers: TrackerLookupDict) -> int:
        """计算跟踪器字典中的跟踪器数量。"""
        return sum(len(ts) for ts in active_trackers.values())

    def _subsample_trackers(
        self,
        incoming_trackers: List[TrackerWithCachedStates],
        max_number_of_trackers: int,
    ) -> List[TrackerWithCachedStates]:
        """对跟踪器列表进行子采样以获取随机子集。"""

        # 如果流程变得很长并且有很多分支，我们
        # 通过收集太多跟踪器而陷入麻烦
        # 因此进行子采样
        if max_number_of_trackers is not None:
            return _subsample_array(
                incoming_trackers, max_number_of_trackers, rand=self.config.rand
            )
        else:
            return incoming_trackers

    def _find_start_checkpoint_name(self, end_name: Text) -> Text:
        """给定循环的结束检查点名称，查找开始检查点名称"""
        return self.story_graph.story_end_checkpoints.get(end_name, end_name)

    @staticmethod
    def _add_unused_end_checkpoints(
        start_checkpoints: Set[Text],
        unused_checkpoints: Set[Text],
        used_checkpoints: Set[Text],
    ) -> Set[Text]:
        """Add unused end checkpoints
        if they were never encountered as start checkpoints
        """

        return unused_checkpoints.union(
            {
                start_name
                for start_name in start_checkpoints
                if start_name not in used_checkpoints
            }
        )

    @staticmethod
    def _filter_active_trackers(
        active_trackers: TrackerLookupDict, unused_checkpoints: Set[Text]
    ) -> TrackerLookupDict:
        """Filter active trackers that ended with unused checkpoint
        or are parts of loops."""
        next_active_trackers = defaultdict(list)

        for start_name in unused_checkpoints:
            # process trackers ended with unused checkpoints further
            if start_name != STORY_START:
                # there is no point to process STORY_START checkpoint again
                next_active_trackers[start_name] = active_trackers.get(start_name, [])

        return next_active_trackers

    def _create_start_trackers_for_augmentation(
        self, story_end_trackers: List[TrackerWithCachedStates]
    ) -> TrackerLookupDict:
        """This is where the augmentation magic happens.

        We will reuse all the trackers that reached the
        end checkpoint `None` (which is the end of a
        story) and start processing all steps again. So instead
        of starting with a fresh tracker, the second and
        all following phases will reuse a couple of the trackers
        that made their way to a story end.

        We need to do some cleanup before processing them again.
        """
        next_active_trackers = defaultdict(list)

        if self.config.use_story_concatenation:
            ending_trackers = _subsample_array(
                story_end_trackers,
                self.config.augmentation_factor,
                rand=self.config.rand,
            )
            for t in ending_trackers:
                # 这是一个讨厌的事情 - 所有故事都结束并且
                # 以动作监听开始 - 所以在记录第一个之后
                # 下一阶段的动作，跟踪器会
                # 包含动作监听后跟动作监听
                # 为了解决这个问题，我们将"撤销"最后一个动作监听

                # 跟踪器应该被复制，
                # 否则原始跟踪器被更新
                aug_t = t.copy()
                aug_t.is_augmented = True
                aug_t.update(ActionReverted())
                next_active_trackers[STORY_START].append(aug_t)

        return next_active_trackers

    def _process_step(
        self, step: StoryStep, incoming_trackers: List[TrackerWithCachedStates]
    ) -> TrackersTuple:
        """使用所有跟踪器处理步骤的事件。

        The trackers that reached the steps starting checkpoint will
        be used to process the events. Collects and returns training
        data while processing the story step."""

        events = step.explicit_events(self.domain)

        trackers = []
        if events:  # small optimization

            # 需要复制跟踪器，因为多个故事步骤
            # 可能以相同的检查点开始，它们都
            # 将使用相同的传入跟踪器集

            for tracker in incoming_trackers:
                # 发送者ID用于让人类看到
                # 此跟踪器的消息和事件来自哪里 - 要做到这一点
                # 我们连接块的块名称
                # 对跟踪器事件有贡献
                if tracker.sender_id:
                    if (
                        step.block_name
                        and step.block_name not in tracker.sender_id.split(" > ")
                    ):
                        new_sender = tracker.sender_id + " > " + step.block_name
                    else:
                        new_sender = tracker.sender_id
                else:
                    new_sender = step.block_name
                trackers.append(tracker.copy(new_sender, step.source_name))

        end_trackers = []
        for event in events:
            if (
                isinstance(event, ActionExecuted)
                and event.action_text
                and event.action_text not in self.domain.action_texts
            ):
                rasa.shared.utils.cli.print_warning(
                    f"Test story '{step.block_name}' in "
                    f"'{step.source_name}' contains the bot utterance "
                    f"'{event.action_text}', which is not part "
                    f"of the training data / domain."
                )
            for tracker in trackers:
                if isinstance(
                    event, (ActionReverted, UserUtteranceReverted, Restarted)
                ):
                    end_trackers.append(tracker.copy(tracker.sender_id))
                if isinstance(step, RuleStep):
                    # The rules can specify that a form or a slot shouldn't be set,
                    # therefore we need to distinguish between not set
                    # and explicitly set to None
                    if isinstance(event, ActiveLoop) and event.name is None:
                        event.name = SHOULD_NOT_BE_SET

                    if isinstance(event, SlotSet) and event.value is None:
                        event.value = SHOULD_NOT_BE_SET

                tracker.update(event)

        # 结束跟踪器应该单独返回
        # 以避免将它们用于增强
        return trackers, end_trackers

    def _remove_duplicate_trackers(
        self, trackers: List[TrackerWithCachedStates]
    ) -> TrackersTuple:
        """移除创建相等特征化的跟踪器
            for current story step.

        From multiple trackers that create equal featurizations
        we only need to keep one. Because as we continue processing
        events and story steps, all trackers that created the
        same featurization once will do so in the future (as we
        feed the same events to all trackers)."""

        step_hashed_featurizations = set()

        # 收集创建不同特征化的跟踪器
        unique_trackers = []  # 对于当前步骤
        end_trackers = []  # 对于所有步骤

        for tracker in trackers:
            states_for_hashing = tuple(tracker.past_states_for_hashing(self.domain))
            hashed = hash(states_for_hashing)

            # 只继续使用创建的跟踪器
            # 我们尚未观察到的哈希特征化
            if hashed not in step_hashed_featurizations:
                if self.config.unique_last_num_states:
                    last_states = states_for_hashing[
                        -self.config.unique_last_num_states :
                    ]
                    last_hashed = hash(last_states)

                    if last_hashed not in step_hashed_featurizations:
                        step_hashed_featurizations.add(last_hashed)
                        unique_trackers.append(tracker)
                    elif (
                        len(states_for_hashing) > len(last_states)
                        and hashed not in self.hashed_featurizations
                    ):
                        self.hashed_featurizations.add(hashed)
                        end_trackers.append(tracker)
                else:
                    unique_trackers.append(tracker)

                step_hashed_featurizations.add(hashed)

        return unique_trackers, end_trackers

    def _remove_duplicate_story_end_trackers(
        self, trackers: List[TrackerWithCachedStates]
    ) -> List[TrackerWithCachedStates]:
        """移除到达故事结尾并创建相等特征化的跟踪器。"""

        # 收集创建不同特征化的跟踪器
        unique_trackers = []  # 对于所有步骤

        # 需要去重已完成的跟踪器，
        # 否则特征化会做很多不必要的工作

        for tracker in trackers:
            states_for_hashing = tuple(tracker.past_states_for_hashing(self.domain))
            hashed = hash(states_for_hashing + (tracker.is_rule_tracker,))

            # 只继续使用创建的跟踪器
            # 我们尚未观察到的哈希特征化

            if hashed not in self.hashed_featurizations:
                self.hashed_featurizations.add(hashed)
                unique_trackers.append(tracker)

        return unique_trackers

    def _mark_first_action_in_story_steps_as_unpredictable(self) -> None:
        """标记在机器学习训练期间不应使用的动作。

        If a story starts with an action, we can not use
        that first action as a training example, as there is no
        history. There is one exception though, we do want to
        predict action listen. But because stories never
        contain action listen events (they are added when a
        story gets converted to a dialogue) we need to apply a
        small trick to avoid marking actions occurring after
        an action listen as unpredictable."""

        for step in self.story_graph.story_steps:
            # TODO：如果步骤是对话开始，这不起作用
            #       以及对话的中间部分
            #       这意味着检查点可以有多个
            #       检查点或对话的开始
            #       但不是两者
            if STORY_START in {s.name for s in step.start_checkpoints}:
                for i, e in enumerate(step.events):
                    if isinstance(e, UserUttered):
                        # 如果有用户话语，那意味着在
                        # 用户说出某些东西之前必须有
                        # 动作监听。因此，任何动作
                        # 在此用户话语之后不是第一个
                        # 动作了，用于预测的跟踪器
                        # 不再为空。因此，
                        # 预测话语后发生的任何事情是可以的
                        break
                    if isinstance(e, ActionExecuted):
                        e.unpredictable = True
                        break

    def _issue_unused_checkpoint_notification(
        self, unused_checkpoints: Set[Text]
    ) -> None:
        """警告未使用的故事块。

        Unused steps are ones having a start or end checkpoint
        that no one provided."""

        if STORY_START in unused_checkpoints:
            rasa.shared.utils.io.raise_warning(
                "There is no starting story block "
                "in the training data. "
                "All your story blocks start with some checkpoint. "
                "There should be at least one story block "
                "that starts without any checkpoint.",
                docs=DOCS_URL_STORIES + "#stories",
            )

        # 首先运行步骤将只产生一个警告
        # 每个块（因为一个块可能有多个步骤）
        collected_start = set()
        collected_end = set()
        for step in self.story_graph.story_steps:
            for start in step.start_checkpoints:
                if start.name in unused_checkpoints:
                    # 处理后，不应该有故事部分剩余
                    # 这表示不存在的开始检查点
                    collected_start.add((start.name, step.block_name))

            for end in step.end_checkpoints:
                if end.name in unused_checkpoints:
                    # 处理后，不应该有故事部分剩余
                    # 这表示不存在的结束检查点
                    collected_end.add((end.name, step.block_name))

        for cp, block_name in collected_start:
            if not cp.startswith(GENERATED_CHECKPOINT_PREFIX):
                rasa.shared.utils.io.raise_warning(
                    f"Unsatisfied start checkpoint '{cp}' "
                    f"in block '{block_name}'. "
                    f"Remove this checkpoint or add "
                    f"story blocks that end "
                    f"with this checkpoint.",
                    docs=DOCS_URL_STORIES + "#checkpoints",
                )

        for cp, block_name in collected_end:
            if not cp.startswith(GENERATED_CHECKPOINT_PREFIX):
                rasa.shared.utils.io.raise_warning(
                    f"Unsatisfied end checkpoint '{cp}' "
                    f"in block '{block_name}'. "
                    f"Remove this checkpoint or add "
                    f"story blocks that start "
                    f"with this checkpoint.",
                    docs=DOCS_URL_STORIES + "#checkpoints",
                )


def _subsample_array(
    arr: List[Any],
    max_values: int,
    can_modify_incoming_array: bool = True,
    rand: Optional[random.Random] = None,
) -> List[Any]:
    """打乱数组并返回 `max_values` 个元素。"""
    if not can_modify_incoming_array:
        arr = arr[:]
    if rand is not None:
        rand.shuffle(arr)
    else:
        random.shuffle(arr)
    return arr[:max_values]
