# 导入未来版本的注解支持，允许在类型注解中使用前向引用
from __future__ import annotations
# 导入深拷贝模块，用于创建对象的深拷贝
import copy
# 导入函数工具模块，用于装饰器等功能
import functools
# 导入日志模块，用于记录日志信息
import logging
# 导入结构化日志模块，用于结构化日志记录
import structlog
# 导入类型注解相关的模块
from typing import Any, List, DefaultDict, Dict, Text, Optional, Set, Tuple, cast

# 导入进度条模块，用于显示训练进度
from tqdm import tqdm
# 导入numpy数值计算库
import numpy as np
# 导入JSON处理模块
import json
# 导入默认字典，用于创建带默认值的字典
from collections import defaultdict

# 导入Rasa引擎相关模块
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入常量
from rasa.shared.constants import DOCS_URL_RULES
# 导入异常类
from rasa.shared.exceptions import RasaException
# 导入工具模块
import rasa.shared.utils.io
# 导入核心事件类
from rasa.shared.core.events import LoopInterrupted, UserUttered, ActionExecuted
# 导入特征化器
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
# 导入记忆化策略基类
from rasa.core.policies.memoization import MemoizationPolicy
# 导入策略基类和预测类
from rasa.core.policies.policy import SupportedData, PolicyPrediction
# 导入跟踪器相关模块
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
# 导入带缓存的跟踪器
from rasa.shared.core.generator import TrackerWithCachedStates
# 导入核心常量
from rasa.core.constants import (
    DEFAULT_CORE_FALLBACK_THRESHOLD,  # 默认核心回退阈值
    RULE_POLICY_PRIORITY,             # 规则策略优先级
    POLICY_PRIORITY,                  # 策略优先级键名
    POLICY_MAX_HISTORY,               # 策略最大历史长度键名
)
# 导入更多核心常量
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,              # 用户重启意图
    USER_INTENT_BACK,                 # 用户返回意图
    USER_INTENT_SESSION_START,        # 用户会话开始意图
    ACTION_LISTEN_NAME,               # 监听动作名称
    ACTION_RESTART_NAME,              # 重启动作名称
    ACTION_SESSION_START_NAME,        # 会话开始动作名称
    ACTION_DEFAULT_FALLBACK_NAME,     # 默认回退动作名称
    ACTION_BACK_NAME,                 # 返回动作名称
    RULE_SNIPPET_ACTION_NAME,         # 规则片段动作名称
    SHOULD_NOT_BE_SET,                # 不应设置的常量
    PREVIOUS_ACTION,                  # 前一个动作
    LOOP_NAME,                        # 循环名称
    SLOTS,                            # 槽位
    ACTIVE_LOOP,                      # 活跃循环
    RULE_ONLY_SLOTS,                  # 仅规则槽位
    RULE_ONLY_LOOPS,                  # 仅规则循环
)
# 导入领域相关模块
from rasa.shared.core.domain import InvalidDomain, State, Domain
# 导入NLU常量
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
# 导入测试模块
import rasa.core.test
# 导入训练相关模块
from rasa.core.training.training import create_action_fingerprints, ActionFingerprint

# 创建日志记录器
logger = logging.getLogger(__name__)
# 创建结构化日志记录器
structlogger = structlog.get_logger()


# 这些是Rasa开源默认动作，在任何时候都会覆盖其他所有动作
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,        # 重启意图映射到重启动作
    USER_INTENT_BACK: ACTION_BACK_NAME,               # 返回意图映射到返回动作
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,  # 会话开始意图映射到会话开始动作
}

# 规则相关的常量定义
RULES = "rules"                                    # 规则键名
RULES_FOR_LOOP_UNHAPPY_PATH = "rules_for_loop_unhappy_path"  # 循环不愉快路径规则键名
RULES_NOT_IN_STORIES = "rules_not_in_stories"      # 不在故事中的规则键名

# 循环状态相关常量
LOOP_WAS_INTERRUPTED = "loop_was_interrupted"      # 循环被中断
DO_NOT_PREDICT_LOOP_ACTION = "do_not_predict_loop_action"  # 不预测循环动作

# 规则描述相关常量
DEFAULT_RULES = "predicting default action with intent "  # 默认规则描述前缀
LOOP_RULES = "handling active loops and forms - "         # 循环规则描述前缀
LOOP_RULES_SEPARATOR = " - "                              # 循环规则分隔符


class InvalidRule(RasaException):
    """当规则无效时可以引发的异常。"""

    def __init__(self, message: Text) -> None:
        """初始化无效规则异常。
        
        Args:
            message: 错误消息
        """
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        """返回异常字符串表示。
        
        Returns:
            包含错误消息和文档链接的字符串
        """
        return self.message + (
            f"\nYou can find more information about the usage of "
            f"rules at {DOCS_URL_RULES}. "
        )


# 注册为默认V1配方组件，不支持端到端，但可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class RulePolicy(MemoizationPolicy):
    """处理所有规则的策略。"""

    # 规则使用显式JSON字符串，不使用特征字符串压缩
    ENABLE_FEATURE_STRING_COMPRESSION = False

    # 在规则受限情况下允许的用户输入数量
    ALLOWED_NUMBER_OF_USER_INPUTS = 1

    @staticmethod
    def supported_data() -> SupportedData:
        """此策略支持的数据类型。

        Returns:
            此策略支持的数据类型（机器学习和规则数据）。
        """
        return SupportedData.ML_AND_RULE_DATA

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回默认配置（完整文档字符串请参见父类）。"""
        return {
            # 策略优先级，当多个策略以相同置信度预测动作时使用
            POLICY_PRIORITY: RULE_POLICY_PRIORITY,
            # 当没有规则匹配时的预测置信度，实际上是核心回退的阈值
            "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
            # 当没有规则匹配时应该预测的动作名称
            "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
            # 如果为`True`，在没有规则匹配时预测`core_fallback_action_name`
            "enable_fallback_prediction": True,
            # 如果为`True`，规则被限制为最多包含1个用户消息。
            # 这用于避免用户使用规则构建状态机
            "restrict_rules": True,
            # 是否检查规则和故事之间的矛盾
            "check_for_contradictions": True,
            # 策略将使用NLU对最新用户消息的置信度来设置动作的置信度
            "use_nlu_confidence_as_score": False,
        }

    def __init__(
        self,
        config: Dict[Text, Any],           # 策略配置字典
        model_storage: ModelStorage,       # 模型存储对象
        resource: Resource,                # 资源对象
        execution_context: ExecutionContext,  # 执行上下文
        featurizer: Optional[TrackerFeaturizer] = None,  # 可选的跟踪器特征化器
        lookup: Optional[Dict] = None,     # 可选的查找字典
    ) -> None:
        """初始化策略。"""
        # 将最大历史设置为`None`以捕获任何长度的规则故事
        config[POLICY_MAX_HISTORY] = None

        # 调用父类初始化方法
        super().__init__(
            config, model_storage, resource, execution_context, featurizer, lookup
        )

        # 从配置中获取回退动作名称
        self._fallback_action_name = config["core_fallback_action_name"]
        # 从配置中获取是否启用回退预测
        self._enable_fallback_prediction = config["enable_fallback_prediction"]
        # 从配置中获取是否检查矛盾
        self._check_for_contradictions = config["check_for_contradictions"]

        # 初始化规则源字典，用于存储规则来源信息
        self._rules_sources: DefaultDict[Text, List[Tuple[Text, Text]]] = defaultdict(
            list
        )

    @classmethod
    def raise_if_incompatible_with_domain(
        cls, config: Dict[Text, Any], domain: Domain
    ) -> None:
        """检查领域动作名称是否与配置的回退动作匹配。

        Args:
            config: `RulePolicy`的配置
            domain: 领域对象
        Raises:
            `InvalidDomain`: 如果此策略与领域不兼容
        """
        # 获取配置中的回退动作名称
        fallback_action_name = config.get("core_fallback_action_name", None)
        # 检查回退动作是否存在于领域中
        if (
            fallback_action_name
            and fallback_action_name not in domain.action_names_or_texts
        ):
            raise InvalidDomain(
                f"The fallback action '{fallback_action_name}' which was "
                f"configured for the {RulePolicy.__name__} must be "
                f"present in the domain."
            )

    @staticmethod
    def _is_rule_snippet_state(state: State) -> bool:
        """检查状态是否为规则片段状态。
        
        Args:
            state: 要检查的状态
            
        Returns:
            如果是规则片段状态则返回True
        """
        # 获取前一个动作名称
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        # 检查是否为规则片段动作
        return prev_action_name == RULE_SNIPPET_ACTION_NAME

    def _create_feature_key(self, states: List[State]) -> Optional[Text]:
        """从状态列表创建特征键。
        
        Args:
            states: 状态列表
            
        Returns:
            特征键字符串，如果没有有效状态则返回None
        """
        new_states: List[State] = []
        # 从后往前遍历状态
        for state in reversed(states):
            if self._is_rule_snippet_state(state):
                # 移除RULE_SNIPPET_ACTION_NAME之前的所有状态
                break
            new_states.insert(0, state)

        if not new_states:
            return None

        # 我们对键进行排序以确保相同的状态
        # 表示为字典时具有相同的JSON字符串
        return json.dumps(new_states, sort_keys=True)

    @staticmethod
    def _states_for_unhappy_loop_predictions(states: List[State]) -> List[State]:
        """修改状态以创建循环不愉快路径条件的特征键。

        Args:
            states: 跟踪器的表示，作为包含特征的字典列表

        Returns:
            修改后的状态
        """
        # 只保留最后2个对话轮次以
        # - 捕获action_listen之前的前一个有意义的动作
        # - 忽略前一个意图
        if len(states) == 1 or not states[-2].get(PREVIOUS_ACTION):
            return [states[-1]]
        else:
            return [{PREVIOUS_ACTION: states[-2][PREVIOUS_ACTION]}, states[-1]]

    @staticmethod
    def _remove_rule_snippet_predictions(lookup: Dict[Text, Text]) -> Dict[Text, Text]:
        """删除会预测RULE_SNIPPET_ACTION_NAME动作的规则。
        
        Args:
            lookup: 查找字典
            
        Returns:
            过滤后的查找字典
        """
        # 如果规则会预测RULE_SNIPPET_ACTION_NAME动作，则删除该规则
        return {
            feature_key: action
            for feature_key, action in lookup.items()
            if action != RULE_SNIPPET_ACTION_NAME
        }

    def _create_loop_unhappy_lookup_from_states(
        self,
        trackers_as_states: List[List[State]],  # 跟踪器作为状态列表的表示
        trackers_as_actions: List[List[Text]],  # 跟踪器作为动作列表的表示
    ) -> Dict[Text, Text]:
        """从表示为状态的跟踪器创建查找字典。

        Args:
            trackers_as_states: 跟踪器作为状态列表的表示
            trackers_as_actions: 跟踪器作为动作列表的表示

        Returns:
            查找字典
        """
        lookup = {}
        # 遍历状态和动作
        for states, actions in zip(trackers_as_states, trackers_as_actions):
            action = actions[0]
            # 获取活跃循环名称
            active_loop = get_active_loop_name(states[-1])
            # 即使有两个相同的特征键，它们的循环也是相同的
            if not active_loop:
                continue

            # 为不愉快循环预测修改状态
            states = self._states_for_unhappy_loop_predictions(states)
            # 创建特征键
            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            # 由于循环内的规则片段和故事只包含不愉快路径，
            # 通知循环它是在回答不同问题后预测的，
            # 因此不应该验证用户输入
            if (
                # 循环在不愉快路径中的action_listen之后被预测，
                # 因此不需要验证
                is_prev_action_listen_in_state(states[-1])
                and action == active_loop
            ):
                lookup[feature_key] = LOOP_WAS_INTERRUPTED
            elif (
                # 在不愉快路径中预测了除active_loop之外的某个动作，
                # 因此active_loop不应该被规则预测
                not is_prev_action_listen_in_state(states[-1])
                and action != active_loop
            ):
                lookup[feature_key] = DO_NOT_PREDICT_LOOP_ACTION
        return lookup

    def _check_rule_restriction(
        self, rule_trackers: List[TrackerWithCachedStates]
    ) -> None:
        """检查规则限制，确保规则不包含过多用户输入。
        
        Args:
            rule_trackers: 规则跟踪器列表
            
        Raises:
            InvalidRule: 如果规则包含超过允许数量的用户输入
        """
        rules_exceeding_max_user_turns = []
        # 遍历所有规则跟踪器
        for tracker in rule_trackers:
            # 计算用户发言次数
            number_of_user_uttered = sum(
                isinstance(event, UserUttered) for event in tracker.events
            )
            # 如果超过允许的用户输入数量
            if number_of_user_uttered > self.ALLOWED_NUMBER_OF_USER_INPUTS:
                rules_exceeding_max_user_turns.append(tracker.sender_id)

        # 如果有规则超过限制，抛出异常
        if rules_exceeding_max_user_turns:
            raise InvalidRule(
                f"Found rules '{', '.join(rules_exceeding_max_user_turns)}' "
                f"that contain more than {self.ALLOWED_NUMBER_OF_USER_INPUTS} "
                f"user message. Rules are not meant to hardcode a state machine. "
                f"Please use stories for these cases."
            )

    @staticmethod
    def _expected_but_missing_slots(
        fingerprint: ActionFingerprint, state: State
    ) -> Set[Text]:
        """检查期望但缺失的槽位。
        
        Args:
            fingerprint: 动作指纹
            state: 当前状态
            
        Returns:
            期望但缺失的槽位集合
        """
        # 获取期望的槽位
        expected_slots = set(fingerprint.slots)
        # 获取当前槽位
        current_slots = set(state.get(SLOTS, {}).keys())
        # 报告所有期望但在当前槽位中未设置的槽位
        return expected_slots.difference(current_slots)

    @staticmethod
    def _check_active_loops_fingerprint(
        fingerprint: ActionFingerprint, state: State
    ) -> Set[Optional[Text]]:
        """检查活跃循环指纹。
        
        Args:
            fingerprint: 动作指纹
            state: 当前状态
            
        Returns:
            期望的活跃循环集合
        """
        # 获取期望的活跃循环
        expected_active_loops = set(fingerprint.active_loop)
        # 我们不使用tracker.active_loop_name
        # 因为我们需要保持should_not_be_set
        current_active_loop = state.get(ACTIVE_LOOP, {}).get(LOOP_NAME)
        if current_active_loop in expected_active_loops:
            # 期望的活跃循环之一已设置
            return set()

        return expected_active_loops

    @staticmethod
    def _error_messages_from_fingerprints(
        action_name: Text,                    # 动作名称
        missing_fingerprint_slots: Set[Text], # 缺失的指纹槽位
        fingerprint_active_loops: Set[Text],  # 指纹活跃循环
        rule_name: Text,                      # 规则名称
    ) -> List[Text]:
        """从指纹生成错误消息。
        
        Args:
            action_name: 动作名称
            missing_fingerprint_slots: 缺失的指纹槽位
            fingerprint_active_loops: 指纹活跃循环
            rule_name: 规则名称
            
        Returns:
            错误消息列表
        """
        error_messages = []
        # 检查缺失的槽位
        if action_name and missing_fingerprint_slots:
            error_messages.append(
                f"- the action '{action_name}' in rule '{rule_name}' does not set some "
                f"of the slots that it sets in other rules. Slots not set in rule "
                f"'{rule_name}': '{', '.join(missing_fingerprint_slots)}'. Please "
                f"update the rule with an appropriate slot or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        # 检查活跃循环
        if action_name and fingerprint_active_loops:
            # 将`SHOULD_NOT_BE_SET`替换为`null`，以便用户知道在规则中放什么
            fingerprint_active_loops = set(
                "null" if active_loop == SHOULD_NOT_BE_SET else active_loop
                for active_loop in fingerprint_active_loops
            )
            # 将action_name添加到活跃循环中，以便用户知道在规则中放什么
            fingerprint_active_loops.add(action_name)

            error_messages.append(
                f"- the form '{action_name}' in rule '{rule_name}' does not set "
                f"the 'active_loop', that it sets in other rules: "
                f"'{', '.join(fingerprint_active_loops)}'. Please update the rule with "
                f"the appropriate 'active loop' property or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        return error_messages

    def _check_for_incomplete_rules(
        self, rule_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> None:
        """检查不完整的规则。
        
        Args:
            rule_trackers: 规则跟踪器列表
            domain: 领域对象
            
        Raises:
            InvalidRule: 如果发现不完整的规则
        """
        logger.debug("Started checking if some rules are incomplete.")
        # 我们只需要使用规则中的指纹
        rule_fingerprints = create_action_fingerprints(rule_trackers, domain)
        if not rule_fingerprints:
            return

        error_messages: List[Text] = []
        # 遍历所有规则跟踪器
        for tracker in rule_trackers:
            states = tracker.past_states(domain)
            # 最后一个动作总是action_listen
            action_names = [
                state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME) for state in states[1:]
            ] + [ACTION_LISTEN_NAME]

            # 遍历状态和动作名称
            for state, action_name in zip(states, action_names):
                previous_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
                fingerprint = rule_fingerprints.get(previous_action_name)
                if (
                    not previous_action_name
                    or not fingerprint
                    or action_name == RULE_SNIPPET_ACTION_NAME
                    or previous_action_name == RULE_SNIPPET_ACTION_NAME
                ):
                    # 不要检查规则片段动作的指纹
                    # 如果当前动作是规则片段动作，不要在前一个动作的指纹不满足时抛出异常
                    continue

                # 检查缺失的期望槽位
                missing_expected_slots = self._expected_but_missing_slots(
                    fingerprint, state
                )
                # 检查活跃循环指纹
                expected_active_loops = self._check_active_loops_fingerprint(
                    fingerprint, state
                )
                # 生成错误消息
                error_messages.extend(
                    self._error_messages_from_fingerprints(
                        previous_action_name,
                        missing_expected_slots,
                        expected_active_loops,
                        tracker.sender_id,
                    )
                )

        # 如果有错误消息，抛出异常
        if error_messages:
            error_text = "\n".join(error_messages)
            raise InvalidRule(
                f"\nIncomplete rules found🚨\n\n{error_text}\n"
                f"Please note that if some slots or active loops should not be set "
                f"during prediction you need to explicitly set them to 'null' in the "
                f"rules."
            )

        logger.debug("Found no incompletions in rules.")

    @staticmethod
    def _get_slots_loops_from_states(
        trackers_as_states: List[List[State]],
    ) -> Tuple[Set[Text], Set[Text]]:
        """从状态中获取槽位和循环。
        
        Args:
            trackers_as_states: 跟踪器作为状态列表的表示
            
        Returns:
            槽位和循环的元组
        """
        slots = set()
        loops = set()
        # 遍历所有跟踪器状态
        for states in trackers_as_states:
            for state in states:
                # 更新槽位集合
                slots.update(set(state.get(SLOTS, {}).keys()))
                # FIXME: 理想情况下我们有更好的State注解，TypedDict
                # 可以工作但mypy支持非常有限。Dataclass是另一个选项
                active_loop = cast(Text, state.get(ACTIVE_LOOP, {}).get(LOOP_NAME))
                if active_loop:
                    loops.add(active_loop)
        return slots, loops

    def _find_rule_only_slots_loops(
        self,
        rule_trackers_as_states: List[List[State]],  # 规则跟踪器作为状态列表
        story_trackers_as_states: List[List[State]], # 故事跟踪器作为状态列表
    ) -> Tuple[List[Text], List[Text]]:
        """查找仅规则使用的槽位和循环。
        
        Args:
            rule_trackers_as_states: 规则跟踪器作为状态列表
            story_trackers_as_states: 故事跟踪器作为状态列表
            
        Returns:
            仅规则使用的槽位和循环列表
        """
        # 获取规则中的槽位和循环
        rule_slots, rule_loops = self._get_slots_loops_from_states(
            rule_trackers_as_states
        )
        # 获取故事中的槽位和循环
        story_slots, story_loops = self._get_slots_loops_from_states(
            story_trackers_as_states
        )

        # 集合不是JSON可序列化的，所以转换为列表
        return (
            list(rule_slots - story_slots - {SHOULD_NOT_BE_SET}),
            list(rule_loops - story_loops - {SHOULD_NOT_BE_SET}),
        )

    def _predict_next_action(
        self, tracker: TrackerWithCachedStates, domain: Domain
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """预测下一个动作。
        
        Args:
            tracker: 带缓存的跟踪器
            domain: 领域对象
            
        Returns:
            预测的动作名称和预测源
        """
        prediction, prediction_source = self._predict(tracker, domain)
        probabilities = prediction.probabilities
        # 如果RulePolicy没有为故事预测任何内容，不要抛出错误；
        # 但是对于规则，RulePolicy应该总是预测一个动作
        predicted_action_name = None
        if (
            probabilities != self._default_predictions(domain)
            or tracker.is_rule_tracker
        ):
            predicted_action_name = domain.action_names_or_texts[
                np.argmax(probabilities)
            ]

        return predicted_action_name, prediction_source

    def _predicted_action_name(
        self, tracker: TrackerWithCachedStates, domain: Domain, gold_action_name: Text
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """获取预测的动作名称。
        
        Args:
            tracker: 带缓存的跟踪器
            domain: 领域对象
            gold_action_name: 黄金动作名称
            
        Returns:
            预测的动作名称和预测源
        """
        predicted_action_name, prediction_source = self._predict_next_action(
            tracker, domain
        )
        # 如果有活跃循环，
        # RulePolicy总是首先预测active_loop，
        # 但在循环不愉快路径内部可能有另一个动作
        if (
            tracker.active_loop_name
            and predicted_action_name != gold_action_name
            and predicted_action_name == tracker.active_loop_name
        ):
            # 模拟循环拒绝
            rasa.core.test.emulate_loop_rejection(tracker)
            predicted_action_name, prediction_source = self._predict_next_action(
                tracker, domain
            )

        return predicted_action_name, prediction_source

    def _collect_sources(
        self,
        tracker: TrackerWithCachedStates,      # 带缓存的跟踪器
        predicted_action_name: Optional[Text], # 预测的动作名称
        gold_action_name: Optional[Text],      # 黄金动作名称
        prediction_source: Text,               # 预测源
    ) -> None:
        """收集规则源信息。
        
        Args:
            tracker: 带缓存的跟踪器
            predicted_action_name: 预测的动作名称
            gold_action_name: 黄金动作名称
            prediction_source: 预测源
        """
        # 我们需要记住规则应该预测哪个动作
        # 以便正确输出矛盾规则的名称
        rule_name = tracker.sender_id

        if prediction_source is not None and (
            prediction_source.startswith(DEFAULT_RULES)
            or prediction_source.startswith(LOOP_RULES)
        ):
            # 在这种情况下，真正的黄金动作与规则中的动作矛盾
            gold_action_name = predicted_action_name
            rule_name = prediction_source

        self._rules_sources[prediction_source].append((rule_name, gold_action_name))

    @staticmethod
    def _default_sources() -> Set[Text]:
        """获取默认源集合。
        
        Returns:
            默认源集合
        """
        return {
            DEFAULT_RULES + default_intent
            for default_intent in DEFAULT_ACTION_MAPPINGS.keys()
        }

    @staticmethod
    def _handling_loop_sources(domain: Domain) -> Set[Text]:
        """获取循环处理源集合。
        
        Args:
            domain: 领域对象
            
        Returns:
            循环处理源集合
        """
        loop_sources = set()
        # 遍历所有表单名称
        for loop_name in domain.form_names:
            loop_sources.add(LOOP_RULES + loop_name)
            loop_sources.add(
                LOOP_RULES + loop_name + LOOP_RULES_SEPARATOR + ACTION_LISTEN_NAME
            )
        return loop_sources

    def _should_delete(
        self,
        prediction_source: Text,               # 预测源
        tracker: TrackerWithCachedStates,      # 带缓存的跟踪器
        predicted_action_name: Text,           # 预测的动作名称
    ) -> bool:
        """检查此矛盾是否由于动作、意图对造成。

        Args:
            prediction_source: 导致预测的状态
            tracker: 引发矛盾的跟踪器
            predicted_action_name: 预测的动作名称

        Returns:
            如果矛盾是规则中动作、意图对的结果，则返回true
        """
        if (
            # 只适用于矛盾的故事，不适用于规则
            tracker.is_rule_tracker
            # 只适用于不可预测动作后的预测
            or prediction_source.count(PREVIOUS_ACTION) > 1
            # 只适用于action_listen的预测
            or predicted_action_name != ACTION_LISTEN_NAME
        ):
            return False
        # 遍历规则查找表
        for source in self.lookup[RULES]:
            # 只有在action_listen后预测了另一个动作时才删除规则
            if (
                source.startswith(prediction_source[:-2])
                and not prediction_source == source
            ):
                return True
        return False

    def _check_prediction(
        self,
        tracker: TrackerWithCachedStates,      # 带缓存的跟踪器
        predicted_action_name: Optional[Text], # 预测的动作名称
        gold_action_name: Text,                # 黄金动作名称
        prediction_source: Optional[Text],     # 预测源
    ) -> List[Text]:
        """检查预测是否正确。
        
        Args:
            tracker: 带缓存的跟踪器
            predicted_action_name: 预测的动作名称
            gold_action_name: 黄金动作名称
            prediction_source: 预测源
            
        Returns:
            错误消息列表
        """
        # FIXME: `predicted_action_name`和`prediction_source`要么都是None，
        # 要么都定义。这可以通过更好的类型注解来改进，但需要一些重构
        if (
            not predicted_action_name
            or not prediction_source
            or predicted_action_name == gold_action_name
        ):
            return []

        # 检查是否应该删除规则
        if self._should_delete(prediction_source, tracker, predicted_action_name):
            self.lookup[RULES].pop(prediction_source)
            return []

        # 确定跟踪器类型
        tracker_type = "rule" if tracker.is_rule_tracker else "story"
        # 查找矛盾的规则
        contradicting_rules = {
            rule_name
            for rule_name, action_name in self._rules_sources[prediction_source]
            if action_name != gold_action_name
        }

        if not contradicting_rules:
            return []

        # 构建错误消息
        error_message = (
            f"- the prediction of the action '{gold_action_name}' in {tracker_type} "
            f"'{tracker.sender_id}' "
            f"is contradicting with rule(s) '{', '.join(contradicting_rules)}'"
        )
        # 输出预测动作'action_default_fallback'是令人困惑的
        if predicted_action_name != self._fallback_action_name:
            error_message += f" which predicted action '{predicted_action_name}'"

        return [error_message + "."]

    def _run_prediction_on_trackers(
        self,
        trackers: List[TrackerWithCachedStates],  # 跟踪器列表
        domain: Domain,                           # 领域对象
        collect_sources: bool,                    # 是否收集源信息
    ) -> Tuple[List[Text], Set[Optional[Text]]]:
        """在跟踪器上运行预测。
        
        Args:
            trackers: 跟踪器列表
            domain: 领域对象
            collect_sources: 是否收集源信息
            
        Returns:
            错误消息列表和故事中使用的规则集合
        """
        if collect_sources:
            self._rules_sources = defaultdict(list)

        error_messages = []
        rules_used_in_stories = set()
        # 创建进度条
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        # 遍历所有跟踪器
        for tracker in pbar:
            running_tracker = tracker.init_copy()
            running_tracker.sender_id = tracker.sender_id
            # 第一个动作总是不可预测的
            next_action_is_unpredictable = True
            # 遍历所有应用的事件
            for event in tracker.applied_events():
                if not isinstance(event, ActionExecuted):
                    running_tracker.update(event)
                    continue

                if event.action_name == RULE_SNIPPET_ACTION_NAME:
                    # 通知RULE_SNIPPET_ACTION_NAME之后的动作是不可预测的
                    next_action_is_unpredictable = True
                    running_tracker.update(event)
                    continue

                # 不要在不可预测的动作上运行预测
                if next_action_is_unpredictable or event.unpredictable:
                    next_action_is_unpredictable = False  # 重置不可预测性
                    running_tracker.update(event)
                    continue

                # 获取黄金动作名称
                gold_action_name = event.action_name or event.action_text
                # 获取预测的动作名称
                predicted_action_name, prediction_source = self._predicted_action_name(
                    running_tracker, domain, gold_action_name
                )
                if collect_sources:
                    if prediction_source:
                        self._collect_sources(
                            running_tracker,
                            predicted_action_name,
                            gold_action_name,
                            prediction_source,
                        )
                else:
                    # 为了能够从对话历史中只删除规则轮次
                    # 对于ML策略，
                    # 我们需要知道哪些规则在ML跟踪器中使用
                    if (
                        not tracker.is_rule_tracker
                        and predicted_action_name == gold_action_name
                    ):
                        rules_used_in_stories.add(prediction_source)

                    error_messages += self._check_prediction(
                        running_tracker,
                        predicted_action_name,
                        gold_action_name,
                        prediction_source,
                    )

                running_tracker.update(event)

        return error_messages, rules_used_in_stories

    def _collect_rule_sources(
        self, rule_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> None:
        """收集规则源信息。
        
        Args:
            rule_trackers: 规则跟踪器列表
            domain: 领域对象
        """
        self._run_prediction_on_trackers(rule_trackers, domain, collect_sources=True)

    def _find_contradicting_and_used_in_stories_rules(
        self, trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> Tuple[List[Text], Set[Optional[Text]]]:
        """查找矛盾的规则和故事中使用的规则。
        
        Args:
            trackers: 跟踪器列表
            domain: 领域对象
            
        Returns:
            错误消息列表和故事中使用的规则集合
        """
        return self._run_prediction_on_trackers(trackers, domain, collect_sources=False)

    def _analyze_rules(
        self,
        rule_trackers: List[TrackerWithCachedStates],  # 规则跟踪器列表
        all_trackers: List[TrackerWithCachedStates],   # 所有跟踪器列表
        domain: Domain,                                 # 领域对象
    ) -> List[Text]:
        """通过运行预测分析学习的规则。

        此方法收集矛盾规则的错误消息
        并创建不在故事中的规则的查找表。

        Args:
            rule_trackers: 规则跟踪器列表
            all_trackers: 所有跟踪器列表
            domain: 领域对象

        Returns:
            不在故事中的规则列表
        """
        logger.debug("Started checking rules and stories for contradictions.")
        # 在训练期间，我们运行`predict_action_probabilities`来检查矛盾规则
        # 我们静默预测调试以避免在这些检查期间产生太多日志
        logger_level = logger.level
        logger.setLevel(logging.WARNING)

        # 我们需要在规则跟踪器上运行预测两次，因为我们需要收集
        # 关于哪些规则片段贡献给学习规则的信息
        self._collect_rule_sources(rule_trackers, domain)
        (
            error_messages,
            rules_used_in_stories,
        ) = self._find_contradicting_and_used_in_stories_rules(all_trackers, domain)

        logger.setLevel(logger_level)  # 重置日志级别
        if error_messages:
            error_text = "\n".join(error_messages)
            raise InvalidRule(
                f"\nContradicting rules or stories found 🚨\n\n{error_text}\n"
                f"Please update your stories and rules so that they don't contradict "
                f"each other."
            )

        logger.debug("Found no contradicting rules.")
        # 获取所有规则
        all_rules = (
            set(self._rules_sources.keys())
            | self._default_sources()
            | self._handling_loop_sources(domain)
        )
        # 集合不是JSON可序列化的，所以转换为列表
        return list(all_rules - rules_used_in_stories)

    def _create_lookup_from_trackers(
        self,
        rule_trackers: List[TrackerWithCachedStates],    # 规则跟踪器列表
        story_trackers: List[TrackerWithCachedStates],   # 故事跟踪器列表
        domain: Domain,                                   # 领域对象
    ) -> None:
        """从跟踪器创建查找表。
        
        Args:
            rule_trackers: 规则跟踪器列表
            story_trackers: 故事跟踪器列表
            domain: 领域对象
        """
        # 获取规则跟踪器的状态和动作
        (
            rule_trackers_as_states,
            rule_trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(
            rule_trackers, domain, omit_unset_slots=True
        )

        # 从状态创建规则查找表
        rules_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )
        # 移除规则片段预测
        self.lookup[RULES] = self._remove_rule_snippet_predictions(rules_lookup)

        # 获取故事跟踪器的状态和动作
        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(story_trackers, domain)

        # 如果检查矛盾，查找仅规则使用的槽位和循环
        if self._check_for_contradictions:
            (
                self.lookup[RULE_ONLY_SLOTS],
                self.lookup[RULE_ONLY_LOOPS],
            ) = self._find_rule_only_slots_loops(
                rule_trackers_as_states, story_trackers_as_states
            )

        # 使用所有跟踪器在不愉快路径中查找负规则
        trackers_as_states = rule_trackers_as_states + story_trackers_as_states
        trackers_as_actions = rule_trackers_as_actions + story_trackers_as_actions

        # 负规则不是反规则，它们是实际规则的辅助
        self.lookup[
            RULES_FOR_LOOP_UNHAPPY_PATH
        ] = self._create_loop_unhappy_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],  # 训练跟踪器列表
        domain: Domain,                                     # 领域对象
        **kwargs: Any,                                      # 其他关键字参数
    ) -> Resource:
        """在给定的训练跟踪器上训练策略。

        Args:
            training_trackers: 跟踪器列表
            domain: 领域对象

        Returns:
            可用于加载训练策略的资源
        """
        # 检查策略是否与领域兼容
        self.raise_if_incompatible_with_domain(self.config, domain)

        # 只考虑原始跟踪器（没有增强的）
        training_trackers = [
            t for t in training_trackers if not getattr(t, "is_augmented", False)
        ]
        # 来自基于规则的训练数据的跟踪器
        rule_trackers = [t for t in training_trackers if t.is_rule_tracker]
        # 如果限制规则，检查规则限制
        if self.config["restrict_rules"]:
            self._check_rule_restriction(rule_trackers)
        # 如果检查矛盾，检查不完整的规则
        if self._check_for_contradictions:
            self._check_for_incomplete_rules(rule_trackers, domain)

        # 来自基于机器学习的训练数据的跟踪器
        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]

        # 从跟踪器创建查找表
        self._create_lookup_from_trackers(rule_trackers, story_trackers, domain)

        # 使这可配置，因为检查可能需要很多时间
        if self._check_for_contradictions:
            # 在这里使用跟踪器可能不是最有效的方式，但是
            # 它允许我们直接测试`predict_action_probabilities`方法
            self.lookup[RULES_NOT_IN_STORIES] = self._analyze_rules(
                rule_trackers, training_trackers, domain
            )

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

        # 持久化策略
        self.persist()

        return self._resource

    @staticmethod
    def _does_rule_match_state(rule_state: State, conversation_state: State) -> bool:
        """检查规则状态是否匹配对话状态。
        
        Args:
            rule_state: 规则状态
            conversation_state: 对话状态
            
        Returns:
            如果规则状态匹配对话状态则返回True
        """
        # 遍历规则状态的每个状态类型
        for state_type, rule_sub_state in rule_state.items():
            conversation_sub_state = conversation_state.get(state_type, {})
            # 遍历规则子状态的每个键值对
            for key, value_from_rules in rule_sub_state.items():
                if isinstance(value_from_rules, list):
                    # json dumps和loads将元组作为列表，
                    # 所以我们需要将它们转换回来
                    value_from_rules = tuple(value_from_rules)
                value_from_conversation = conversation_sub_state.get(key)
                if (
                    # 值应该被设置，因此
                    # 检查它是否与状态中的相同
                    value_from_rules
                    and value_from_rules != SHOULD_NOT_BE_SET
                    and value_from_conversation != value_from_rules
                ) or (
                    # 值不应该被设置，因此
                    # 它应该是None或在状态中不存在
                    value_from_rules == SHOULD_NOT_BE_SET
                    and value_from_conversation
                    # 在训练期间提供`SHOULD_NOT_BE_SET`。因此，我们
                    # 也必须检查槽状态的值
                    and value_from_conversation != SHOULD_NOT_BE_SET
                ):
                    return False

        return True

    @staticmethod
    # 这个函数被调用很多次（例如检查矛盾），所以我们缓存其结果
    @functools.lru_cache(maxsize=1000)
    def _rule_key_to_state(rule_key: Text) -> List[State]:
        """将规则键转换为状态列表。
        
        Args:
            rule_key: 规则键
            
        Returns:
            状态列表
        """
        return json.loads(rule_key)

    def _is_rule_applicable(
        self, rule_key: Text, turn_index: int, conversation_state: State
    ) -> bool:
        """检查规则是否在当前轮次的状态下满足。

        Args:
            rule_key: 学习规则的文本表示
            turn_index: 当前对话轮次的索引
            conversation_state: 对应于turn_index的状态

        Returns:
            表示规则是否适用于当前状态的布尔值
        """
        # turn_index向后追溯时间
        reversed_rule_states = list(reversed(self._rule_key_to_state(rule_key)))

        # 规则必须适用，因为我们（没有任何适用性问题）
        # 在对话历史中比规则的长度更远
        if turn_index >= len(reversed_rule_states):
            return True

        # 状态有前一个动作当且仅当它不是对话开始状态
        current_previous_action = conversation_state.get(PREVIOUS_ACTION)
        rule_previous_action = reversed_rule_states[turn_index].get(PREVIOUS_ACTION)

        # 当前对话状态和规则状态都是对话启动器
        # 任何设置了initial_value的槽位都必然在两个状态中，不需要检查
        if not rule_previous_action and not current_previous_action:
            return True

        # 当前规则状态是对话启动器（由于conversation_start: true）
        # 但当前对话状态不是
        # 或者
        # 当前对话状态是启动器
        # 但当前规则状态不是
        if not rule_previous_action or not current_previous_action:
            return False

        # 检查：当前规则状态特征存在于当前对话状态中
        return self._does_rule_match_state(
            reversed_rule_states[turn_index], conversation_state
        )

    def _get_possible_keys(
        self, lookup: Dict[Text, Text], states: List[State]
    ) -> Set[Text]:
        """获取可能的规则键。
        
        Args:
            lookup: 查找字典
            states: 状态列表
            
        Returns:
            可能的规则键集合
        """
        possible_keys = set(lookup.keys())
        # 从后往前遍历状态
        for i, state in enumerate(reversed(states)):
            # 查找对应于当前状态的规则键
            possible_keys = set(
                filter(
                    lambda _key: self._is_rule_applicable(_key, i, state), possible_keys
                )
            )
        return possible_keys

    @staticmethod
    def _find_action_from_default_actions(
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """从默认动作中查找动作。
        
        Args:
            tracker: 对话状态跟踪器
            
        Returns:
            动作名称和预测源的元组
        """
        if (
            not tracker.latest_action_name == ACTION_LISTEN_NAME
            or not tracker.latest_message
        ):
            return None, None

        # 获取意图名称
        intent_name = tracker.latest_message.intent.get(INTENT_NAME_KEY)
        if intent_name is None:
            return None, None

        # 获取默认动作名称
        default_action_name = DEFAULT_ACTION_MAPPINGS.get(intent_name)
        if default_action_name is None:
            return None, None

        logger.debug(f"Predicted default action '{default_action_name}'.")
        return (
            default_action_name,
            # 创建对应于`_default_sources()`中默认预测源之一的预测源
            DEFAULT_RULES + intent_name,
        )

    @staticmethod
    def _find_action_from_loop_happy_path(
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """从循环愉快路径中查找动作。
        
        Args:
            tracker: 对话状态跟踪器
            
        Returns:
            动作名称和预测源的元组
        """
        # 获取活跃循环名称
        active_loop_name = tracker.active_loop_name
        if active_loop_name is None:
            return None, None

        # 检查循环是否被拒绝
        active_loop_rejected = tracker.is_active_loop_rejected
        # 检查是否应该预测循环
        should_predict_loop = (
            not active_loop_rejected
            and tracker.latest_action
            and tracker.latest_action.get(ACTION_NAME) != active_loop_name
        )
        # 检查是否应该预测监听
        should_predict_listen = (
            not active_loop_rejected and tracker.latest_action_name == active_loop_name
        )

        if should_predict_loop:
            logger.debug(f"Predicted loop '{active_loop_name}'.")
            return active_loop_name, LOOP_RULES + active_loop_name

        # 如果循环动作成功运行，预测`action_listen`
        if should_predict_listen:
            logger.debug(
                f"Predicted '{ACTION_LISTEN_NAME}' after loop '{active_loop_name}'."
            )
            return (
                ACTION_LISTEN_NAME,
                (
                    f"{LOOP_RULES}{active_loop_name}"
                    f"{LOOP_RULES_SEPARATOR}{ACTION_LISTEN_NAME}"
                ),
            )

        return None, None

    def _find_action_from_rules(
        self,
        tracker: DialogueStateTracker,        # 当前对话跟踪器
        domain: Domain,                       # 当前模型的领域
        use_text_for_last_user_input: bool,   # 是否使用最后一个用户消息的文本
    ) -> Tuple[Optional[Text], Optional[Text], bool]:
        """基于记忆化规则预测下一个动作。

        Args:
            tracker: 当前对话跟踪器
            domain: 当前模型的领域
            use_text_for_last_user_input: 如果为`True`，则使用最后一个用户消息的文本
                进行预测。如果为`False`，则使用意图。

        Returns:
            预测的动作名称或文本（如果没有找到匹配规则则为`None`）、
            匹配规则的描述，以及如果循环动作在不愉快路径后被预测则为`True`的元组
        """
        if (
            use_text_for_last_user_input
            and not tracker.latest_action_name == ACTION_LISTEN_NAME
        ):
            # 只在用户发言后直接进行文本预测
            # 因为我们已经决定了是使用文本还是意图
            return None, None, False

        # 获取预测状态
        states = self._prediction_states(
            tracker,
            domain,
            use_text_for_last_user_input,
            rule_only_data=self._get_rule_only_data(),
        )

        current_states = self.format_tracker_states(states)
        structlogger.debug(
            "rule_policy.actions.find", current_states=copy.deepcopy(current_states)
        )

        # 跟踪我们是否从不愉快的循环路径返回。如果这变为`True`
        # 策略返回一个事件，通知循环动作它从不愉快的路径返回。
        # 例如，`FormAction`使用它来跳过在不愉快路径后第一次执行时的槽验证
        returning_from_unhappy_path = False

        # 获取可能的规则键
        rule_keys = self._get_possible_keys(self.lookup[RULES], states)
        predicted_action_name = None
        best_rule_key = ""
        if rule_keys:
            # 如果有几个规则，
            # 这意味着某个规则是另一个规则的子集
            # 因此我们选择最大长度的规则
            best_rule_key = max(rule_keys, key=len)
            predicted_action_name = self.lookup[RULES].get(best_rule_key)

        active_loop_name = tracker.active_loop_name
        if active_loop_name:
            # 查找循环不愉快路径的规则
            loop_unhappy_keys = self._get_possible_keys(
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH], states
            )
            # 可能有几个不愉快路径条件
            unhappy_path_conditions = [
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH].get(key)
                for key in loop_unhappy_keys
            ]

            # 检查预测action_listen的规则是否在循环内应用
            # 规则可能不会显式切换回循环
            # 因此，我们必须处理这种情况
            predicted_listen_from_general_rule = (
                predicted_action_name == ACTION_LISTEN_NAME
                and not get_active_loop_name(self._rule_key_to_state(best_rule_key)[-1])
            )
            if predicted_listen_from_general_rule:
                if DO_NOT_PREDICT_LOOP_ACTION not in unhappy_path_conditions:
                    # 负规则不包含对应于active_loop不应该被预测的事实的键
                    logger.debug(
                        f"Predicted loop '{active_loop_name}' by overwriting "
                        f"'{ACTION_LISTEN_NAME}' predicted by general rule."
                    )
                    return (
                        active_loop_name,
                        best_rule_key,
                        returning_from_unhappy_path,
                    )

                # 不预测任何东西
                predicted_action_name = None

            if LOOP_WAS_INTERRUPTED in unhappy_path_conditions:
                logger.debug(
                    "Returning from unhappy path. Loop will be notified that "
                    "it was interrupted."
                )
                returning_from_unhappy_path = True

        if predicted_action_name is not None:
            logger.debug(
                f"There is a rule for the next action '{predicted_action_name}'."
            )
        else:
            logger.debug("There is no applicable rule.")

        # 如果我们没有从规则中预测任何东西，那么从状态创建的特征键
        # 可以用作此状态将导致回退的指示器
        return (
            predicted_action_name,
            best_rule_key or self._create_feature_key(states),
            returning_from_unhappy_path,
        )

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,        # 对话状态跟踪器
        domain: Domain,                       # 领域对象
        rule_only_data: Optional[Dict[Text, Any]] = None,  # 仅规则数据
        **kwargs: Any,                        # 其他关键字参数
    ) -> PolicyPrediction:
        """预测下一个动作（更多信息请参见父类）。"""
        prediction, _ = self._predict(tracker, domain)
        return prediction

    def _predict(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[PolicyPrediction, Optional[Text]]:
        """执行预测。
        
        Args:
            tracker: 对话状态跟踪器
            domain: 领域对象
            
        Returns:
            策略预测和预测源的元组
        """
        # 从文本规则中查找动作
        (
            rules_action_name_from_text,
            prediction_source_from_text,
            returning_from_unhappy_path_from_text,
        ) = self._find_action_from_rules(
            tracker, domain, use_text_for_last_user_input=True
        )

        # Rasa开源默认动作覆盖任何东西。如果用户想要实现
        # 同样的效果，他们需要编写规则或确保他们的循环相应地拒绝
        (
            default_action_name,
            default_prediction_source,
        ) = self._find_action_from_default_actions(tracker)

        # 文本优先于意图包括默认，
        # 但是循环愉快路径优先于规则预测
        if default_action_name and not rules_action_name_from_text:
            return (
                self._rule_prediction(
                    self._prediction_result(default_action_name, tracker, domain),
                    default_prediction_source,
                ),
                default_prediction_source,
            )

        # 循环优先于除默认之外的任何其他规则
        # 规则或任何其他预测只有在循环被拒绝时才会应用
        # 如果我们在循环中，并且循环之前没有运行或被拒绝，我们可以
        # 简单地强制预测循环
        (
            loop_happy_path_action_name,
            loop_happy_path_prediction_source,
        ) = self._find_action_from_loop_happy_path(tracker)
        if loop_happy_path_action_name:
            # 这个预测不使用用户输入
            # 无论如何，愉快的用户输入在特征化期间应该被忽略
            return (
                self._rule_prediction(
                    self._prediction_result(
                        loop_happy_path_action_name, tracker, domain
                    ),
                    loop_happy_path_prediction_source,
                    is_no_user_prediction=True,
                ),
                loop_happy_path_prediction_source,
            )

        # 首先从文本预测规则
        if rules_action_name_from_text:
            return (
                self._rule_prediction(
                    self._prediction_result(
                        rules_action_name_from_text, tracker, domain
                    ),
                    prediction_source_from_text,
                    returning_from_unhappy_path=returning_from_unhappy_path_from_text,
                    is_end_to_end_prediction=True,
                ),
                prediction_source_from_text,
            )

        # 从意图规则中查找动作
        (
            rules_action_name_from_intent,
            # 即使规则没有预测任何动作，我们也想记住源
            prediction_source_from_intent,
            returning_from_unhappy_path_from_intent,
        ) = self._find_action_from_rules(
            tracker, domain, use_text_for_last_user_input=False
        )
        if rules_action_name_from_intent:
            probabilities = self._prediction_result(
                rules_action_name_from_intent, tracker, domain
            )
        else:
            probabilities = self._default_predictions(domain)

        return (
            self._rule_prediction(
                probabilities,
                prediction_source_from_intent,
                returning_from_unhappy_path=(
                    # returning_from_unhappy_path是一个负条件，
                    # 所以应该应用`or`
                    returning_from_unhappy_path_from_text
                    or returning_from_unhappy_path_from_intent
                ),
                is_end_to_end_prediction=False,
            ),
            prediction_source_from_intent,
        )

    def _rule_prediction(
        self,
        probabilities: List[float],                    # 概率列表
        prediction_source: Text,                       # 预测源
        returning_from_unhappy_path: bool = False,     # 是否从不愉快路径返回
        is_end_to_end_prediction: bool = False,        # 是否为端到端预测
        is_no_user_prediction: bool = False,           # 是否为无用户预测
    ) -> PolicyPrediction:
        """创建规则预测。
        
        Args:
            probabilities: 概率列表
            prediction_source: 预测源
            returning_from_unhappy_path: 是否从不愉快路径返回
            is_end_to_end_prediction: 是否为端到端预测
            is_no_user_prediction: 是否为无用户预测
            
        Returns:
            策略预测对象
        """
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events=[LoopInterrupted(True)] if returning_from_unhappy_path else [],
            is_end_to_end_prediction=is_end_to_end_prediction,
            is_no_user_prediction=is_no_user_prediction,
            hide_rule_turn=(
                True
                if prediction_source in self.lookup.get(RULES_NOT_IN_STORIES, [])
                else False
            ),
        )

    def _default_predictions(self, domain: Domain) -> List[float]:
        """获取默认预测。
        
        Args:
            domain: 领域对象
            
        Returns:
            默认预测概率列表
        """
        result = super()._default_predictions(domain)

        # 如果启用回退预测，设置回退动作的置信度
        if self._enable_fallback_prediction:
            result[domain.index_for_action(self._fallback_action_name)] = self.config[
                "core_fallback_threshold"
            ]
        return result

    def persist(self) -> None:
        """持久化训练的策略。"""
        super().persist()
        with self._model_storage.write_to(self._resource) as directory:
            rule_only_data = self._get_rule_only_data()
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                directory / "rule_only_data.json", rule_only_data
            )

    def _metadata(self) -> Dict[Text, Any]:
        """获取元数据。
        
        Returns:
            包含查找表的元数据字典
        """
        return {"lookup": self.lookup}

    @classmethod
    def _metadata_filename(cls) -> Text:
        """获取元数据文件名。
        
        Returns:
            元数据文件名
        """
        return "rule_policy.json"

    def _get_rule_only_data(self) -> Dict[Text, Any]:
        """获取仅在规则数据中使用的槽位和循环。

        Returns:
            仅在规则数据中使用的槽位和循环
        """
        return {
            key: self.lookup.get(key, []) for key in [RULE_ONLY_SLOTS, RULE_ONLY_LOOPS]
        }
