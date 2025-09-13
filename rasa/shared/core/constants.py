from __future__ import annotations
from enum import Enum

import rasa.shared.constants as constants

# =============================================================================
# Core 模块常量定义 - 定义 Core 对话管理相关的常量
# =============================================================================
# 此模块包含 Core 模块中使用的所有常量定义，包括默认意图、
# 动作名称、槽位映射类型、状态键等。这些常量为整个
# Core 对话管理系统提供了统一的配置和标识符。


# =============================================================================
# 默认槽位和意图常量
# =============================================================================
DEFAULT_CATEGORICAL_SLOT_VALUE = "__other__"  # 分类槽位的默认值

# 用户意图常量
USER_INTENT_RESTART = "restart"                    # 重启意图
USER_INTENT_BACK = "back"                          # 返回意图
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"          # 超出范围意图
USER_INTENT_SESSION_START = "session_start"        # 会话开始意图
SESSION_START_METADATA_SLOT = "session_started_metadata"  # 会话开始元数据槽位

# 默认意图列表
DEFAULT_INTENTS = [
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_OUT_OF_SCOPE,
    USER_INTENT_SESSION_START,
    constants.DEFAULT_NLU_FALLBACK_INTENT_NAME,  # NLU 回退意图名称
]

# =============================================================================
# 循环和动作常量
# =============================================================================
LOOP_NAME = "name"  # 循环名称键

# 默认动作名称常量
ACTION_LISTEN_NAME = "action_listen"                              # 监听动作
ACTION_RESTART_NAME = "action_restart"                            # 重启动作
ACTION_SESSION_START_NAME = "action_session_start"                # 会话开始动作
ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"          # 默认回退动作
ACTION_DEACTIVATE_LOOP_NAME = "action_deactivate_loop"            # 停用循环动作
ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"  # 撤销回退事件动作
ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"  # 默认询问确认动作
ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"        # 默认询问重述动作
ACTION_BACK_NAME = "action_back"                                  # 返回动作
ACTION_TWO_STAGE_FALLBACK_NAME = "action_two_stage_fallback"      # 两阶段回退动作
ACTION_UNLIKELY_INTENT_NAME = "action_unlikely_intent"            # 不太可能的意图动作
RULE_SNIPPET_ACTION_NAME = "..."                                  # 规则片段动作
ACTION_EXTRACT_SLOTS = "action_extract_slots"                     # 提取槽位动作
ACTION_VALIDATE_SLOT_MAPPINGS = "action_validate_slot_mappings"   # 验证槽位映射动作

# 默认动作名称列表
DEFAULT_ACTION_NAMES = [
    ACTION_LISTEN_NAME,                              # 监听动作
    ACTION_RESTART_NAME,                             # 重启动作
    ACTION_SESSION_START_NAME,                       # 会话开始动作
    ACTION_DEFAULT_FALLBACK_NAME,                    # 默认回退动作
    ACTION_DEACTIVATE_LOOP_NAME,                     # 停用循环动作
    ACTION_REVERT_FALLBACK_EVENTS_NAME,              # 撤销回退事件动作
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,             # 默认询问确认动作
    ACTION_DEFAULT_ASK_REPHRASE_NAME,                # 默认询问重述动作
    ACTION_TWO_STAGE_FALLBACK_NAME,                  # 两阶段回退动作
    ACTION_UNLIKELY_INTENT_NAME,                     # 不太可能的意图动作
    ACTION_BACK_NAME,                                # 返回动作
    RULE_SNIPPET_ACTION_NAME,                        # 规则片段动作
    ACTION_EXTRACT_SLOTS,                            # 提取槽位动作
]

# 动作配置常量
ACTION_SHOULD_SEND_DOMAIN = "send_domain"  # 动作是否应发送域

# =============================================================================
# 状态和循环相关常量
# =============================================================================
# 规则允许将槽位或活动循环的值设置为 None；
# 生成器用此常量替换 `None` 以通知规则策略
# 在预测期间不应设置值以激活规则
SHOULD_NOT_BE_SET = "should_not_be_set"  # 不应设置常量

# 状态键常量
PREVIOUS_ACTION = "prev_action"        # 前一动作
ACTIVE_LOOP = "active_loop"            # 活动循环
LOOP_INTERRUPTED = "is_interrupted"    # 循环中断
LOOP_REJECTED = "rejected"             # 循环拒绝
TRIGGER_MESSAGE = "trigger_message"    # 触发消息
FOLLOWUP_ACTION = "followup_action"    # 后续动作

# =============================================================================
# 消息和外部事件常量
# =============================================================================
# 特殊用户消息部分的开始
EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "  # 外部消息前缀
# 访问事件元数据中数据的键
# 它指定事件是否由外部实体（例如传感器）引起
IS_EXTERNAL = "is_external"  # 是否为外部事件

# 动作和发送者相关常量
ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"  # 动作名称发送者ID连接符

REQUESTED_SLOT = "requested_slot"  # 请求的槽位

# =============================================================================
# 知识库相关常量
# =============================================================================
# 知识库的槽位
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"        # 列出的项目槽位
SLOT_LAST_OBJECT = "knowledge_base_last_object"            # 最后对象槽位
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"  # 最后对象类型槽位
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"  # 默认知识库查询动作

# 默认槽位名称集合
DEFAULT_SLOT_NAMES = {
    REQUESTED_SLOT,              # 请求的槽位
    SESSION_START_METADATA_SLOT, # 会话开始元数据槽位
    SLOT_LISTED_ITEMS,           # 列出的项目槽位
    SLOT_LAST_OBJECT,            # 最后对象槽位
    SLOT_LAST_OBJECT_TYPE,       # 最后对象类型槽位
}


# =============================================================================
# 槽位映射相关常量
# =============================================================================
SLOT_MAPPINGS = "mappings"        # 槽位映射
MAPPING_CONDITIONS = "conditions"  # 映射条件
MAPPING_TYPE = "type"              # 映射类型


class SlotMappingType(Enum):
    """槽位映射类型枚举。
    
    定义了槽位值可以从哪些来源获取。
    """

    FROM_ENTITY = "from_entity"                    # 从实体获取
    FROM_INTENT = "from_intent"                     # 从意图获取
    FROM_TRIGGER_INTENT = "from_trigger_intent"     # 从触发意图获取
    FROM_TEXT = "from_text"                         # 从文本获取
    CUSTOM = "custom"                               # 自定义映射

    def __str__(self) -> str:
        """返回应在配置文件中使用的字符串表示。
        
        Returns:
            映射类型的字符串值
        """
        return self.value

    def is_predefined_type(self) -> bool:
        """返回映射类型是否为预定义类型。
        
        也就是说，评估映射不需要自定义动作执行。
        
        Returns:
            如果是预定义类型则返回True，否则返回False
        """
        return self != SlotMappingType.CUSTOM


# =============================================================================
# 状态和特征化相关常量
# =============================================================================
# `State` 的键（USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP）
# 表示 `SubState` 的来源
USER = "user"    # 用户状态
SLOTS = "slots"  # 槽位状态

# 特征化相关常量
USE_TEXT_FOR_FEATURIZATION = "use_text_for_featurization"  # 是否使用文本进行特征化
ENTITY_LABEL_SEPARATOR = "#"                               # 实体标签分隔符

# 规则相关常量
RULE_ONLY_SLOTS = "rule_only_slots"    # 仅规则槽位
RULE_ONLY_LOOPS = "rule_only_loops"    # 仅规则循环

# =============================================================================
# 策略和分类器名称常量
# =============================================================================
# 如果添加更多策略/分类器名称，请确保添加测试以确保
# 名称和类保持同步
POLICY_NAME_TWO_STAGE_FALLBACK = "TwoStageFallbackPolicy"  # 两阶段回退策略
POLICY_NAME_MAPPING = "MappingPolicy"                      # 映射策略
POLICY_NAME_FALLBACK = "FallbackPolicy"                    # 回退策略
POLICY_NAME_FORM = "FormPolicy"                            # 表单策略
POLICY_NAME_RULE = "RulePolicy"                            # 规则策略

CLASSIFIER_NAME_FALLBACK = "FallbackClassifier"            # 回退分类器

# 提取实体的策略集合
POLICIES_THAT_EXTRACT_ENTITIES = {"TEDPolicy"}             # 提取实体的策略
