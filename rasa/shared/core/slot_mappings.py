# =============================================================================
# 槽位映射系统模块 - 定义槽位映射功能和验证
# =============================================================================
# 此模块定义了 Rasa Core 中的槽位映射系统，包括槽位映射的验证、
# 意图匹配、实体匹配等功能。槽位映射用于自动填充槽位值。

# 标准库导入
from typing import Text, Dict, Any, List, Optional, TYPE_CHECKING  # 类型提示

# Rasa 内部模块导入
from rasa.shared.constants import DOCS_URL_SLOTS, IGNORED_INTENTS  # 共享常量
import rasa.shared.utils.io  # IO 工具函数
from rasa.shared.nlu.constants import (  # NLU 常量
    ENTITY_ATTRIBUTE_TYPE,    # 实体类型属性
    ENTITY_ATTRIBUTE_ROLE,    # 实体角色属性
    ENTITY_ATTRIBUTE_GROUP,   # 实体组属性
    INTENT,                   # 意图常量
    NOT_INTENT,              # 非意图常量
    INTENT_NAME_KEY,         # 意图名称键
)
from rasa.shared.core.constants import (  # Core 常量
    SLOT_MAPPINGS,           # 槽位映射常量
    MAPPING_TYPE,            # 映射类型常量
    SlotMappingType,         # 槽位映射类型枚举
    MAPPING_CONDITIONS,      # 映射条件常量
)

# 类型检查导入（避免循环导入）
if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker  # 对话状态跟踪器
    from rasa.shared.core.domain import Domain  # 域


# =============================================================================
# 槽位映射类定义
# =============================================================================

class SlotMapping:
    """定义可用槽位映射的功能。
    
    此类提供了槽位映射的验证、意图匹配、实体匹配等功能，
    用于自动填充槽位值。
    """

    @staticmethod
    def validate(mapping: Dict[Text, Any], slot_name: Text) -> None:
        """验证槽位映射。

        Args:
            mapping: 要验证的映射
            slot_name: 此映射映射的槽位名称

        Raises:
            InvalidDomain: 如果槽位映射无效
        """
        from rasa.shared.core.domain import InvalidDomain  # 导入域异常

        if not isinstance(mapping, dict):  # 如果映射不是字典
            raise InvalidDomain(
                f"请确保槽位 '{slot_name}' 的槽位映射在 "
                f"您的域中是有效的字典。请参阅 "
                f"{DOCS_URL_SLOTS} 获取更多信息。"
            )  # 抛出域异常

        try:
            mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))  # 获取映射类型
        except ValueError:  # 如果映射类型无效
            raise InvalidDomain(
                f"您的域对槽位 '{slot_name}' 使用了无效的槽位映射类型 "
                f"'{mapping.get(MAPPING_TYPE)}'。请参阅 "
                f"{DOCS_URL_SLOTS} 获取更多信息。"
            )  # 抛出域异常

        # 定义每种映射类型所需的键
        validations: Dict[SlotMappingType, List[Text]] = {
            SlotMappingType.FROM_ENTITY: ["entity"],           # 从实体映射需要entity键
            SlotMappingType.FROM_INTENT: ["value"],            # 从意图映射需要value键
            SlotMappingType.FROM_TRIGGER_INTENT: ["value"],    # 从触发意图映射需要value键
            SlotMappingType.FROM_TEXT: [],                     # 从文本映射不需要额外键
            SlotMappingType.CUSTOM: [],                        # 自定义映射不需要额外键
        }

        required_keys = validations[mapping_type]  # 获取所需键列表
        for required_key in required_keys:  # 遍历所需键
            if mapping.get(required_key) is None:  # 如果键值为None
                raise InvalidDomain(
                    f"您需要为槽位 '{slot_name}' 的 '{mapping_type}' 类型槽位映射 "
                    f"指定键 '{required_key}' 的值。请参阅 "
                    f"{DOCS_URL_SLOTS} 获取更多信息。"
                )  # 抛出域异常

    @staticmethod
    def _get_active_loop_ignored_intents(
        mapping: Dict[Text, Any], domain: "Domain", active_loop_name: Text
    ) -> List[Text]:
        """获取活动循环的忽略意图列表。
        
        Args:
            mapping: 槽位映射
            domain: 域对象
            active_loop_name: 活动循环名称
            
        Returns:
            忽略的意图列表
        """
        from rasa.shared.core.constants import ACTIVE_LOOP  # 导入活动循环常量

        mapping_conditions = mapping.get(MAPPING_CONDITIONS)  # 获取映射条件
        active_loop_match = True  # 初始化活动循环匹配标志
        ignored_intents = []  # 初始化忽略意图列表

        if mapping_conditions:  # 如果存在映射条件
            match_list = [
                condition.get(ACTIVE_LOOP) == active_loop_name  # 检查条件中的活动循环
                for condition in mapping_conditions
            ]
            active_loop_match = any(match_list)  # 检查是否有匹配的条件

        if active_loop_match:  # 如果活动循环匹配
            form_ignored_intents = domain.forms.get(active_loop_name, {}).get(
                IGNORED_INTENTS, []
            )  # 获取表单的忽略意图
            ignored_intents = SlotMapping.to_list(form_ignored_intents)  # 转换为列表

        return ignored_intents  # 返回忽略意图列表

    @staticmethod
    def intent_is_desired(
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker", domain: "Domain"
    ) -> bool:
        """检查用户意图是否匹配槽位映射的意图规范。
        
        Args:
            mapping: 槽位映射
            tracker: 对话状态跟踪器
            domain: 域对象
            
        Returns:
            如果意图匹配则返回True，否则返回False
        """
        mapping_intents = SlotMapping.to_list(mapping.get(INTENT, []))  # 获取映射意图列表
        mapping_not_intents = SlotMapping.to_list(mapping.get(NOT_INTENT, []))  # 获取映射非意图列表

        active_loop_name = tracker.active_loop_name  # 获取活动循环名称
        if active_loop_name:  # 如果存在活动循环
            mapping_not_intents = (
                mapping_not_intents
                + SlotMapping._get_active_loop_ignored_intents(
                    mapping, domain, active_loop_name
                )
            )  # 添加活动循环的忽略意图

        if tracker.latest_message:  # 如果存在最新消息
            intent = tracker.latest_message.intent.get(INTENT_NAME_KEY)  # 获取意图名称
        else:
            intent = None  # 设置为None

        # 检查意图是否未被阻止（没有指定意图且不在非意图列表中）
        intent_not_blocked = not mapping_intents and intent not in set(
            mapping_not_intents
        )

        return intent_not_blocked or intent in mapping_intents  # 返回意图匹配结果

    # =============================================================================
    # 辅助方法
    # =============================================================================
    
    @staticmethod
    def to_list(x: Optional[Any]) -> List[Any]:
        """如果对象不是列表，则将其转换为列表。
        
        Args:
            x: 要转换的对象
            
        Returns:
            转换后的列表
        """
        if x is None:  # 如果对象为None
            x = []  # 设置为空列表
        elif not isinstance(x, list):  # 如果对象不是列表
            x = [x]  # 将对象包装在列表中

        return x  # 返回列表

    @staticmethod
    def entity_is_desired(
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker"
    ) -> bool:
        """检查槽位是否应该由输入中的实体填充。

        Args:
            mapping: 槽位映射
            tracker: 跟踪器

        Returns:
            如果槽位应该被填充则返回True，否则返回False
        """
        slot_fulfils_entity_mapping = False  # 初始化槽位满足实体映射标志
        if tracker.latest_message:  # 如果存在最新消息
            extracted_entities = tracker.latest_message.entities  # 获取提取的实体
        else:
            extracted_entities = []  # 设置为空列表

        for entity in extracted_entities:  # 遍历提取的实体
            if (
                mapping.get(ENTITY_ATTRIBUTE_TYPE) == entity[ENTITY_ATTRIBUTE_TYPE]  # 检查实体类型
                and mapping.get(ENTITY_ATTRIBUTE_ROLE)
                == entity.get(ENTITY_ATTRIBUTE_ROLE)  # 检查实体角色
                and mapping.get(ENTITY_ATTRIBUTE_GROUP)
                == entity.get(ENTITY_ATTRIBUTE_GROUP)  # 检查实体组
            ):
                # 获取匹配的实体值
                matching_values = tracker.get_latest_entity_values(
                    mapping.get(ENTITY_ATTRIBUTE_TYPE),
                    mapping.get(ENTITY_ATTRIBUTE_ROLE),
                    mapping.get(ENTITY_ATTRIBUTE_GROUP),
                )
                slot_fulfils_entity_mapping = matching_values is not None  # 设置满足标志
                break  # 跳出循环

        return slot_fulfils_entity_mapping  # 返回满足标志

    @staticmethod
    def check_mapping_validity(
        slot_name: Text,
        mapping_type: SlotMappingType,
        mapping: Dict[Text, Any],
        domain: "Domain",
    ) -> bool:
        """检查映射的有效性。

        Args:
            slot_name: 要验证的槽位名称
            mapping_type: 槽位映射的类型
            mapping: 槽位映射
            domain: 要检查的域

        Returns:
            如果映射中指定的意图和实体在域中存在则返回True
        """
        # 检查实体映射的有效性
        if (
            mapping_type == SlotMappingType.FROM_ENTITY  # 如果是从实体映射
            and mapping.get(ENTITY_ATTRIBUTE_TYPE) not in domain.entities  # 且实体不在域中
        ):
            rasa.shared.utils.io.raise_warning(
                f"槽位 '{slot_name}' 使用 'from_entity' 映射 "
                f"引用不存在的实体 '{mapping.get(ENTITY_ATTRIBUTE_TYPE)}'。 "
                f"由于映射无效，跳过槽位提取。"
            )  # 发出警告
            return False  # 返回False

        # 检查意图映射的有效性
        if (
            mapping_type == SlotMappingType.FROM_INTENT  # 如果是从意图映射
            and mapping.get(INTENT) is not None  # 且指定了意图
        ):
            intent_list = SlotMapping.to_list(mapping.get(INTENT))  # 获取意图列表
            for intent in intent_list:  # 遍历意图列表
                if intent and intent not in domain.intents:  # 如果意图不在域中
                    rasa.shared.utils.io.raise_warning(
                        f"槽位 '{slot_name}' 使用 'from_intent' 映射引用 "
                        f"不存在的意图 '{mapping.get('intent')}'。 "
                        f"由于映射无效，跳过槽位提取。"
                    )  # 发出警告
                    return False  # 返回False

        return True  # 返回True


# =============================================================================
# 槽位映射验证函数
# =============================================================================

def validate_slot_mappings(domain_slots: Dict[Text, Any]) -> None:
    """如果槽位映射无效则抛出InvalidDomain异常。
    
    Args:
        domain_slots: 域槽位字典
        
    Raises:
        InvalidDomain: 如果槽位映射无效
    """
    # 发出关于槽位自动填充已移除的警告
    rasa.shared.utils.io.raise_warning(
        f"槽位自动填充已在3.0版本中移除，并被新的 "
        f"显式槽位设置机制取代。 "
        f"请参阅 {DOCS_URL_SLOTS} 了解更多信息。",
        UserWarning,
    )

    for slot_name, properties in domain_slots.items():  # 遍历域槽位
        mappings = properties.get(SLOT_MAPPINGS)  # 获取槽位映射

        for slot_mapping in mappings:  # 遍历槽位映射
            SlotMapping.validate(slot_mapping, slot_name)  # 验证槽位映射
