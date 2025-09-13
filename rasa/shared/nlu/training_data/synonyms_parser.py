from typing import Any, Text, List, Dict

from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
)

# =============================================================================
# 同义词解析模块 - 提供实体同义词的解析和管理功能
# =============================================================================


def add_synonyms_from_entities(
    plain_text: Text, entities: List[Dict], existing_synonyms: Dict[Text, Any]
) -> None:
    """从意图示例中添加发现的同义词。

    Args:
        plain_text: 纯文本（已移除特殊符号）用户话语
        entities: 从原始用户话语中提取的实体
        existing_synonyms: 将扩展的现有同义词映射字典
    """
    for e in entities:
        e_text = plain_text[e[ENTITY_ATTRIBUTE_START] : e[ENTITY_ATTRIBUTE_END]]
        if e_text != e[ENTITY_ATTRIBUTE_VALUE]:
            add_synonym(e_text, e[ENTITY_ATTRIBUTE_VALUE], existing_synonyms)


def add_synonym(
    synonym_value: Text, synonym_name: Text, existing_synonyms: Dict[Text, Any]
) -> None:
    """向提供的同义词列表添加新的同义词映射。

    Args:
        synonym_value: 同义词的值
        synonym_name: 同义词的名称
        existing_synonyms: 将被扩展的同义词映射字典
    """
    import rasa.shared.nlu.training_data.util as training_data_util

    training_data_util.check_duplicate_synonym(
        existing_synonyms, synonym_value, synonym_name, "读取 markdown"
    )
    existing_synonyms[synonym_value] = synonym_name
