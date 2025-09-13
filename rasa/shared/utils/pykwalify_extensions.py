"""
此模块重新组织自定义验证函数，并作为 pykwalify 库的扩展加载：

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions
"""
from typing import Any, List, Dict, Text, Union

from pykwalify.errors import SchemaError

# =============================================================================
# Pykwalify 自定义验证函数扩展
# =============================================================================

def require_response_keys(
    responses: List[Dict[Text, Any]], _: Dict, __: Text
) -> Union[SchemaError, bool]:
    """验证响应字典具有 "text" 键或 "custom" 键。
    
    Args:
        responses: 响应字典列表
        _: 未使用的字典参数
        __: 未使用的文本参数
        
    Returns:
        SchemaError 如果验证失败，否则 True
    """
    for response in responses:
        if not isinstance(response, dict):
            # 这由其他验证规则处理
            continue

        if response.get("text") is None and not response.get("custom"):
            return SchemaError(
                "响应中缺少 'text' 或 'custom' 键或 "
                "响应中的 'text' 值为空。"
            )

    return True
