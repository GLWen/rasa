# 导入深拷贝模块
import copy
# 导入正则表达式模块
import re
# 导入日志记录模块
import logging
# 导入结构化日志模块
import structlog
# 导入类型提示相关类型
from typing import Text, Dict, Union, Any, List

# 创建标准日志记录器
logger = logging.getLogger(__name__)
# 创建结构化日志记录器
structlogger = structlog.get_logger()


def interpolate_text(response: Text, values: Dict[Text, Text]) -> Text:
    """将值插值到带有占位符的响应中。

    将响应标签从 "{tag_name}" 转换为 "{0[tag_name]}"，如这里所述：
    https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
    阻止字符，确保不允许：
    (a) 槽位名称中的换行符
    (b) 槽位名称中的 { 或 }

    Args:
        response: 应该被插值的文本片段
        values: 键和这些键应该被替换的值的字典

    Returns:
        进行任何替换后的文本片段
    """
    try:
        # 使用正则表达式将 {tag_name} 格式转换为 {0[tag_name]} 格式
        text = re.sub(r"{([^\n{}]+?)}", r"{0[\1]}", response)
        # 使用format方法进行插值
        text = text.format(values)
        # 检查是否还有未替换的标签
        if "0[" in text:
            # 正则表达式替换了标签但format没有替换
            # 可能的原因是标签名称被双花括号包围
            # format函数只是转义了它
            # 我们不想返回 {0[SLOTNAME]} 因此
            # 恢复原始值，{ 被转义
            return response.format({})

        return text
    except KeyError as e:
        # 处理键错误异常
        event_info = (
            "指定的槽位名称不存在，"
            "在响应调用期间没有提供显式值。"
            "返回未填充的响应。"
        )
        # 记录结构化日志异常
        structlogger.exception(
            "interpolator.interpolate.text",
            response=copy.deepcopy(response),
            placeholder_key=e.args[0],
            event_info=event_info,
        )
        return response


def interpolate(
    response: Union[List[Any], Dict[Text, Any], Text], values: Dict[Text, Text]
) -> Union[List[Any], Dict[Text, Any], Text]:
    """递归处理响应并插值任何文本键。

    Args:
        response: 应该被插值的响应
        values: 键和这些键应该被替换的值的字典

    Returns:
        进行任何替换后的响应
    """
    # 如果响应是字符串，直接调用文本插值函数
    if isinstance(response, str):
        return interpolate_text(response, values)
    # 如果响应是字典，递归处理每个值
    elif isinstance(response, dict):
        for k, v in response.items():
            if isinstance(v, dict):
                # 如果值是字典，递归插值
                interpolate(v, values)
            elif isinstance(v, list):
                # 如果值是列表，对列表中的每个元素进行插值
                response[k] = [interpolate(i, values) for i in v]
            elif isinstance(v, str):
                # 如果值是字符串，进行文本插值
                response[k] = interpolate_text(v, values)
        return response
    # 如果响应是列表，对列表中的每个元素进行插值
    elif isinstance(response, list):
        return [interpolate(i, values) for i in response]
    # 如果响应是其他类型，直接返回
    return response
