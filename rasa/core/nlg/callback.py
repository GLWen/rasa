# 导入JSON处理模块
import json
# 导入日志记录模块
import logging
# 导入类型提示相关类型
from typing import List, Text, Any, Dict, Optional

# 导入Rasa核心常量
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
# 导入自然语言生成器和响应变体过滤器
from rasa.core.nlg.generator import NaturalLanguageGenerator, ResponseVariationFilter
# 导入对话状态跟踪器和事件详细程度
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
# 导入Rasa异常类
from rasa.shared.exceptions import RasaException
# 导入端点配置类
from rasa.utils.endpoints import EndpointConfig

# 创建日志记录器
logger = logging.getLogger(__name__)


def nlg_response_format_spec() -> Dict[Text, Any]:
    """NLG端点的预期响应模式。

    用于验证从NLG端点返回的响应。
    
    Returns:
        包含响应格式规范的字典
    """
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},                    # 文本内容
            "id": {"type": ["string", "null"]},            # 响应ID
            "buttons": {"type": ["array", "null"], "items": {"type": "object"}},  # 按钮列表
            "elements": {"type": ["array", "null"], "items": {"type": "object"}},  # 元素列表
            "attachment": {"type": ["object", "null"]},    # 附件对象
            "image": {"type": ["string", "null"]},         # 图片URL
            "custom": {"type": "object"},                  # 自定义内容
        },
    }


# 响应ID键常量
RESPONSE_ID_KEY = "response_ids"


def nlg_request_format(
    utter_action: Text,
    tracker: DialogueStateTracker,
    output_channel: Text,
    **kwargs: Any,
) -> Dict[Text, Any]:
    """为NLG请求创建JSON请求体。
    
    Args:
        utter_action: utter动作名称
        tracker: 对话状态跟踪器
        output_channel: 输出通道名称
        **kwargs: 其他关键字参数
        
    Returns:
        包含请求信息的字典
    """
    # 获取跟踪器的完整状态
    tracker_state = tracker.current_state(EventVerbosity.ALL)
    # 从关键字参数中提取响应ID
    response_id = kwargs.pop("response_id", None)

    return {
        "response": utter_action,           # utter动作名称
        "id": response_id,                  # 响应ID
        "arguments": kwargs,                # 其他参数
        "tracker": tracker_state,           # 跟踪器状态
        "channel": {"name": output_channel}, # 输出通道信息
    }


class CallbackNaturalLanguageGenerator(NaturalLanguageGenerator):
    """通过使用远程端点进行生成来生成机器人话语。

    生成器将为每个要生成的消息调用端点。端点需要响应
    格式正确的JSON。生成器将使用此消息为机器人创建响应。
    """

    def __init__(self, endpoint_config: EndpointConfig) -> None:
        """初始化回调自然语言生成器。
        
        Args:
            endpoint_config: 端点配置对象
        """
        # 存储NLG端点配置
        self.nlg_endpoint = endpoint_config

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """使用端点从领域检索命名响应。
        
        Args:
            utter_action: utter动作名称
            tracker: 对话状态跟踪器
            output_channel: 输出通道名称
            **kwargs: 其他关键字参数
            
        Returns:
            生成的响应字典
            
        Raises:
            RasaException: 当端点返回无效响应时抛出异常
        """
        # 从关键字参数中提取领域响应
        domain_responses = kwargs.pop("domain_responses", None)
        # 获取响应ID
        response_id = self.fetch_response_id(
            utter_action, tracker, output_channel, domain_responses
        )
        # 将响应ID添加到关键字参数中
        kwargs["response_id"] = response_id

        # 创建请求体
        body = nlg_request_format(utter_action, tracker, output_channel, **kwargs)

        # 记录调试信息
        logger.debug(
            "从 {} 请求NLG for {}。"
            "请求体是 {}。"
            "".format(utter_action, self.nlg_endpoint.url, json.dumps(body))
        )

        # 发送请求到NLG端点
        response = await self.nlg_endpoint.request(
            method="post", json=body, timeout=DEFAULT_REQUEST_TIMEOUT
        )

        # 记录接收到的响应
        logger.debug(f"接收到NLG响应: {json.dumps(response)}")

        # 验证响应格式
        if isinstance(response, dict) and self.validate_response(response):
            return response
        else:
            raise RasaException("NLG web端点返回了无效响应。")

    @staticmethod
    def validate_response(content: Optional[Dict[Text, Any]]) -> bool:
        """验证NLG响应。失败时抛出异常。
        
        Args:
            content: 要验证的响应内容
            
        Returns:
            如果验证通过则返回True
            
        Raises:
            RasaException: 当响应格式无效时抛出异常
        """
        # 导入JSON模式验证库
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            # 检查内容是否为空
            if content is None or content == "":
                # 表示端点不想响应任何内容
                return True
            else:
                # 使用JSON模式验证响应格式
                validate(content, nlg_response_format_spec())
                return True
        except ValidationError as e:
            # 验证失败时抛出异常
            raise RasaException(
                f"{e.message}. 无法验证来自API的NLG响应，请确保 "
                f"来自NLG端点的响应是有效的。"
                f"有关格式的更多信息，请查阅 "
                f"同一模块中的 `nlg_response_format_spec` 函数: "
                f"https://github.com/RasaHQ/rasa/blob/main/rasa/core/nlg/callback.py"
            )

    @staticmethod
    def fetch_response_id(
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        domain_responses: Optional[Dict[Text, List[Dict[Text, Any]]]],
    ) -> Optional[Text]:
        """获取utter动作的响应ID。

        响应ID是从领域响应中根据跟踪器状态和通道为utter动作检索的。
        
        Args:
            utter_action: utter动作名称
            tracker: 对话状态跟踪器
            output_channel: 输出通道名称
            domain_responses: 领域响应字典
            
        Returns:
            响应ID，如果无法获取则返回None
        """
        # 检查领域响应是否提供
        if domain_responses is None:
            logger.debug("无法获取响应ID。未提供响应。")
            return None

        # 创建响应变体过滤器
        response_filter = ResponseVariationFilter(domain_responses)
        # 获取响应变体ID
        response_id = response_filter.get_response_variation_id(
            utter_action, tracker, output_channel
        )

        # 如果无法获取响应ID，记录调试信息
        if response_id is None:
            logger.debug(f"无法为动作 '{utter_action}' 获取响应ID。")

        return response_id
