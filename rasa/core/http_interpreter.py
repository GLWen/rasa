# 第三方库导入
import aiohttp  # 异步HTTP客户端库

# 标准库导入
import copy  # 深拷贝和浅拷贝操作
import logging  # 日志记录模块
import structlog  # 结构化日志记录

# 类型提示导入
from typing import Text, Dict, Any, Optional  # 类型提示

# Rasa 内部模块导入
from rasa.core import constants  # 核心常量
from rasa.core.channels import UserMessage  # 用户消息类
from rasa.shared.nlu.constants import INTENT_NAME_KEY  # NLU常量
from rasa.utils.endpoints import EndpointConfig  # 端点配置类

# 日志记录器
logger = logging.getLogger(__name__)  # 标准日志记录器
structlogger = structlog.get_logger()  # 结构化日志记录器


class RasaNLUHttpInterpreter:
    """允许使用HTTP端点来解析消息。"""

    def __init__(self, endpoint_config: Optional[EndpointConfig] = None) -> None:
        """初始化一个 `RasaNLUHttpInterpreter` 实例。"""
        self.session = aiohttp.ClientSession()  # 创建异步HTTP客户端会话
        if endpoint_config:  # 如果提供了端点配置
            self.endpoint_config = endpoint_config  # 使用提供的配置
        else:  # 否则
            self.endpoint_config = EndpointConfig(constants.DEFAULT_SERVER_URL)  # 使用默认服务器URL

    async def parse(self, message: UserMessage) -> Dict[Text, Any]:
        """解析文本消息。

        如果文本解析失败，返回默认值。
        """
        default_return = {  # 默认返回值
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},  # 意图（空名称，0置信度）
            "entities": [],  # 实体（空列表）
            "text": "",  # 文本（空字符串）
        }

        result = await self._rasa_http_parse(message.text, message.sender_id)  # 异步解析消息
        return result if result is not None else default_return  # 如果解析成功返回结果，否则返回默认值

    async def _rasa_http_parse(
        self, text: Text, message_id: Optional[Text] = None
    ) -> Optional[Dict[Text, Any]]:
        """将文本消息发送到运行的rasa NLU HTTP服务器。

        失败时返回 `None`。
        """
        if not self.endpoint_config or self.endpoint_config.url is None:  # 如果没有端点配置或URL为空
            structlogger.error(  # 记录错误日志
                "http.parse.text",
                text=copy.deepcopy(text),  # 深拷贝文本
                event_info="No rasa NLU server specified!",  # 事件信息
            )
            return None  # 返回None

        params = {  # 请求参数
            "token": self.endpoint_config.token,  # 认证令牌
            "text": text,  # 要解析的文本
            "message_id": message_id,  # 消息ID
        }

        if self.endpoint_config.url.endswith("/"):  # 如果URL以斜杠结尾
            url = self.endpoint_config.url + "model/parse"  # 直接拼接解析端点
        else:  # 否则
            url = self.endpoint_config.url + "/model/parse"  # 添加斜杠后拼接解析端点

        # noinspection PyBroadException
        try:  # 尝试发送HTTP请求
            async with self.session.post(url, json=params) as resp:  # 发送POST请求
                if resp.status == 200:  # 如果响应状态码为200
                    return await resp.json()  # 返回JSON响应
                else:  # 否则
                    response_text = await resp.text()  # 获取响应文本
                    structlogger.error(  # 记录错误日志
                        "http.parse.text.failure",
                        text=copy.deepcopy(text),  # 深拷贝文本
                        response_text=copy.deepcopy(response_text),  # 深拷贝响应文本
                    )
                    return None  # 返回None
        except Exception:  # skipcq: PYL-W0703
            # 在进行HTTP请求时需要捕获所有可能的异常
            # （超时、值错误、解析器错误等）
            structlogger.exception(  # 记录异常日志
                "http.parse.text.exception",
                text=copy.deepcopy(text),  # 深拷贝文本
            )
            return None  # 返回None
