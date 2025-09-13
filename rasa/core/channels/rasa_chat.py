# 导入JSON处理模块，用于序列化和反序列化
import json
# 导入类型提示相关的类型
from typing import Text, Optional, Dict, Any

# 导入异步HTTP客户端
import aiohttp
# 导入日志模块，用于记录日志信息
import logging
# 导入Sanic异常类
from sanic.exceptions import SanicException
# 导入JWT处理模块
import jwt
# 导入JWT异常类
import jwt.exceptions

# 导入Rasa核心通道模块
import rasa.core.channels.channel
# 导入输入通道基类
from rasa.core.channels.channel import InputChannel
# 导入REST输入通道
from rasa.core.channels.rest import RestInput
# 导入默认请求超时常量
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
# 导入Sanic请求类
from sanic.request import Request

# 创建日志记录器
logger = logging.getLogger(__name__)

# 对话ID键名常量
CONVERSATION_ID_KEY = "conversation_id"
# JWT用户名键名常量
JWT_USERNAME_KEY = "username"
# 交互式学习权限常量
INTERACTIVE_LEARNING_PERMISSION = "clientEvents:create"


class RasaChatInput(RestInput):
    """Rasa企业版的聊天输入通道。"""

    @classmethod
    def name(cls) -> Text:
        """返回通道名称。
        
        Returns:
            通道名称"rasa"
        """
        return "rasa"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """从凭据创建输入通道。
        
        Args:
            credentials: 包含通道配置的凭据字典
            
        Returns:
            创建的输入通道实例
        """
        if not credentials:
            # 如果没有凭据，抛出缺少凭据异常
            cls.raise_missing_credentials_exception()

        # 从凭据中获取URL并创建通道实例
        return cls(credentials.get("url"))

    def __init__(self, url: Optional[Text]) -> None:
        """使用属性初始化通道。
        
        Args:
            url: Rasa企业版的基础URL
        """
        self.base_url = url                    # 基础URL
        self.jwt_key: Optional[Text] = None    # JWT公钥
        self.jwt_algorithm = None              # JWT算法

    async def _fetch_public_key(self) -> None:
        """从Rasa企业版获取JWT公钥。
        
        从Rasa企业版的/version端点获取JWT公钥，用于验证JWT令牌。
        """
        # 构建公钥获取URL
        public_key_url = f"{self.base_url}/version"
        # 创建HTTP会话并发送GET请求
        async with aiohttp.ClientSession() as session:
            async with session.get(
                public_key_url, timeout=DEFAULT_REQUEST_TIMEOUT
            ) as resp:
                # 获取响应状态码
                status_code = resp.status
                if status_code != 200:
                    # 如果请求失败，记录错误日志
                    logger.error(
                        "Failed to fetch JWT public key from URL '{}' with "
                        "status code {}: {}"
                        "".format(public_key_url, status_code, await resp.text())
                    )
                    return
                # 解析JSON响应
                rjs = await resp.json()
                public_key_field = "keys"
                if public_key_field in rjs:
                    # 从响应中提取JWT公钥和算法
                    self.jwt_key = rjs["keys"][0]["key"]
                    self.jwt_algorithm = rjs["keys"][0]["alg"]
                    logger.debug(
                        "Fetched JWT public key from URL '{}' for algorithm '{}':\n{}"
                        "".format(public_key_url, self.jwt_algorithm, self.jwt_key)
                    )
                else:
                    # 如果响应中没有找到公钥字段，记录错误日志
                    logger.error(
                        "Retrieved json response from URL '{}' but could not find "
                        "'{}' field containing the JWT public key. Please make sure "
                        "you use an up-to-date version of Rasa Enterprise (>= 0.20.2). "
                        "Response was: {}"
                        "".format(public_key_url, public_key_field, json.dumps(rjs))
                    )

    async def _decode_bearer_token(self, bearer_token: Text) -> Optional[Dict]:
        """解码Bearer令牌。
        
        Args:
            bearer_token: 要解码的Bearer令牌
            
        Returns:
            解码后的JWT载荷字典，如果解码失败则返回None
        """
        # 如果没有JWT公钥，先获取公钥
        if self.jwt_key is None:
            await self._fetch_public_key()

        try:
            # 尝试解码JWT令牌
            return rasa.core.channels.channel.decode_jwt(
                bearer_token, self.jwt_key, self.jwt_algorithm
            )
        except jwt.InvalidSignatureError:
            # 如果签名无效，记录错误并重新获取公钥
            logger.error("JWT public key invalid, fetching new one.")
            await self._fetch_public_key()
            # 使用新公钥再次尝试解码
            return rasa.core.channels.channel.decode_jwt(
                bearer_token, self.jwt_key, self.jwt_algorithm
            )

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        """从Rasa企业版管理API获取用户。
        
        Args:
            req: Sanic请求对象
            
        Returns:
            发送者ID（用户名或对话ID）
            
        Raises:
            SanicException: 如果认证失败或权限不足
        """
        jwt_payload = None
        # 首先尝试从Authorization头获取JWT令牌
        if req.headers.get("Authorization"):
            jwt_payload = await self._decode_bearer_token(req.headers["Authorization"])

        # 如果从头部没有获取到，尝试从查询参数获取
        if not jwt_payload:
            jwt_payload = await self._decode_bearer_token(req.args.get("token"))

        # 如果仍然没有获取到JWT载荷，抛出401未授权异常
        if not jwt_payload:
            raise SanicException(status_code=401)

        # 如果请求中包含对话ID
        if CONVERSATION_ID_KEY in req.json:
            # 检查用户是否有权限向该对话发送消息
            if self._has_user_permission_to_send_messages_to_conversation(
                jwt_payload, req.json
            ):
                # 返回对话ID作为发送者
                return req.json[CONVERSATION_ID_KEY]
            else:
                # 记录权限不足错误
                logger.error(
                    "User '{}' does not have permissions to send messages to "
                    "conversation '{}'.".format(
                        jwt_payload[JWT_USERNAME_KEY], req.json[CONVERSATION_ID_KEY]
                    )
                )
                # 抛出401未授权异常
                raise SanicException(status_code=401)

        # 返回JWT载荷中的用户名作为发送者
        return jwt_payload[JWT_USERNAME_KEY]

    @staticmethod
    def _has_user_permission_to_send_messages_to_conversation(
        jwt_payload: Dict, message: Dict
    ) -> bool:
        """检查用户是否有权限向指定对话发送消息。
        
        Args:
            jwt_payload: JWT载荷字典
            message: 消息字典
            
        Returns:
            如果用户有权限则返回True，否则返回False
        """
        # 获取用户的作用域列表
        user_scopes = jwt_payload.get("scopes", [])
        # 检查用户是否有交互式学习权限，或者对话ID与用户名相同
        return INTERACTIVE_LEARNING_PERMISSION in user_scopes or message[
            CONVERSATION_ID_KEY
        ] == jwt_payload.get(JWT_USERNAME_KEY)
