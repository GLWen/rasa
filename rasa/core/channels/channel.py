# 标准库导入
import json  # JSON数据处理
import logging  # 日志记录
import uuid  # UUID生成
import jwt  # JWT令牌处理

# 第三方库导入
from sanic import Sanic, Blueprint  # Sanic Web框架和蓝图
from sanic.request import Request  # Sanic请求对象

# 类型提示导入
from typing import (
    Text,  # 文本类型
    List,  # 列表类型
    Dict,  # 字典类型
    Any,  # 任意类型
    Optional,  # 可选类型
    Callable,  # 可调用类型
    Iterable,  # 可迭代类型
    Awaitable,  # 可等待类型
    NoReturn,  # 无返回值类型
)

# Rasa 内部模块导入
from rasa.cli import utils as cli_utils  # CLI工具
from rasa.shared.constants import DOCS_BASE_URL, DEFAULT_SENDER_ID  # 共享常量
from rasa.core.constants import BEARER_TOKEN_PREFIX  # 核心常量
from rasa.shared.exceptions import RasaException  # Rasa异常

# 兼容性导入
try:
    from urlparse import urljoin  # Python 2 的URL处理
except ImportError:
    from urllib.parse import urljoin  # Python 3 的URL处理

# 日志记录器
logger = logging.getLogger(__name__)


class UserMessage:
    """表示传入的消息。

    包括应该发送机器人响应的通道。"""

    def __init__(
        self,
        text: Optional[Text] = None,
        output_channel: Optional["OutputChannel"] = None,
        sender_id: Optional[Text] = None,
        parse_data: Dict[Text, Any] = None,
        input_channel: Optional[Text] = None,
        message_id: Optional[Text] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """创建一个 ``UserMessage`` 对象。

        Args:
            text: 消息文本内容。
            output_channel: 用于向用户发送机器人响应的输出通道。
            sender_id: 消息所有者ID。
            parse_data: 关于消息的rasa数据。
            input_channel: 接收此消息的通道名称。
            message_id: 消息的ID。
            metadata: 此消息的附加元数据。

        """
        self.text = text.strip() if text else text  # 如果文本存在则去除首尾空白

        if message_id is not None:  # 如果提供了消息ID
            self.message_id = str(message_id)  # 转换为字符串
        else:  # 否则
            self.message_id = uuid.uuid4().hex  # 生成新的UUID

        if output_channel is not None:  # 如果提供了输出通道
            self.output_channel = output_channel  # 使用提供的通道
        else:  # 否则
            self.output_channel = CollectingOutputChannel()  # 使用收集输出通道

        if sender_id is not None:  # 如果提供了发送者ID
            self.sender_id = str(sender_id)  # 转换为字符串
        else:  # 否则
            self.sender_id = DEFAULT_SENDER_ID  # 使用默认发送者ID

        self.input_channel = input_channel  # 输入通道

        self.parse_data = parse_data  # 解析数据
        self.metadata = metadata  # 元数据


def register(
    input_channels: List["InputChannel"], app: Sanic, route: Optional[Text]
) -> None:
    """向Sanic注册输入通道蓝图。"""

    async def handler(message: UserMessage) -> None:
        """消息处理器，将消息传递给代理处理。"""
        await app.ctx.agent.handle_message(message)  # 调用代理的消息处理方法

    for channel in input_channels:  # 遍历所有输入通道
        if route:  # 如果提供了路由
            p = urljoin(route, channel.url_prefix())  # 拼接完整路由
        else:  # 否则
            p = None  # 使用None
        app.blueprint(channel.blueprint(handler), url_prefix=p)  # 注册通道蓝图

    app.ctx.input_channels = input_channels  # 将输入通道存储到应用上下文中


class InputChannel:
    """输入通道基类。"""

    @classmethod
    def name(cls) -> Text:
        """每个输入通道都需要一个名称来标识它。"""
        return cls.__name__  # 返回类名

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> "InputChannel":
        """从凭据创建输入通道实例。"""
        return cls()  # 返回新实例

    def url_prefix(self) -> Text:
        """返回URL前缀。"""
        return self.name()  # 默认使用类名作为前缀

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """定义Sanic蓝图。

        蓝图将附加到运行的sanic服务器并处理它注册的传入路由。"""
        raise NotImplementedError("组件监听器需要提供蓝图。")  # 子类必须实现

    @classmethod
    def raise_missing_credentials_exception(cls) -> NoReturn:
        """抛出缺少凭据的异常。"""
        raise RasaException(
            f"要使用 {cls.name()} 输入通道，您需要使用 '--credentials' 传递凭据文件。 "
            f"参数应该是指向包含 {cls.name()} 身份验证信息的yml文件的文件路径。 "
            f"详细信息请参阅文档： "
            f"{DOCS_BASE_URL}/messaging-and-voice-channels/"
        )

    def get_output_channel(self) -> Optional["OutputChannel"]:
        """基于输入通道提供的信息创建 ``OutputChannel``。

        实现此函数不是必需的。如果此函数返回有效的 ``OutputChannel``，
        则Rasa可以使用它向用户发送机器人响应，而无需用户发起交互。

        Returns:
            ``OutputChannel`` 实例，如果仅基于 ``InputChannel`` 中存在的信息
            无法创建输出通道，则返回 ``None``。
        """
        pass  # 默认实现为空

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """从传入请求中提取附加信息。

         实现此函数不是必需的。但是，它可以用于从请求中提取元数据。
         返回值传递给 ``UserMessage`` 对象并存储在对话跟踪器中。

        Args:
            request: 包含用户消息的传入请求

        Returns:
            从请求中提取的元数据。
        """
        pass  # 默认实现为空


def decode_jwt(bearer_token: Text, jwt_key: Text, jwt_algorithm: Text) -> Dict:
    """使用特定的JWT密钥和算法解码Bearer令牌。

    Args:
        bearer_token: 编码的Bearer令牌
        jwt_key: 用于解码Bearer令牌的公共JWT密钥
        jwt_algorithm: 用于解码Bearer令牌的JWT算法

    Returns:
        成功时包含解码负载的 `Dict`，失败时抛出异常
    """
    authorization_header_value = bearer_token.replace(BEARER_TOKEN_PREFIX, "")  # 移除Bearer前缀
    return jwt.decode(authorization_header_value, jwt_key, algorithms=jwt_algorithm)  # 解码JWT


def decode_bearer_token(
    bearer_token: Text, jwt_key: Text, jwt_algorithm: Text
) -> Optional[Dict]:
    """使用特定的JWT密钥和算法解码Bearer令牌。

    Args:
        bearer_token: 编码的Bearer令牌
        jwt_key: 用于解码Bearer令牌的公共JWT密钥
        jwt_algorithm: 用于解码Bearer令牌的JWT算法

    Returns:
        成功时包含解码负载的 `Dict`，失败时返回 `None`
    """
    # noinspection PyBroadException
    try:  # 尝试解码JWT
        return decode_jwt(bearer_token, jwt_key, jwt_algorithm)  # 调用解码函数
    except jwt.exceptions.InvalidSignatureError:  # 捕获无效签名错误
        logger.error("JWT public key invalid.")  # 记录错误日志
    except Exception:  # 捕获其他异常
        logger.exception("Failed to decode bearer token.")  # 记录异常日志

    return None  # 返回None表示解码失败


class OutputChannel:
    """输出通道基类。

    为仅文本输出通道提供发送方法的合理实现。
    """

    @classmethod
    def name(cls) -> Text:
        """每个输出通道都需要一个名称来标识它。"""
        return cls.__name__  # 返回类名

    async def send_response(self, recipient_id: Text, message: Dict[Text, Any]) -> None:
        """向客户端发送消息。"""

        if message.get("quick_replies"):  # 如果有快速回复
            await self.send_quick_replies(  # 发送快速回复
                recipient_id,
                message.pop("text"),  # 弹出文本
                message.pop("quick_replies"),  # 弹出快速回复
                **message,  # 传递其他参数
            )
        elif message.get("buttons"):  # 如果有按钮
            await self.send_text_with_buttons(  # 发送带按钮的文本
                recipient_id, message.pop("text"), message.pop("buttons"), **message
            )
        elif message.get("text"):  # 如果有文本
            await self.send_text_message(recipient_id, message.pop("text"), **message)  # 发送文本消息

        if message.get("custom"):  # 如果有自定义内容
            await self.send_custom_json(recipient_id, message.pop("custom"), **message)  # 发送自定义JSON

        # 如果有图像，我们将其作为附件单独处理
        if message.get("image"):  # 如果有图像
            await self.send_image_url(recipient_id, message.pop("image"), **message)  # 发送图像URL

        if message.get("attachment"):  # 如果有附件
            await self.send_attachment(  # 发送附件
                recipient_id, message.pop("attachment"), **message
            )

        if message.get("elements"):  # 如果有元素
            await self.send_elements(recipient_id, message.pop("elements"), **message)  # 发送元素

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """通过此通道发送消息。"""

        raise NotImplementedError(
            "输出通道需要实现简单文本的发送消息方法。"
        )  # 子类必须实现

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """发送图像。默认实现只是将URL作为字符串发布。"""

        await self.send_text_message(recipient_id, f"Image: {image}")  # 发送图像URL作为文本

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """发送附件。默认实现只是将其作为字符串发布。"""

        await self.send_text_message(recipient_id, f"Attachment: {attachment}")  # 发送附件信息作为文本

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """向输出发送按钮。

        默认实现只是将按钮作为字符串发布。"""

        await self.send_text_message(recipient_id, text)  # 发送文本
        for idx, button in enumerate(buttons):  # 遍历按钮
            button_msg = cli_utils.button_to_string(button, idx)  # 将按钮转换为字符串
            await self.send_text_message(recipient_id, button_msg)  # 发送按钮消息

    async def send_quick_replies(
        self,
        recipient_id: Text,
        text: Text,
        quick_replies: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """向输出发送快速回复。

        默认实现只是将其作为按钮发送。"""

        await self.send_text_with_buttons(recipient_id, text, quick_replies)  # 将快速回复作为按钮发送

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """向输出发送元素。

        默认实现只是将元素作为字符串发布。"""

        for element in elements:  # 遍历元素
            element_msg = "{title} : {subtitle}".format(  # 格式化元素消息
                title=element.get("title", ""), subtitle=element.get("subtitle", "")
            )
            await self.send_text_with_buttons(  # 发送带按钮的元素
                recipient_id, element_msg, element.get("buttons", [])
            )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """向输出通道发送json字典。

        默认实现只是将json内容作为字符串发布。"""

        await self.send_text_message(recipient_id, json.dumps(json_message))  # 将JSON转换为字符串发送


class CollectingOutputChannel(OutputChannel):
    """收集发送消息到列表中的输出通道

    （不发送到任何地方，只是收集它们）。"""

    def __init__(self) -> None:
        """初始化用于收集消息的列表。"""
        self.messages: List[Dict[Text, Any]] = []  # 消息列表

    @classmethod
    def name(cls) -> Text:
        """通道名称。"""
        return "collector"  # 返回收集器名称

    @staticmethod
    def _message(
        recipient_id: Text,
        text: Text = None,
        image: Text = None,
        buttons: List[Dict[Text, Any]] = None,
        attachment: Text = None,
        custom: Dict[Text, Any] = None,
    ) -> Dict:
        """创建将被存储的消息对象。"""

        obj = {  # 消息对象
            "recipient_id": recipient_id,  # 接收者ID
            "text": text,  # 文本
            "image": image,  # 图像
            "buttons": buttons,  # 按钮
            "attachment": attachment,  # 附件
            "custom": custom,  # 自定义内容
        }

        # 过滤掉任何值为 `None` 的项
        return {k: v for k, v in obj.items() if v is not None}

    def latest_output(self) -> Optional[Dict[Text, Any]]:
        """获取最新的输出消息。"""
        if self.messages:  # 如果有消息
            return self.messages[-1]  # 返回最后一个消息
        else:  # 否则
            return None  # 返回None

    async def _persist_message(self, message: Dict[Text, Any]) -> None:
        """持久化消息到列表中。"""
        self.messages.append(message)  # 添加消息到列表

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """发送文本消息。"""
        for message_part in text.strip().split("\n\n"):  # 按双换行符分割文本
            await self._persist_message(self._message(recipient_id, text=message_part))  # 持久化每个部分

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """发送图像。默认实现只是将URL作为字符串发布。"""

        await self._persist_message(self._message(recipient_id, image=image))  # 持久化图像消息

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """发送附件。默认实现只是将其作为字符串发布。"""

        await self._persist_message(self._message(recipient_id, attachment=attachment))  # 持久化附件消息

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """发送带按钮的文本消息。"""
        await self._persist_message(  # 持久化消息
            self._message(recipient_id, text=text, buttons=buttons)
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """发送自定义JSON消息。"""
        await self._persist_message(self._message(recipient_id, custom=json_message))  # 持久化自定义消息
