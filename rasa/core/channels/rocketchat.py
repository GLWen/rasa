# 导入日志记录模块
import logging
# 导入Sanic框架的蓝图和响应模块
from sanic import Blueprint, response
# 导入Sanic请求对象
from sanic.request import Request
# 导入类型提示相关类型
from typing import Text, Dict, Any, List, Iterable, Optional, Callable, Awaitable

# 导入Rasa核心通道相关类
from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
# 导入Sanic HTTP响应类
from sanic.response import HTTPResponse

# 创建日志记录器
logger = logging.getLogger(__name__)


class RocketChatBot(OutputChannel):
    """RocketChat机器人输出通道实现。"""
    
    @classmethod
    def name(cls) -> Text:
        """返回通道名称。"""
        return "rocketchat"

    def __init__(self, user: Text, password: Text, server_url: Text) -> None:
        """初始化RocketChat机器人。
        
        Args:
            user: RocketChat用户名
            password: RocketChat密码
            server_url: RocketChat服务器URL
        """
        # 导入RocketChat API库
        from rocketchat_API.rocketchat import RocketChat

        # 创建RocketChat客户端实例
        self.rocket = RocketChat(user, password, server_url=server_url)

    @staticmethod
    def _convert_to_rocket_buttons(buttons: List[Dict]) -> List[Dict]:
        """将Rasa按钮格式转换为RocketChat按钮格式。
        
        Args:
            buttons: Rasa格式的按钮列表
            
        Returns:
            RocketChat格式的按钮列表
        """
        return [
            {
                "text": b["title"],                    # 按钮显示文本
                "msg": b["payload"],                   # 按钮点击时发送的消息
                "type": "button",                      # 按钮类型
                "msg_in_chat_window": True,            # 消息是否显示在聊天窗口中
            }
            for b in buttons
        ]

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """发送文本消息到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            text: 要发送的文本消息
            **kwargs: 其他关键字参数
        """
        # 将文本按双换行符分割成多个部分，分别发送
        for message_part in text.strip().split("\n\n"):
            self.rocket.chat_post_message(message_part, room_id=recipient_id)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """发送图片URL到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            image: 图片URL
            **kwargs: 其他关键字参数
        """
        # 创建图片附件
        image_attachment = [{"image_url": image, "collapsed": False}]

        # 发送带图片附件的消息
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=image_attachment
        )

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """发送附件到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            attachment: 附件内容
            **kwargs: 其他关键字参数
        """
        # 发送带附件的消息
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=[attachment]
        )

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """发送带按钮的文本消息到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            text: 要发送的文本消息
            buttons: 按钮列表
            **kwargs: 其他关键字参数
        """
        # 实现基于
        # https://github.com/RocketChat/Rocket.Chat/pull/11473
        # 应该在rocket chat >= 0.69.0版本中工作
        # 创建按钮附件
        button_attachment = [{"actions": self._convert_to_rocket_buttons(buttons)}]

        # 发送带按钮附件的消息
        return self.rocket.chat_post_message(
            text, room_id=recipient_id, attachments=button_attachment
        )

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """发送元素到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            elements: 元素列表
            **kwargs: 其他关键字参数
        """
        # 发送带元素的消息
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=elements
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """发送自定义JSON消息到输出通道。
        
        Args:
            recipient_id: 接收者ID（房间ID）
            json_message: JSON消息内容
            **kwargs: 其他关键字参数
        """
        # 从JSON消息中提取文本内容
        text = json_message.pop("text")

        # 如果指定了channel参数
        if json_message.get("channel"):
            # 如果同时指定了room_id，发出警告并删除room_id
            if json_message.get("room_id"):
                logger.warning(
                    "只能向RocketChat消息发布传递`channel`或`room_id`中的一个。"
                    "默认使用`channel`。"
                )
                del json_message["room_id"]
            # 使用channel发送消息
            return self.rocket.chat_post_message(text, **json_message)
        else:
            # 如果没有指定channel，使用默认的room_id
            json_message.setdefault("room_id", recipient_id)
            return self.rocket.chat_post_message(text, **json_message)


class RocketChatInput(InputChannel):
    """RocketChat输入通道实现。"""

    @classmethod
    def name(cls) -> Text:
        """返回通道名称。"""
        return "rocketchat"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """从凭据创建输入通道实例。
        
        Args:
            credentials: 包含认证信息的字典
            
        Returns:
            RocketChatInput实例
        """
        # 检查凭据是否存在
        if not credentials:
            cls.raise_missing_credentials_exception()

        # 从凭据中提取参数并创建实例
        return cls(
            credentials.get("user"),           # 用户名
            credentials.get("password"),       # 密码
            credentials.get("server_url"),     # 服务器URL
        )

    def __init__(self, user: Text, password: Text, server_url: Text) -> None:
        """初始化RocketChat输入通道。
        
        Args:
            user: RocketChat用户名
            password: RocketChat密码
            server_url: RocketChat服务器URL
        """
        # 保存认证信息
        self.user = user
        self.password = password
        self.server_url = server_url

    async def send_message(
        self,
        text: Optional[Text],
        sender_name: Optional[Text],
        recipient_id: Optional[Text],
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        metadata: Optional[Dict],
    ) -> None:
        """发送消息到Rasa处理器。
        
        Args:
            text: 消息文本
            sender_name: 发送者名称
            recipient_id: 接收者ID
            on_new_message: 新消息回调函数
            metadata: 元数据
        """
        # 只处理不是来自机器人自己的消息
        if sender_name != self.user:
            # 获取输出通道
            output_channel = self.get_output_channel()

            # 创建用户消息对象
            user_msg = UserMessage(
                text,                           # 消息文本
                output_channel,                # 输出通道
                recipient_id,                  # 接收者ID
                input_channel=self.name(),     # 输入通道名称
                metadata=metadata,             # 元数据
            )
            # 调用新消息回调函数
            await on_new_message(user_msg)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """创建Sanic蓝图用于处理RocketChat webhook。
        
        Args:
            on_new_message: 新消息回调函数
            
        Returns:
            Sanic蓝图对象
        """
        # 创建RocketChat webhook蓝图
        rocketchat_webhook = Blueprint("rocketchat_webhook", __name__)

        @rocketchat_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            """健康检查端点。"""
            return response.json({"status": "ok"})

        @rocketchat_webhook.route("/webhook", methods=["GET", "POST"])
        async def webhook(request: Request) -> HTTPResponse:
            """处理RocketChat webhook请求。"""
            # 获取请求JSON数据
            output = request.json
            # 获取元数据
            metadata = self.get_metadata(request)
            
            if output:
                # 检查是否为访客消息
                if "visitor" not in output:
                    # 普通用户消息
                    sender_name = output.get("user_name", None)
                    text = output.get("text", None)
                    recipient_id = output.get("channel_id", None)
                else:
                    # 访客消息
                    messages_list = output.get("messages", None)
                    text = messages_list[0].get("msg", None)
                    sender_name = messages_list[0].get("username", None)
                    recipient_id = output.get("_id")

                # 发送消息到Rasa处理器
                await self.send_message(
                    text, sender_name, recipient_id, on_new_message, metadata
                )

            # 返回空响应
            return response.text("")

        return rocketchat_webhook

    def get_output_channel(self) -> OutputChannel:
        """获取输出通道实例。
        
        Returns:
            RocketChatBot输出通道实例
        """
        return RocketChatBot(self.user, self.password, self.server_url)
