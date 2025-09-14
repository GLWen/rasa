# 导入异步IO模块
import asyncio
# 导入JSON处理模块
import json
# 导入日志记录模块
import logging
# 导入深拷贝函数
from copy import deepcopy
# 导入Sanic框架的蓝图和响应模块
from sanic import Blueprint, response
# 导入Sanic请求对象
from sanic.request import Request
# 导入Sanic HTTP响应类
from sanic.response import HTTPResponse
# 导入aiogram机器人基类
from aiogram import Bot
# 导入aiogram类型定义
from aiogram.types import (
    InlineKeyboardButton,      # 内联键盘按钮
    Update,                    # 更新对象
    InlineKeyboardMarkup,      # 内联键盘标记
    KeyboardButton,            # 键盘按钮
    ReplyKeyboardMarkup,       # 回复键盘标记
    Message,                   # 消息对象
)
# 导入Telegram API异常类
from aiogram.utils.exceptions import TelegramAPIError
# 导入类型提示相关类型
from typing import Dict, Text, Any, List, Optional, Callable, Awaitable

# 导入Rasa核心通道相关类
from rasa.core.channels.channel import InputChannel, UserMessage, OutputChannel
# 导入意图消息前缀常量
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
# 导入用户重启意图常量
from rasa.shared.core.constants import USER_INTENT_RESTART
# 导入Rasa异常类
from rasa.shared.exceptions import RasaException

# 创建日志记录器
logger = logging.getLogger(__name__)


class TelegramOutput(Bot, OutputChannel):
    """Telegram输出通道实现。"""

    # 跳过代码质量检查警告
    # skipcq: PYL-W0236
    @classmethod
    def name(cls) -> Text:
        """返回通道名称。"""
        return "telegram"

    def __init__(self, access_token: Optional[Text]) -> None:
        """初始化Telegram输出通道。
        
        Args:
            access_token: Telegram机器人访问令牌
        """
        # 调用父类初始化方法
        super().__init__(access_token)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """发送文本消息。
        
        Args:
            recipient_id: 接收者ID（聊天ID）
            text: 要发送的文本消息
            **kwargs: 其他关键字参数
        """
        # 将文本按双换行符分割成多个部分，分别发送
        for message_part in text.strip().split("\n\n"):
            await self.send_message(recipient_id, message_part)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """发送图片。
        
        Args:
            recipient_id: 接收者ID（聊天ID）
            image: 图片URL
            **kwargs: 其他关键字参数
        """
        # 发送图片
        await self.send_photo(recipient_id, image)

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        button_type: Optional[Text] = "inline",
        **kwargs: Any,
    ) -> None:
        """发送带键盘的消息。

        更多信息请参考: https://core.telegram.org/bots#keyboards

        :button_type inline: 水平内联键盘

        :button_type vertical: 垂直内联键盘

        :button_type reply: 回复键盘
        
        Args:
            recipient_id: 接收者ID（聊天ID）
            text: 要发送的文本消息
            buttons: 按钮列表
            button_type: 按钮类型（inline/vertical/reply）
            **kwargs: 其他关键字参数
        """
        # 根据按钮类型创建不同的键盘
        if button_type == "inline":
            # 创建水平内联键盘
            reply_markup = InlineKeyboardMarkup()
            button_list = [
                InlineKeyboardButton(s["title"], callback_data=s["payload"])
                for s in buttons
            ]
            reply_markup.row(*button_list)

        elif button_type == "vertical":
            # 创建垂直内联键盘
            reply_markup = InlineKeyboardMarkup()
            [
                reply_markup.row(
                    InlineKeyboardButton(s["title"], callback_data=s["payload"])
                )
                for s in buttons
            ]

        elif button_type == "reply":
            # 创建回复键盘
            reply_markup = ReplyKeyboardMarkup(
                resize_keyboard=False, one_time_keyboard=True
            )
            # 从按钮列表中过滤出有标题的按钮
            button_list = [b for b in buttons if b.get("title")]
            for idx, button in enumerate(buttons):
                if isinstance(button, list):
                    # 如果是按钮列表，添加多个按钮
                    reply_markup.add(KeyboardButton(s["title"]) for s in button)
                else:
                    # 如果是单个按钮，添加单个按钮
                    reply_markup.add(KeyboardButton(button["title"]))
        else:
            # 未知按钮类型，记录错误并返回
            logger.error(
                "尝试发送未知按钮类型的文本消息: {}".format(button_type)
            )
            return

        # 发送带键盘的消息
        await self.send_message(recipient_id, text, reply_markup=reply_markup)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """发送自定义JSON载荷的消息。
        
        Args:
            recipient_id: 接收者ID（聊天ID）
            json_message: 自定义JSON消息内容
            **kwargs: 其他关键字参数
        """
        # 深拷贝JSON消息以避免修改原始数据
        json_message = deepcopy(json_message)

        # 从JSON消息中提取聊天ID，如果没有则使用默认的recipient_id
        recipient_id = json_message.pop("chat_id", recipient_id)

        # 定义发送函数映射表，根据参数组合确定使用哪个API方法
        send_functions = {
            ("text",): "send_message",                    # 发送文本消息
            ("photo",): "send_photo",                     # 发送图片
            ("audio",): "send_audio",                     # 发送音频
            ("document",): "send_document",               # 发送文档
            ("sticker",): "send_sticker",                 # 发送贴纸
            ("video",): "send_video",                     # 发送视频
            ("video_note",): "send_video_note",           # 发送视频笔记
            ("animation",): "send_animation",             # 发送动画
            ("voice",): "send_voice",                     # 发送语音
            ("media",): "send_media_group",               # 发送媒体组
            ("latitude", "longitude", "title", "address"): "send_venue",  # 发送地点
            ("latitude", "longitude"): "send_location",   # 发送位置
            ("phone_number", "first_name"): "send_contact",  # 发送联系人
            ("game_short_name",): "send_game",            # 发送游戏
            ("action",): "send_chat_action",              # 发送聊天动作
            (
                "title",
                "decription",
                "payload",
                "provider_token",
                "start_parameter",
                "currency",
                "prices",
            ): "send_invoice",                            # 发送发票
        }

        # 遍历所有参数组合，找到匹配的发送函数
        for params in send_functions.keys():
            # 检查JSON消息是否包含所有必需的参数
            if all(json_message.get(p) is not None for p in params):
                # 提取参数值
                args = [json_message.pop(p) for p in params]
                # 获取对应的API方法
                api_call = getattr(self, send_functions[params])
                # 调用API方法发送消息
                await api_call(recipient_id, *args, **json_message)


class TelegramInput(InputChannel):
    """Telegram输入通道实现"""

    @classmethod
    def name(cls) -> Text:
        """返回通道名称。"""
        return "telegram"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """从凭据创建输入通道实例。
        
        Args:
            credentials: 包含认证信息的字典
            
        Returns:
            TelegramInput实例
        """
        # 检查凭据是否存在
        if not credentials:
            cls.raise_missing_credentials_exception()

        # 从凭据中提取参数并创建实例
        return cls(
            credentials.get("access_token"),    # 访问令牌
            credentials.get("verify"),          # 验证令牌
            credentials.get("webhook_url"),     # Webhook URL
        )

    def __init__(
        self,
        access_token: Optional[Text],
        verify: Optional[Text],
        webhook_url: Optional[Text],
        debug_mode: bool = True,
    ) -> None:
        """初始化Telegram输入通道。
        
        Args:
            access_token: Telegram机器人访问令牌
            verify: 验证令牌
            webhook_url: Webhook URL
            debug_mode: 调试模式标志
        """
        # 保存认证信息
        self.access_token = access_token
        self.verify = verify
        self.webhook_url = webhook_url
        self.debug_mode = debug_mode

    @staticmethod
    def _is_location(message: Message) -> bool:
        """检查消息是否包含位置信息。
        
        Args:
            message: Telegram消息对象
            
        Returns:
            如果消息包含位置信息则返回True
        """
        return message.location is not None

    @staticmethod
    def _is_user_message(message: Message) -> bool:
        """检查消息是否为用户文本消息。
        
        Args:
            message: Telegram消息对象
            
        Returns:
            如果消息包含文本则返回True
        """
        return message.text is not None

    @staticmethod
    def _is_edited_message(message: Update) -> bool:
        """检查更新是否为编辑消息。
        
        Args:
            message: Telegram更新对象
            
        Returns:
            如果更新包含编辑消息则返回True
        """
        return message.edited_message is not None

    @staticmethod
    def _is_button(message: Update) -> bool:
        """检查更新是否为按钮点击。
        
        Args:
            message: Telegram更新对象
            
        Returns:
            如果更新包含回调查询则返回True
        """
        return message.callback_query is not None

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """创建Sanic蓝图用于处理Telegram webhook。
        
        Args:
            on_new_message: 新消息回调函数
            
        Returns:
            Sanic蓝图对象
        """
        # 创建Telegram webhook蓝图
        telegram_webhook = Blueprint("telegram_webhook", __name__)
        # 获取输出通道
        out_channel = self.get_output_channel()

        @telegram_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            """健康检查端点。"""
            return response.json({"status": "ok"})

        @telegram_webhook.route("/set_webhook", methods=["GET", "POST"])
        async def set_webhook(_: Request) -> HTTPResponse:
            """设置Telegram webhook端点。"""
            # 设置webhook
            s = await out_channel.set_webhook(self.webhook_url)
            if s:
                logger.info("Webhook设置成功")
                return response.text("Webhook setup successful")
            else:
                logger.warning("Webhook设置失败")
                return response.text("Invalid webhook")

        @telegram_webhook.route("/webhook", methods=["GET", "POST"])
        async def message(request: Request) -> Any:
            """处理Telegram webhook消息。"""
            if request.method == "POST":
                # 获取请求JSON数据
                request_dict = request.json
                if isinstance(request_dict, Text):
                    request_dict = json.loads(request_dict)
                # 创建Telegram更新对象
                update = Update(**request_dict)
                # 获取机器人凭据
                credentials = await out_channel.get_me()
                # 验证访问令牌
                if not credentials.username == self.verify:
                    logger.debug("无效的访问令牌，请检查是否与Telegram匹配")
                    return response.text("failed")

                # 根据更新类型提取消息和文本
                if self._is_button(update):
                    # 按钮点击消息
                    msg = update.callback_query.message
                    text = update.callback_query.data
                elif self._is_edited_message(update):
                    # 编辑消息
                    msg = update.edited_message
                    text = update.edited_message.text
                else:
                    # 普通消息
                    msg = update.message
                    if self._is_user_message(msg):
                        # 用户文本消息，移除/bot前缀
                        text = msg.text.replace("/bot", "")
                    elif self._is_location(msg):
                        # 位置消息，格式化为JSON
                        text = '{{"lng":{0}, "lat":{1}}}'.format(
                            msg.location.longitude, msg.location.latitude
                        )
                    else:
                        # 其他类型消息，直接返回成功
                        return response.text("success")
                
                # 获取发送者ID和元数据
                sender_id = msg.chat.id
                metadata = self.get_metadata(request)
                
                try:
                    # 检查是否为重启意图
                    if text == (INTENT_MESSAGE_PREFIX + USER_INTENT_RESTART):
                        # 发送重启意图消息
                        await on_new_message(
                            UserMessage(
                                text,
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                        # 发送/start消息
                        await on_new_message(
                            UserMessage(
                                "/start",
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                    else:
                        # 发送普通消息
                        await on_new_message(
                            UserMessage(
                                text,
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                except Exception as e:
                    # 异常处理
                    logger.error(f"处理消息时发生异常: {e}")
                    logger.debug(e, exc_info=True)
                    if self.debug_mode:
                        raise
                    pass

                return response.text("success")

        return telegram_webhook

    def get_output_channel(self) -> TelegramOutput:
        """加载Telegram通道。
        
        Returns:
            TelegramOutput输出通道实例
        """
        # 创建Telegram输出通道
        channel = TelegramOutput(self.access_token)

        try:
            # 设置webhook
            asyncio.run(channel.set_webhook(url=self.webhook_url))
        except TelegramAPIError as error:
            # 如果设置webhook失败，抛出Rasa异常
            raise RasaException(
                "设置通道webhook失败: " + str(error)
            ) from error

        return channel
