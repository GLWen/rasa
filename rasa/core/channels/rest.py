# 导入异步IO模块
import asyncio
# 导入深拷贝模块
import copy
# 导入反射检查模块
import inspect
# 导入JSON处理模块
import json
# 导入日志记录模块
import logging
# 导入结构化日志模块
import structlog
# 导入异步队列和取消异常
from asyncio import Queue, CancelledError
# 导入Sanic框架的蓝图和响应模块
from sanic import Blueprint, response
# 导入Sanic请求对象
from sanic.request import Request
# 导入Sanic HTTP响应和流响应类
from sanic.response import HTTPResponse, ResponseStream
# 导入类型提示相关类型
from typing import Text, Dict, Any, Optional, Callable, Awaitable, NoReturn, Union

# 导入Rasa端点工具模块
import rasa.utils.endpoints
# 导入Rasa核心通道相关类
from rasa.core.channels.channel import (
    InputChannel,                    # 输入通道基类
    CollectingOutputChannel,         # 收集输出通道
    UserMessage,                     # 用户消息类
)


# 创建标准日志记录器
logger = logging.getLogger(__name__)
# 创建结构化日志记录器
structlogger = structlog.get_logger()


class RestInput(InputChannel):
    """自定义HTTP输入通道。

    此实现是自定义聊天前端实现的基础。您可以自定义此通道来向Rasa发送消息
    并从助手那里检索响应。
    """

    @classmethod
    def name(cls) -> Text:
        """返回通道名称。"""
        return "rest"

    @staticmethod
    async def on_message_wrapper(
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        text: Text,
        queue: Queue,
        sender_id: Text,
        input_channel: Text,
        metadata: Optional[Dict[Text, Any]],
    ) -> None:
        """消息包装器，用于处理新消息。
        
        Args:
            on_new_message: 新消息回调函数
            text: 消息文本
            queue: 消息队列
            sender_id: 发送者ID
            input_channel: 输入通道名称
            metadata: 可选的元数据
        """
        # 创建队列输出通道收集器
        collector = QueueOutputChannel(queue)

        # 创建用户消息对象
        message = UserMessage(
            text, collector, sender_id, input_channel=input_channel, metadata=metadata
        )
        # 调用新消息处理函数
        await on_new_message(message)

        # 向队列发送完成信号
        await queue.put("DONE")

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        """从请求中提取发送者ID。
        
        Args:
            req: HTTP请求对象
            
        Returns:
            发送者ID，如果不存在则返回None
        """
        return req.json.get("sender", None)

    # 忽略静态方法检查警告
    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        """从请求中提取消息文本。
        
        Args:
            req: HTTP请求对象
            
        Returns:
            消息文本，如果不存在则返回None
        """
        return req.json.get("message", None)

    def _extract_input_channel(self, req: Request) -> Text:
        """从请求中提取输入通道名称。
        
        Args:
            req: HTTP请求对象
            
        Returns:
            输入通道名称，如果不存在则使用默认通道名称
        """
        return req.json.get("input_channel") or self.name()

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """从传入请求中提取附加信息。

        实现此函数不是必需的。但是，它可以用于从请求中提取元数据。
        返回值传递给``UserMessage``对象并存储在对话跟踪器中。

        Args:
            request: 包含用户消息的传入请求

        Returns:
            从请求中提取的元数据。
        """
        return request.json.get("metadata", None)

    def stream_response(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        text: Text,
        sender_id: Text,
        input_channel: Text,
        metadata: Optional[Dict[Text, Any]],
    ) -> Callable[[Any], Awaitable[None]]:
        """将响应流式传输到客户端。

         如果启用了流选项，将调用此方法将响应流式传输到客户端

        Args:
            on_new_message: sanic事件
            text: 消息文本
            sender_id: 消息发送者ID
            input_channel: 输入通道名称
            metadata: 随消息发送的可选元数据

        Returns:
            Sanic流
        """

        async def stream(resp: Any) -> None:
            """流式响应处理函数。"""
            # 创建消息队列
            q: Queue = Queue()
            # 创建异步任务处理消息
            task = asyncio.ensure_future(
                self.on_message_wrapper(
                    on_new_message, text, q, sender_id, input_channel, metadata
                )
            )
            # 循环处理队列中的消息
            while True:
                result = await q.get()
                if result == "DONE":
                    # 收到完成信号，退出循环
                    break
                else:
                    # 将结果写入响应流
                    await resp.write(json.dumps(result) + "\n")
            # 等待任务完成
            await task

        return stream

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        """分组rest通道使用的端点集合。
        
        Args:
            on_new_message: 新消息回调函数
            
        Returns:
            Sanic蓝图对象
        """
        # 获取当前模块类型
        module_type = inspect.getmodule(self)
        if module_type is not None:
            module_name = module_type.__name__
        else:
            module_name = None

        # 创建自定义webhook蓝图
        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            module_name,
        )

        # 忽略未使用局部变量警告
        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            """健康检查端点。"""
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> Union[ResponseStream, HTTPResponse]:
            """接收webhook消息的端点。"""
            # 从请求中提取各种信息
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)
            # 检查是否使用流式响应
            should_use_stream = rasa.utils.endpoints.bool_arg(
                request, "stream", default=False
            )
            input_channel = self._extract_input_channel(request)
            metadata = self.get_metadata(request)

            if should_use_stream:
                # 使用流式响应
                return response.stream(
                    self.stream_response(
                        on_new_message, text, sender_id, input_channel, metadata
                    ),
                    content_type="text/event-stream",
                )
            else:
                # 使用普通响应
                collector = CollectingOutputChannel()
                # 忽略广泛异常捕获警告
                # noinspection PyBroadException
                try:
                    # 处理新消息
                    await on_new_message(
                        UserMessage(
                            text,
                            collector,
                            sender_id,
                            input_channel=input_channel,
                            metadata=metadata,
                        )
                    )
                except CancelledError:
                    # 处理取消异常
                    structlogger.error(
                        "rest.message.received.timeout", text=copy.deepcopy(text)
                    )
                except Exception:
                    # 处理其他异常
                    structlogger.exception(
                        "rest.message.received.failure", text=copy.deepcopy(text)
                    )

                # 返回收集的消息
                return response.json(collector.messages)

        return custom_webhook


class QueueOutputChannel(CollectingOutputChannel):
    """在列表中收集发送消息的输出通道。

    （不发送到任何地方，只是收集它们）。
    """

    # 注意：这违反了里氏替换原则
    # 需要一些面向用户的重构来解决
    # FIXME: this is breaking Liskov substitution principle
    # and would require some user-facing refactoring to address
    messages: Queue  # type: ignore[assignment]

    @classmethod
    def name(cls) -> Text:
        """返回QueueOutputChannel的名称。"""
        return "queue"

    # 忽略缺少构造函数警告
    # noinspection PyMissingConstructor
    def __init__(self, message_queue: Optional[Queue] = None) -> None:
        """初始化队列输出通道。
        
        Args:
            message_queue: 可选的现有消息队列
        """
        # 调用父类构造函数
        super().__init__()
        # 设置消息队列
        self.messages = Queue() if not message_queue else message_queue

    def latest_output(self) -> NoReturn:
        """获取最新输出（队列不支持此功能）。
        
        Raises:
            NotImplementedError: 队列不允许查看消息
        """
        raise NotImplementedError("队列不允许查看消息。")

    async def _persist_message(self, message: Dict[Text, Any]) -> None:
        """将消息持久化到队列中。
        
        Args:
            message: 要持久化的消息字典
        """
        await self.messages.put(message)
