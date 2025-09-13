import asyncio  # 异步编程模块
import json  # JSON处理模块
import logging  # 日志记录模块
import structlog  # 结构化日志模块
import os  # 操作系统接口模块
import ssl  # SSL/TLS模块
from asyncio import AbstractEventLoop  # 异步事件循环抽象基类
from collections import deque  # 双端队列
from typing import Deque, Dict, Optional, Text, Union, Any, List, Set, Tuple, cast  # 类型注解模块
from urllib.parse import urlparse  # URL解析模块

import aio_pika  # 异步Pika客户端库

from rasa.shared.exceptions import RasaException  # Rasa异常类
from rasa.shared.constants import DOCS_URL_PIKA_EVENT_BROKER  # Pika事件代理文档URL
from rasa.core.brokers.broker import EventBroker  # 事件代理基类
import rasa.shared.utils.io  # Rasa共享IO工具
from rasa.utils.endpoints import EndpointConfig  # 端点配置类
from rasa.shared.utils.io import DEFAULT_ENCODING  # 默认编码
import rasa.shared.utils.common  # Rasa共享通用工具

logger = logging.getLogger(__name__)  # 创建日志记录器
structlogger = structlog.get_logger()  # 创建结构化日志记录器

RABBITMQ_EXCHANGE = "rasa-exchange"  # RabbitMQ交换器名称
DEFAULT_QUEUE_NAME = "rasa_core_events"  # 默认队列名称


class PikaEventBroker(EventBroker):
    """基于Pika的事件代理，用于向RabbitMQ发布消息。"""

    def __init__(
        self,
        host: Text,  # Pika主机
        username: Text,  # 用户名
        password: Text,  # 密码
        port: Union[int, Text] = 5672,  # 端口
        queues: Union[List[Text], Tuple[Text, ...], Text, None] = None,  # 队列列表
        should_keep_unpublished_messages: bool = True,  # 是否保留未发布的消息
        raise_on_failure: bool = False,  # 失败时是否抛出异常
        event_loop: Optional[AbstractEventLoop] = None,  # 事件循环
        connection_attempts: int = 20,  # 连接尝试次数
        retry_delay_in_seconds: float = 5,  # 重试延迟（秒）
        exchange_name: Text = RABBITMQ_EXCHANGE,  # 交换器名称
        **kwargs: Any,  # 其他参数
    ):
        """初始化RabbitMQ事件代理。

        Args:
            host: Pika主机。
            username: 用于Pika主机认证的用户名。
            password: 用于Pika主机认证的密码。
            port: Pika主机的端口。
            queues: 要声明和发布到的Pika队列。
            should_keep_unpublished_messages: 事件代理是否应该维护一个未发布消息的队列，
                以便在出现错误时稍后发布。
            raise_on_failure: 如果发布失败是否抛出异常。如果为`False`，则继续重试。
            event_loop: 用于运行`async`函数的事件循环。如果为`None`，
                则使用`asyncio.get_event_loop()`获取循环。
            connection_attempts: 在抛出异常之前连接RabbitMQ的尝试次数。
            retry_delay_in_seconds: 连接尝试之间的时间（秒）。
            exchange_name: 队列绑定到的交换器名称。
                如果未提及，则使用默认交换器名称。
        """
        super().__init__()  # 调用父类初始化

        self.host = host  # 主机地址
        self.username = username  # 用户名
        self.password = password  # 密码

        try:
            self.port = int(port)  # 将端口转换为整数
        except ValueError as e:
            raise RasaException("Port could not be converted to integer.") from e

        self.queues = self._get_queues_from_args(queues)  # 获取队列列表
        self.raise_on_failure = raise_on_failure  # 失败时是否抛出异常
        self._connection_attempts = connection_attempts  # 连接尝试次数
        self._retry_delay_in_seconds = retry_delay_in_seconds  # 重试延迟
        self.exchange_name = exchange_name  # 交换器名称

        # 未发布的消息，希望稍后能够发布 🤞
        self._unpublished_events: Deque[Dict[Text, Any]] = deque()
        self.should_keep_unpublished_messages = should_keep_unpublished_messages  # 是否保留未发布消息

        self._loop = event_loop or asyncio.get_event_loop()  # 事件循环
        self._background_tasks: Set[asyncio.Task] = set()  # 后台任务集合

        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None  # 连接对象
        self._exchange: Optional[aio_pika.RobustExchange] = None  # 交换器对象

    @staticmethod
    def _get_queues_from_args(
        queues_arg: Union[List[Text], Tuple[Text, ...], Text, None]  # 队列参数
    ) -> Union[List[Text], Tuple[Text, ...]]:
        """获取此事件代理的队列。

        定义`PikaEventBroker`应发布到的RabbitMQ队列的首选参数是`queues`
        （从Rasa开源版本1.8.2开始）。此方法将来可以删除，
        `self.queues`应该只接收构造函数中`queues`关键字参数的值。

        Args:
            queues_arg: 提供的`queues`参数的值。

        Returns:
            此事件代理发布到的队列。

        Raises:
            如果未找到有效的`queues`参数，则抛出`ValueError`。
        """
        if queues_arg and isinstance(queues_arg, (list, tuple)):
            return queues_arg  # 如果是列表或元组，直接返回

        if queues_arg and isinstance(queues_arg, str):
            # 如果是字符串，记录调试信息并转换为列表
            logger.debug(
                f"Found a string value under the `queues` key of the Pika event broker "
                f"config. Please supply a list of queues under this key, even if it is "
                f"just a single one. See {DOCS_URL_PIKA_EVENT_BROKER}"
            )
            return [queues_arg]

        # 如果没有提供队列参数，发出警告并使用默认队列
        rasa.shared.utils.io.raise_warning(
            f"No `queues` argument provided. It is suggested to "
            f"explicitly specify a queue as described in "
            f"{DOCS_URL_PIKA_EVENT_BROKER}. "
            f"Using the default queue '{DEFAULT_QUEUE_NAME}' for now."
        )

        return [DEFAULT_QUEUE_NAME]  # 返回默认队列

    @classmethod
    async def from_endpoint_config(
        cls,  # 类本身
        broker_config: Optional["EndpointConfig"],  # 代理配置
        event_loop: Optional[AbstractEventLoop] = None,  # 可选的异步事件循环
    ) -> Optional["PikaEventBroker"]:
        """创建代理。更多信息请参见父类。
        
        Args:
            cls: 类本身
            broker_config: 代理配置
            event_loop: 可选的异步事件循环
            
        Returns:
            Pika事件代理实例或None
        """
        if broker_config is None:
            return None  # 如果没有配置，返回None

        broker = cls(broker_config.url, **broker_config.kwargs, event_loop=event_loop)  # 创建代理实例
        await broker.connect()  # 连接到RabbitMQ

        return broker

    async def connect(self) -> None:
        """连接到RabbitMQ。"""
        self._connection = await self._connect()  # 建立连接
        self._connection.reconnect_callbacks.add(self._publish_unpublished_messages)  # 添加重连回调
        logger.info(f"RabbitMQ connection to '{self.host}' was established.")  # 记录连接成功

        channel = await self._connection.channel()  # 打开通道
        logger.debug(
            f"RabbitMQ channel was opened. "
            f"Declaring fanout exchange '{self.exchange_name}'."
        )

        self._exchange = await self._set_up_exchange(channel)  # 设置交换器

    def _configure_url(self) -> Optional[Text]:
        """配置连接到RabbitMQ的URL。
        
        Returns:
            配置好的URL或None
        """
        url = None

        if self.host.startswith("amqp"):  # 如果主机以amqp开头
            parsed_host = urlparse(self.host)  # 解析主机URL

            amqp_user = f"{self.username}:{self.password}"  # 构建认证信息
            if amqp_user not in parsed_host.netloc:  # 如果认证信息不在网络位置中
                url = f"{parsed_host.scheme}://{amqp_user}@{parsed_host.netloc}"
            else:
                url = f"{parsed_host.scheme}://{parsed_host.netloc}"

            if str(self.port) not in url:  # 如果端口不在URL中
                url = f"{url}:{self.port}"

            if parsed_host.path:  # 如果有路径
                url = f"{url}{parsed_host.path}"

            if parsed_host.query:  # 如果有查询参数
                url = f"{url}?{parsed_host.query}"

        return url

    async def _connect(self) -> aio_pika.abc.AbstractRobustConnection:
        """连接到RabbitMQ。
        
        Returns:
            RabbitMQ连接对象
        """
        # `url`参数将优先于`login`或`password`等参数
        url = self._configure_url()

        ssl_options = _create_rabbitmq_ssl_options(self.host)  # 创建SSL选项
        logger.info("Connecting to RabbitMQ ...")  # 记录连接信息

        last_exception: Optional[Exception] = None
        for _ in range(self._connection_attempts):  # 尝试连接指定次数
            try:
                return await aio_pika.connect_robust(  # 建立robust连接
                    url=url,
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    login=self.username,
                    loop=self._loop,
                    ssl=ssl_options is not None,  # 是否使用SSL
                    ssl_options=ssl_options,  # SSL选项
                )
            # 在RabbitMQ稳定之前可能发生各种异常
            except Exception as e:
                last_exception = e
                logger.debug(
                    f"Connecting to '{self.host}' failed with error '{e}'. "
                    f"Trying again."
                )
                await asyncio.sleep(self._retry_delay_in_seconds)  # 等待重试

        last_exception = cast(Exception, last_exception)
        logger.error(
            f"Connecting to '{self.host}' failed with error '{last_exception}'."
        )
        raise last_exception  # 抛出最后一个异常

    def _publish_unpublished_messages(self, *_: Any, **__: Any) -> None:
        """发布未发布的消息。
        
        Args:
            *_: 位置参数（未使用）
            **__: 关键字参数（未使用）
        """
        while self._unpublished_events:  # 当有未发布的消息时
            # 发送未发布的消息
            message = self._unpublished_events.popleft()  # 从队列中取出消息
            self.publish(message)  # 发布消息
            logger.debug(
                f"Published message from queue of unpublished messages. "
                f"Remaining unpublished messages: {len(self._unpublished_events)}."
            )

    async def _set_up_exchange(
        self, channel: aio_pika.RobustChannel  # 通道对象
    ) -> aio_pika.RobustExchange:
        """设置交换器。
        
        Args:
            channel: RabbitMQ通道
            
        Returns:
            交换器对象
        """
        exchange = await channel.declare_exchange(  # 声明交换器
            self.exchange_name, type=aio_pika.ExchangeType.FANOUT  # 扇出类型
        )

        await asyncio.gather(  # 并发绑定所有队列
            *[
                self._bind_queue(queue_name, channel, exchange)  # 绑定队列
                for queue_name in self.queues
            ]
        )

        return exchange

    @staticmethod
    async def _bind_queue(
        queue_name: Text,  # 队列名称
        channel: aio_pika.RobustChannel,  # 通道对象
        exchange: aio_pika.Exchange  # 交换器对象
    ) -> None:
        """绑定队列到交换器。
        
        Args:
            queue_name: 队列名称
            channel: RabbitMQ通道
            exchange: 交换器对象
        """
        queue = await channel.declare_queue(queue_name, durable=True)  # 声明持久队列

        await queue.bind(exchange, "")  # 绑定队列到交换器

    async def close(self) -> None:
        """关闭与RabbitMQ的连接。"""
        if not self._connection:
            return  # 如果没有连接，直接返回

        # 进入上下文管理器什么都不做。退出时关闭通道和连接
        async with self._connection:
            logger.debug("Closing RabbitMQ connection.")  # 记录关闭连接

    def is_ready(self) -> bool:
        """如果已建立连接则返回`True`。
        
        Returns:
            是否已建立连接
        """
        if self._connection is None:
            return False  # 如果没有连接，返回False

        return not self._connection.is_closed  # 返回连接是否未关闭

    def publish(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]] = None  # 事件和头部
    ) -> None:
        """将`event`发布到Pika队列。

        Args:
            event: 要发布的序列化事件。
            headers: 要附加到发布消息的消息头部。头部可以在消费者中从消息的
                `BasicProperties`的`headers`属性中检索。
        """
        # 我们需要保存对此后台任务的引用以确保它不会消失。参见：
        # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
        task: asyncio.Task = self._loop.create_task(self._publish(event, headers))  # 创建后台任务
        self._background_tasks.add(task)  # 添加到后台任务集合
        task.add_done_callback(self._background_tasks.discard)  # 添加完成回调

    async def _publish(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]] = None  # 事件和头部
    ) -> None:
        """异步发布事件。
        
        Args:
            event: 要发布的事件
            headers: 消息头部
        """
        if self._exchange is None:
            return  # 如果没有交换器，直接返回

        reduced_event = rasa.shared.core.events.remove_parse_data(event)  # 移除解析数据

        try:
            await self._exchange.publish(self._message(event, headers), "")  # 发布消息

            structlogger.debug(
                "pika.events.publish",
                event_info="Logging a reduced version of the Pika event",
                rabbitmq_exchange=RABBITMQ_EXCHANGE,
                host=self.host,
                rasa_event=reduced_event,
            )
        except Exception as e:
            # 发布失败，记录错误
            structlogger.error(
                "pika.events.publish.failed",
                event_info="Logging a reduced version of the failed Pika event",
                host=self.host,
                rasa_event=reduced_event,
            )
            if self.should_keep_unpublished_messages:  # 如果应该保留未发布消息
                self._unpublished_events.append(event)  # 添加到未发布消息队列

            if self.raise_on_failure:  # 如果失败时应该抛出异常
                self.close()  # type: ignore[unused-coroutine]  # 关闭连接
                raise e  # 抛出异常

    def _message(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]]  # 事件和头部
    ) -> aio_pika.Message:
        """创建Pika消息。
        
        Args:
            event: 事件字典
            headers: 消息头部
            
        Returns:
            Pika消息对象
        """
        body = json.dumps(event)  # 将事件序列化为JSON
        return aio_pika.Message(
            bytes(body, DEFAULT_ENCODING),  # 消息体（字节）
            headers=headers,  # 消息头部
            app_id=self.rasa_environment,  # 应用ID
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # 持久化模式
        )

    @rasa.shared.utils.common.lazy_property
    def rasa_environment(self) -> Optional[Text]:
        """获取`RASA_ENVIRONMENT`环境变量的值。
        
        Returns:
            Rasa环境变量值
        """
        return os.environ.get("RASA_ENVIRONMENT")


def _create_rabbitmq_ssl_options(
    rabbitmq_host: Optional[Text] = None,  # RabbitMQ主机名
) -> Optional[Dict]:
    """创建RabbitMQ SSL选项。

    需要设置以下环境变量：

        RABBITMQ_SSL_CLIENT_CERTIFICATE - SSL客户端证书的路径（必需）
        RABBITMQ_SSL_CLIENT_KEY - SSL客户端密钥的路径（必需）

    有关如何启用RabbitMQ TLS支持的详细信息，请参见：
    https://www.rabbitmq.com/ssl.html#enabling-tls

    Args:
        rabbitmq_host: RabbitMQ主机名。

    Returns:
        用于RabbitMQ连接的可选SSL参数。
    """
    client_certificate_path = os.environ.get("RABBITMQ_SSL_CLIENT_CERTIFICATE")  # 获取证书路径
    client_key_path = os.environ.get("RABBITMQ_SSL_CLIENT_KEY")  # 获取密钥路径

    if os.environ.get("RABBITMQ_SSL_CA_FILE"):  # 如果设置了CA文件环境变量
        rasa.shared.utils.io.raise_warning(
            "Specifying 'RABBITMQ_SSL_CA_FILE' via "
            "environment variables is no longer supported. Please specify this "
            "through the RabbitMQ URL parameter 'cacertfile' as described here: "
            "https://www.rabbitmq.com/uri-query-parameters.html "
        )

    if os.environ.get("RABBITMQ_SSL_KEY_PASSWORD"):  # 如果设置了密钥密码环境变量
        rasa.shared.utils.io.raise_warning(
            "Specifying 'RABBITMQ_SSL_KEY_PASSWORD' via environment variables is no "
            "longer supported. Please use an unencrypted key file."
        )

    if client_certificate_path and client_key_path:  # 如果证书和密钥路径都存在
        logger.debug(f"Configuring SSL context for RabbitMQ host '{rabbitmq_host}'.")
        return {
            "certfile": client_certificate_path,  # 证书文件
            "client_key_path": client_key_path,  # 客户端密钥路径
            "cert_reqs": ssl.CERT_REQUIRED,  # 证书要求
        }

    return None  # 如果没有SSL配置，返回None
