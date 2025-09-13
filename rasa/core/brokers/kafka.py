import asyncio  # 异步编程模块
import os  # 操作系统接口模块
import json  # JSON处理模块
import logging  # 日志记录模块
import structlog  # 结构化日志模块
import threading  # 线程模块
from asyncio import AbstractEventLoop  # 异步事件循环抽象基类
from typing import Any, Text, List, Optional, Union, Dict, TYPE_CHECKING  # 类型注解模块
import time  # 时间处理模块

from rasa.core.brokers.broker import EventBroker  # 事件代理基类
from rasa.core.exceptions import KafkaProducerInitializationError  # Kafka生产者初始化错误
from rasa.shared.utils.io import DEFAULT_ENCODING  # 默认编码
from rasa.utils.endpoints import EndpointConfig  # 端点配置类
import rasa.shared.utils.common  # Rasa共享通用工具

if TYPE_CHECKING:  # 类型检查时的导入
    from confluent_kafka import KafkaError, Producer, Message  # Confluent Kafka相关类型

logger = logging.getLogger(__name__)  # 创建日志记录器
structlogger = structlog.get_logger()  # 创建结构化日志记录器


class KafkaEventBroker(EventBroker):
    """Kafka事件代理。"""

    def __init__(
        self,
        url: Union[Text, List[Text], None],  # Kafka服务器URL
        topic: Text = "rasa_core_events",  # 主题名称
        client_id: Optional[Text] = None,  # 客户端ID
        partition_by_sender: bool = False,  # 是否按发送者分区
        sasl_username: Optional[Text] = None,  # SASL用户名
        sasl_password: Optional[Text] = None,  # SASL密码
        sasl_mechanism: Optional[Text] = "PLAIN",  # SASL机制
        ssl_cafile: Optional[Text] = None,  # SSL CA文件
        ssl_certfile: Optional[Text] = None,  # SSL证书文件
        ssl_keyfile: Optional[Text] = None,  # SSL密钥文件
        ssl_check_hostname: bool = False,  # SSL主机名检查
        security_protocol: Text = "SASL_PLAINTEXT",  # 安全协议
        **kwargs: Any,  # 其他参数
    ) -> None:
        """Kafka事件代理。

        Args:
            url: 'url[:port]' 字符串（或 'url[:port]' 字符串列表），
                生产者应联系以引导初始集群元数据。这不必是完整的节点列表。
                只需要至少有一个代理响应元数据API请求。
            topic: 要订阅的主题。
            client_id: 此客户端的名称。此字符串在每个请求中传递给服务器，
                可用于识别与此客户端对应的特定服务器端日志条目。
                也提交给 `GroupCoordinator` 用于生产者组管理的日志记录。
            partition_by_sender: 配置消息是否按 sender_id 分区的标志
            sasl_username: 纯文本认证的用户名。
            sasl_password: 纯文本认证的密码。
            sasl_mechanism: 当 security_protocol 配置为 SASL_PLAINTEXT 或 SASL_SSL 时的认证机制。
                有效值为：PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256, SCRAM-SHA-512。默认：`PLAIN`
            ssl_cafile: 用于证书验证的CA文件的可选文件名。
            ssl_certfile: 包含客户端证书以及建立证书真实性所需的任何CA证书的
                pem格式文件的可选文件名。
            ssl_keyfile: 包含客户端私钥的可选文件名。
            ssl_check_hostname: 配置SSL握手是否应验证证书与代理主机名匹配的标志。
            security_protocol: 用于与代理通信的协议。
                有效值为：PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL。
        """
        self.producer: Optional[Producer] = None  # Kafka生产者实例
        self.url = url  # Kafka服务器URL
        self.topic = topic  # 主题名称
        self.client_id = client_id  # 客户端ID
        self.partition_by_sender = partition_by_sender  # 是否按发送者分区
        self.security_protocol = security_protocol.upper()  # 安全协议（大写）
        self.sasl_username = sasl_username  # SASL用户名
        self.sasl_password = sasl_password  # SASL密码
        self.sasl_mechanism = sasl_mechanism  # SASL机制
        self.ssl_cafile = ssl_cafile  # SSL CA文件
        self.ssl_certfile = ssl_certfile  # SSL证书文件
        self.ssl_keyfile = ssl_keyfile  # SSL密钥文件
        self.queue_size = kwargs.get("queue_size")  # 队列大小
        self.ssl_check_hostname = "https" if ssl_check_hostname else None  # SSL主机名检查

        # 异步生产者实现遵循confluent-kafka asyncio示例：
        # https://github.com/confluentinc/confluent-kafka-python/blob/master/examples/asyncio_example.py#L88  # noqa: E501
        self._loop = asyncio.get_event_loop()  # 获取事件循环
        self._cancelled = False  # 取消标志
        self._poll_thread = threading.Thread(target=self._poll_loop)  # 轮询线程
        self._poll_thread.start()  # 启动轮询线程

    @classmethod
    async def from_endpoint_config(
        cls,  # 类本身
        broker_config: EndpointConfig,  # 代理配置
        event_loop: Optional[AbstractEventLoop] = None,  # 可选的异步事件循环
    ) -> Optional["KafkaEventBroker"]:
        """创建代理。更多信息请参见父类。
        
        Args:
            cls: 类本身
            broker_config: 代理配置
            event_loop: 可选的异步事件循环
            
        Returns:
            Kafka事件代理实例或None
        """
        if broker_config is None:
            return None  # 如果没有配置，返回None

        return cls(broker_config.url, **broker_config.kwargs)  # 使用配置创建实例

    def publish(
        self,
        event: Dict[Text, Any],  # 要发布的事件
        retries: int = 60,  # 重试次数
        retry_delay_in_seconds: float = 5,  # 重试延迟（秒）
    ) -> None:
        """发布事件。
        
        Args:
            event: 要发布的事件字典
            retries: 重试次数
            retry_delay_in_seconds: 重试延迟（秒）
        """
        from confluent_kafka import KafkaException  # 导入Kafka异常

        if retries == 1:
            retries = 2  # 确保至少重试一次

        if self.producer is None:
            # 如果生产者未初始化，创建生产者
            self.producer = self._create_producer()
            try:
                self._check_kafka_connection()  # 检查Kafka连接
                logger.debug("Connection to kafka successful.")
            except KafkaException:
                logger.debug("Failed to connect kafka.")
                return
        while retries:
            try:
                self._publish(event)  # 发布事件
                return
            except BufferError as e:
                # 处理缓冲区错误
                logger.error(
                    f"Could not publish message to kafka url '{self.url}'. "
                    f"Failed with error: {e}"
                )
                self.producer.poll(1)  # 轮询生产者
                retries -= 1
            except Exception as e:
                # 处理其他异常
                logger.error(
                    f"Could not publish message to kafka url '{self.url}'. "
                    f"Failed with error: {e}"
                )
                try:
                    self._check_kafka_connection()  # 检查连接
                except KafkaException:
                    logger.debug("Connection to kafka lost, reconnecting...")
                    self.producer = self._create_producer()  # 重新创建生产者
                    try:
                        self._check_kafka_connection()  # 检查新连接
                        logger.debug("Reconnection to kafka successful")
                        self._publish(event)  # 重新发布事件
                        return
                    except KafkaException:
                        pass
                retries -= 1
                time.sleep(retry_delay_in_seconds)  # 等待重试

        logger.error("Failed to publish Kafka event.")  # 所有重试都失败

    def _check_kafka_connection(self) -> None:
        """验证与Kafka的连接。

        Raises:
            KafkaException: 如果Kafka断开连接。
        """
        if self.producer is not None:
            self.producer.list_topics(timeout=5)  # 列出主题以验证连接

    def _get_kafka_config(self) -> Dict[Text, Any]:
        """获取Kafka配置。
        
        Returns:
            Kafka配置字典
        """
        config = {
            "client.id": self.client_id,  # 客户端ID
            "bootstrap.servers": self.url,  # 引导服务器
            "error_cb": kafka_error_callback,  # 错误回调
        }
        if self.queue_size:
            config["queue.buffering.max.messages"] = self.queue_size  # 队列缓冲最大消息数

        if self.security_protocol == "PLAINTEXT":
            # 纯文本协议
            authentication_params: Dict[Text, Any] = {
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SASL_PLAINTEXT":
            # SASL纯文本协议
            authentication_params = {
                "sasl.username": self.sasl_username,  # SASL用户名
                "sasl.password": self.sasl_password,  # SASL密码
                "sasl.mechanism": self.sasl_mechanism,  # SASL机制
                "security.protocol": self.security_protocol,  # 安全协议
            }
        elif self.security_protocol == "SSL":
            # SSL协议
            authentication_params = {
                "ssl.ca.location": self.ssl_cafile,  # SSL CA位置
                "ssl.certificate.location": self.ssl_certfile,  # SSL证书位置
                "ssl.key.location": self.ssl_keyfile,  # SSL密钥位置
                "security.protocol": self.security_protocol,  # 安全协议
            }
        elif self.security_protocol == "SASL_SSL":
            # SASL SSL协议
            authentication_params = {
                "sasl.username": self.sasl_username,  # SASL用户名
                "sasl.password": self.sasl_password,  # SASL密码
                "ssl.ca.location": self.ssl_cafile,  # SSL CA位置
                "ssl.certificate.location": self.ssl_certfile,  # SSL证书位置
                "ssl.key.location": self.ssl_keyfile,  # SSL密钥位置
                "ssl.endpoint.identification.algorithm": self.ssl_check_hostname,  # SSL端点识别算法
                "security.protocol": self.security_protocol,  # 安全协议
                "sasl.mechanism": self.sasl_mechanism,  # SASL机制
            }
        else:
            # 无效的安全协议
            raise ValueError(
                f"Cannot initialise `KafkaEventBroker`: "
                f"Invalid `security_protocol` ('{self.security_protocol}')."
            )

        return {**config, **authentication_params}  # 合并配置和认证参数

    def _create_producer(self) -> "Producer":
        """创建Kafka生产者。
        
        Returns:
            Kafka生产者实例
        """
        import confluent_kafka  # 导入confluent_kafka

        try:
            return confluent_kafka.Producer(self._get_kafka_config())  # 使用配置创建生产者
        except confluent_kafka.KafkaException as e:
            # 捕获Kafka异常并重新抛出
            raise KafkaProducerInitializationError(
                f"Cannot initialise `KafkaEventBroker`: {e}"
            )

    def _publish(self, event: Dict[Text, Any]) -> None:
        """发布事件到Kafka。
        
        Args:
            event: 要发布的事件字典
        """
        if self.partition_by_sender:
            # 如果按发送者分区，使用sender_id作为分区键
            partition_key = bytes(event.get("sender_id"), encoding=DEFAULT_ENCODING)
        else:
            partition_key = None  # 不分区

        headers = []
        if self.rasa_environment:
            # 如果设置了Rasa环境，添加到头部
            headers = [
                (
                    "RASA_ENVIRONMENT",
                    bytes(self.rasa_environment, encoding=DEFAULT_ENCODING),
                )
            ]

        # 移除解析数据以减少事件大小
        reduced_event = rasa.shared.core.events.remove_parse_data(event)
        structlogger.debug(
            "kafka.publish.event",
            event_info="Logging a reduced version of the Kafka event",
            topic=self.topic,
            rasa_event=reduced_event,
            partition_key=partition_key,
            headers=headers,
        )

        # 序列化事件为JSON并编码
        serialized_event = json.dumps(event).encode(DEFAULT_ENCODING)

        if self.producer is not None:
            # 使用生产者发布事件
            self.producer.produce(
                self.topic,  # 主题
                value=serialized_event,  # 事件值
                key=partition_key,  # 分区键
                headers=headers,  # 头部
                on_delivery=delivery_report,  # 交付报告回调
            )

    async def close(self) -> None:
        """关闭Kafka事件代理。"""
        self._cancelled = True  # 设置取消标志
        self._poll_thread.join()  # 等待轮询线程结束
        if self.producer:
            self.producer.flush()  # 刷新生产者缓冲区

    @rasa.shared.utils.common.lazy_property
    def rasa_environment(self) -> Optional[Text]:
        """获取`RASA_ENVIRONMENT`环境变量的值。
        
        Returns:
            Rasa环境变量值
        """
        return os.environ.get("RASA_ENVIRONMENT", "RASA_ENVIRONMENT_NOT_SET")

    def _poll_loop(self) -> None:
        """轮询生产者事件。

        需要触发传递给produce方法的on_delivery回调。
        """
        if self.producer is not None:
            while not self._cancelled:  # 当未取消时继续轮询
                self.producer.poll(0.1)  # 轮询生产者，超时0.1秒


def kafka_error_callback(err: "KafkaError") -> None:
    """Kafka错误回调。

    从此回调引发的任何异常都将从触发flush()调用中重新引发。
    """
    from confluent_kafka import KafkaException, KafkaError  # 导入Kafka异常

    # 处理认证/连接相关问题，可能指向配置错误
    if (
        err.code() == KafkaError._ALL_BROKERS_DOWN  # 所有代理都关闭
        or err.code() == KafkaError._AUTHENTICATION  # 认证错误
        or err.code() == KafkaError._MAX_POLL_EXCEEDED  # 最大轮询超出
    ):
        raise KafkaException(err)  # 抛出Kafka异常
    else:
        logger.warning("A KafkaError has been raised.", exc_info=True)  # 记录警告


def delivery_report(err: Exception, msg: "Message") -> None:
    """报告消息传递的失败或成功。

    Args:
        err (KafkaError): 成功时为None，失败时发生的错误。
        msg (Message): 已生产或失败的消息。
    """
    if err is not None:
        # 传递失败
        logger.error(f"Delivery failed for User record {msg.key()}: {err}")
        return

    # 传递成功
    logger.info(
        f"User record {msg.key()} successfully produced to "
        f"{msg.topic()} [{msg.partition()}] at offset {msg.offset()}."
    )
