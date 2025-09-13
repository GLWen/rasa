from __future__ import annotations  # 启用延迟注解评估
import logging  # 日志记录模块
from asyncio import AbstractEventLoop  # 异步事件循环抽象基类
from typing import Any, Dict, Text, Optional, Union, TypeVar, Type  # 类型注解模块

import aiormq  # 异步AMQP客户端库

import rasa.shared.utils.common  # Rasa共享通用工具
import rasa.shared.utils.io  # Rasa共享IO工具
from rasa.shared.exceptions import ConnectionException  # 连接异常类
from rasa.utils.endpoints import EndpointConfig  # 端点配置类

logger = logging.getLogger(__name__)  # 创建日志记录器


EB = TypeVar("EB", bound="EventBroker")  # 事件代理类型变量，绑定到EventBroker类


class EventBroker:
    """任何事件代理实现的基础类。"""

    @staticmethod
    async def create(
        obj: Union[EventBroker, EndpointConfig, None],  # 事件代理对象或端点配置
        loop: Optional[AbstractEventLoop] = None,        # 可选的异步事件循环
    ) -> Optional[EventBroker]:
        """创建事件代理的工厂方法。
        
        Args:
            obj: 事件代理对象或端点配置
            loop: 可选的异步事件循环
            
        Returns:
            事件代理实例或None
        """
        if isinstance(obj, EventBroker):
            return obj  # 如果已经是EventBroker实例，直接返回

        import aio_pika.exceptions  # 导入aio_pika异常
        import sqlalchemy.exc  # 导入SQLAlchemy异常

        try:
            # 从端点配置创建事件代理
            return await _create_from_endpoint_config(obj, loop)
        except (
            sqlalchemy.exc.OperationalError,              # SQL操作错误
            aio_pika.exceptions.AMQPConnectionError,      # AMQP连接错误
            aiormq.exceptions.ChannelNotFoundEntity,      # 通道未找到实体错误
            *aio_pika.exceptions.CONNECTION_EXCEPTIONS,   # 其他连接异常
        ) as error:
            # 抛出连接异常
            raise ConnectionException("Cannot connect to event broker.") from error

    @classmethod
    async def from_endpoint_config(
        cls: Type[EB],                                    # 事件代理类类型
        broker_config: EndpointConfig,                    # 代理配置
        event_loop: Optional[AbstractEventLoop] = None,   # 可选的异步事件循环
    ) -> Optional[EB]:
        """从端点配置创建`EventBroker`。

        Args:
            broker_config: 代理的配置
            event_loop: 当前事件循环或`None`

        Returns:
            一个`EventBroker`对象
        """
        raise NotImplementedError(
            "Event broker must implement the `from_endpoint_config` method."
        )

    def publish(self, event: Dict[Text, Any]) -> None:
        """将JSON格式的Rasa Core事件发布到事件队列中。
        
        Args:
            event: 要发布的事件字典
        """
        raise NotImplementedError("Event broker must implement the `publish` method.")

    def is_ready(self) -> bool:
        """确定事件代理是否准备就绪。

        Returns:
            默认为`True`，但子类可以重写此方法
        """
        return True

    async def close(self) -> None:
        """关闭与事件代理的连接。"""
        # 默认实现不执行任何操作
        pass


async def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig],  # 端点配置
    event_loop: Optional[AbstractEventLoop]     # 异步事件循环
) -> Optional[EventBroker]:
    """根据配置实例化事件代理。
    
    Args:
        endpoint_config: 端点配置
        event_loop: 异步事件循环
        
    Returns:
        事件代理实例或None
    """
    if endpoint_config is None:
        broker: Optional[EventBroker] = None  # 如果没有配置，返回None
    elif endpoint_config.type is None or endpoint_config.type.lower() == "pika":
        from rasa.core.brokers.pika import PikaEventBroker  # 导入Pika事件代理

        # 如果未设置类型，则使用默认代理
        broker = await PikaEventBroker.from_endpoint_config(endpoint_config, event_loop)
    elif endpoint_config.type.lower() == "sql":
        from rasa.core.brokers.sql import SQLEventBroker  # 导入SQL事件代理

        broker = await SQLEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "file":
        from rasa.core.brokers.file import FileEventBroker  # 导入文件事件代理

        broker = await FileEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "kafka":
        from rasa.core.brokers.kafka import KafkaEventBroker  # 导入Kafka事件代理

        broker = await KafkaEventBroker.from_endpoint_config(endpoint_config)
    else:
        # 尝试从模块名称加载事件代理
        broker = await _load_from_module_name_in_endpoint_config(endpoint_config)

    if broker:
        logger.debug(f"Instantiated event broker to '{broker.__class__.__name__}'.")
    return broker


async def _load_from_module_name_in_endpoint_config(
    broker_config: EndpointConfig,  # 代理配置
) -> Optional[EventBroker]:
    """根据类名实例化事件代理。
    
    Args:
        broker_config: 代理配置
        
    Returns:
        事件代理实例或None
    """
    try:
        # 从模块路径获取事件代理类
        event_broker_class = rasa.shared.utils.common.class_from_module_path(
            broker_config.type
        )
        # 从端点配置创建事件代理实例
        return await event_broker_class.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        # 记录警告日志
        logger.warning(
            f"The `EventBroker` type '{broker_config.type}' could not be found. "
            f"Not using any event broker. Error: {e}"
        )
        return None
