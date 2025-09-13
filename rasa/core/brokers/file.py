import json  # JSON处理模块
import logging  # 日志记录模块
import typing  # 类型注解模块
from asyncio import AbstractEventLoop  # 异步事件循环抽象基类
from typing import Optional, Text, Dict  # 类型注解

from rasa.core.brokers.broker import EventBroker  # 事件代理基类

if typing.TYPE_CHECKING:  # 类型检查时的导入
    from rasa.utils.endpoints import EndpointConfig  # 端点配置类

logger = logging.getLogger(__name__)  # 创建日志记录器


class FileEventBroker(EventBroker):
    """以JSON格式将事件记录到文件中。

    每行一个事件，每个事件都以JSON格式存储。"""

    DEFAULT_LOG_FILE_NAME = "rasa_event.log"  # 默认日志文件名

    def __init__(self, path: Optional[Text] = None) -> None:
        """初始化文件事件代理。
        
        Args:
            path: 可选的日志文件路径
        """
        self.path = path or self.DEFAULT_LOG_FILE_NAME  # 设置日志文件路径
        self.event_logger = self._event_logger()  # 创建事件日志记录器

    @classmethod
    async def from_endpoint_config(
        cls,  # 类本身
        broker_config: Optional["EndpointConfig"],  # 代理配置
        event_loop: Optional[AbstractEventLoop] = None,  # 可选的异步事件循环
    ) -> Optional["FileEventBroker"]:
        """从端点配置创建代理。更多信息请参见父类。
        
        Args:
            cls: 类本身
            broker_config: 代理配置
            event_loop: 可选的异步事件循环
            
        Returns:
            文件事件代理实例或None
        """
        if broker_config is None:
            return None  # 如果没有配置，返回None

        # noinspection PyArgumentList
        return cls(**broker_config.kwargs)  # 使用配置参数创建实例

    def _event_logger(self) -> logging.Logger:
        """实例化文件日志记录器。
        
        Returns:
            配置好的日志记录器
        """

        logger_file = self.path  # 获取日志文件路径
        # noinspection PyTypeChecker
        query_logger = logging.getLogger("event-logger")  # 创建事件日志记录器
        query_logger.setLevel(logging.INFO)  # 设置日志级别为INFO
        handler = logging.FileHandler(logger_file)  # 创建文件处理器
        handler.setFormatter(logging.Formatter("%(message)s"))  # 设置格式化器，只输出消息
        query_logger.propagate = False  # 禁止向父记录器传播
        query_logger.addHandler(handler)  # 添加处理器

        logger.info(f"Logging events to '{logger_file}'.")  # 记录日志文件路径

        return query_logger  # 返回配置好的日志记录器

    def publish(self, event: Dict) -> None:
        """将事件写入文件。
        
        Args:
            event: 要发布的事件字典
        """

        self.event_logger.info(json.dumps(event))  # 将事件转换为JSON并记录到文件
        self.event_logger.handlers[0].flush()  # 立即刷新文件缓冲区，确保数据写入
