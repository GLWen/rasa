import contextlib  # 上下文管理模块
import json  # JSON处理模块
import logging  # 日志记录模块
from asyncio import AbstractEventLoop  # 异步事件循环抽象基类
from typing import Any, Dict, Optional, Text, Generator  # 类型注解模块

from sqlalchemy.orm import Session  # SQLAlchemy会话类
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta  # SQLAlchemy声明式基类
from sqlalchemy import Column, Integer, String  # SQLAlchemy列类型
from sqlalchemy import Text as SqlAlchemyText  # 避免与typing.Text名称冲突

from rasa.core.brokers.broker import EventBroker  # 事件代理基类
from rasa.utils.endpoints import EndpointConfig  # 端点配置类

logger = logging.getLogger(__name__)  # 创建日志记录器


class SQLEventBroker(EventBroker):
    """将事件保存到SQL数据库中。

    所有事件都将存储在名为`events`的表中。

    """

    Base: DeclarativeMeta = declarative_base()  # 声明式基类

    class SQLBrokerEvent(Base):
        """表示`events`表中一行的ORM模型。"""

        __tablename__ = "events"  # 表名
        id = Column(Integer, primary_key=True)  # 主键ID
        sender_id = Column(String(255))  # 发送者ID
        data = Column(SqlAlchemyText)  # 事件数据（JSON格式）

    def __init__(
        self,
        dialect: Text = "sqlite",  # 数据库方言
        host: Optional[Text] = None,  # 数据库主机
        port: Optional[int] = None,  # 数据库端口
        db: Text = "events.db",  # 数据库名称
        username: Optional[Text] = None,  # 用户名
        password: Optional[Text] = None,  # 密码
    ) -> None:
        """初始化`SQLBrokerEvent`。
        
        Args:
            dialect: 数据库方言，默认为sqlite
            host: 数据库主机地址
            port: 数据库端口
            db: 数据库名称
            username: 数据库用户名
            password: 数据库密码
        """
        from rasa.core.tracker_store import SQLTrackerStore  # 导入SQLTrackerStore
        import sqlalchemy.orm  # 导入SQLAlchemy ORM

        engine_url = SQLTrackerStore.get_db_url(  # 获取数据库URL
            dialect, host, port, db, username, password
        )

        logger.debug(f"SQLEventBroker: Connecting to database: '{engine_url}'.")  # 记录连接信息

        self.engine = sqlalchemy.create_engine(engine_url)  # 创建数据库引擎
        self.Base.metadata.create_all(self.engine)  # 创建所有表
        self.sessionmaker = sqlalchemy.orm.sessionmaker(bind=self.engine)  # 创建会话工厂

    @classmethod
    async def from_endpoint_config(
        cls,  # 类本身
        broker_config: EndpointConfig,  # 代理配置
        event_loop: Optional[AbstractEventLoop] = None,  # 可选的异步事件循环
    ) -> "SQLEventBroker":
        """创建代理。更多信息请参见父类。
        
        Args:
            cls: 类本身
            broker_config: 代理配置
            event_loop: 可选的异步事件循环
            
        Returns:
            SQLEventBroker实例
        """
        return cls(host=broker_config.url, **broker_config.kwargs)  # 使用配置创建实例

    @contextlib.contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """提供围绕一系列操作的事务范围。
        
        Yields:
            SQLAlchemy会话对象
        """
        session = self.sessionmaker()  # 创建会话
        try:
            yield session  # 返回会话
        finally:
            session.close()  # 关闭会话

    def publish(self, event: Dict[Text, Any]) -> None:
        """将JSON格式的Rasa Core事件发布到事件队列中。
        
        Args:
            event: 要发布的事件字典
        """
        with self.session_scope() as session:  # 使用会话上下文管理器
            session.add(  # 添加事件到会话
                self.SQLBrokerEvent(
                    sender_id=event.get("sender_id"),  # 获取发送者ID
                    data=json.dumps(event)  # 将事件序列化为JSON
                )
            )
            session.commit()  # 提交事务
