# 导入未来版本的注解支持，确保类型注解的兼容性
from __future__ import annotations
# 导入异步IO模块，用于异步操作
import asyncio
# 导入异步上下文管理器，用于资源管理
from contextlib import asynccontextmanager
# 导入JSON处理模块，用于序列化和反序列化
import json
# 导入日志模块，用于记录日志信息
import logging
# 导入操作系统接口，用于环境变量等操作
import os

# 导入类型提示相关的类型
from typing import AsyncGenerator, Dict, Optional, Text, Union

# 导入Rasa异常类
from rasa.shared.exceptions import RasaException, ConnectionException
# 导入Rasa共享通用工具
import rasa.shared.utils.common
# 导入默认锁生存时间常量
from rasa.core.constants import DEFAULT_LOCK_LIFETIME
# 导入票据锁类
from rasa.core.lock import TicketLock
# 导入端点配置类
from rasa.utils.endpoints import EndpointConfig

# 创建日志记录器
logger = logging.getLogger(__name__)


def _get_lock_lifetime() -> int:
    """获取锁的生存时间。
    
    Returns:
        从环境变量TICKET_LOCK_LIFETIME获取的值，如果未设置则使用默认值
    """
    return int(os.environ.get("TICKET_LOCK_LIFETIME", 0)) or DEFAULT_LOCK_LIFETIME


# 锁的生存时间（秒）
LOCK_LIFETIME = _get_lock_lifetime()
# 默认套接字超时时间（秒）
DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10

# Redis锁存储的默认键前缀
DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX = "lock:"


# noinspection PyUnresolvedReferences
class LockError(RasaException):
    """当无法获取锁时抛出的异常。

    Attributes:
         message (str): 解释哪个`conversation_id`引发了错误
    """

    pass


class LockStore:
    """票据锁的基类。"""

    @staticmethod
    def create(obj: Union[LockStore, EndpointConfig, None]) -> LockStore:
        """创建锁存储的工厂方法。"""
        # 如果对象已经是LockStore实例，直接返回
        if isinstance(obj, LockStore):
            return obj

        try:
            # 尝试从端点配置创建锁存储
            return _create_from_endpoint_config(obj)
        except ConnectionError as error:
            # 抛出连接异常
            raise ConnectionException("Cannot connect to lock store.") from error

    @staticmethod
    def create_lock(conversation_id: Text) -> TicketLock:
        """为`conversation_id`创建新的`TicketLock`。
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            新创建的票据锁实例
        """
        return TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """从存储中获取`conversation_id`的锁。
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            找到的票据锁，如果不存在则返回None
        """
        raise NotImplementedError

    def delete_lock(self, conversation_id: Text) -> None:
        """从存储中删除`conversation_id`的锁。
        
        Args:
            conversation_id: 对话ID
        """
        raise NotImplementedError

    def save_lock(self, lock: TicketLock) -> None:
        """将`lock`提交到存储。
        
        Args:
            lock: 要保存的票据锁
        """
        raise NotImplementedError

    def issue_ticket(
        self, conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME
    ) -> int:
        """为与`conversation_id`关联的锁发行具有`lock_lifetime`的新票据。

        如果没有找到锁，则创建一个新的。

        Args:
            conversation_id: 对话ID
            lock_lifetime: 锁的生存时间（秒）

        Returns:
            新发行的票据编号
        """
        logger.debug(f"Issuing ticket for conversation '{conversation_id}'.")
        try:
            # 获取或创建锁
            lock = self.get_or_create_lock(conversation_id)
            # 发行票据
            ticket = lock.issue_ticket(lock_lifetime)
            # 保存锁状态
            self.save_lock(lock)

            return ticket
        except Exception as e:
            # 抛出锁错误
            raise LockError(f"Error while acquiring lock. Error:\n{e}")

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: float = LOCK_LIFETIME,
        wait_time_in_seconds: float = 1,
    ) -> AsyncGenerator[TicketLock, None]:
        """为`conversation_id`获取具有生存时间`lock_lifetime`的锁。

        尝试获取锁，每次尝试之间等待`wait_time_in_seconds`秒。
        如果锁已过期则抛出`LockError`。

        Args:
            conversation_id: 对话ID
            lock_lifetime: 锁的生存时间（秒）
            wait_time_in_seconds: 重试间隔时间（秒）

        Yields:
            获取到的票据锁
        """
        # 发行票据
        ticket = self.issue_ticket(conversation_id, lock_lifetime)
        try:
            # 异步获取锁并返回
            yield await self._acquire_lock(
                conversation_id, ticket, wait_time_in_seconds
            )
        finally:
            # 清理资源
            self.cleanup(conversation_id, ticket)

    async def _acquire_lock(
        self, conversation_id: Text, ticket: int, wait_time_in_seconds: float
    ) -> TicketLock:
        """异步获取锁。
        
        Args:
            conversation_id: 对话ID
            ticket: 票据编号
            wait_time_in_seconds: 重试间隔时间（秒）
            
        Returns:
            获取到的票据锁
            
        Raises:
            LockError: 如果无法获取锁
        """
        logger.debug(f"Acquiring lock for conversation '{conversation_id}'.")
        while True:
            # 每次迭代都获取锁，因为锁可能不再存在
            lock = self.get_lock(conversation_id)

            # 如果锁不再存在（已过期），退出循环
            if not lock:
                break

            # 如果锁未被锁定，则获取锁
            if not lock.is_locked(ticket):
                logger.debug(f"Acquired lock for conversation '{conversation_id}'.")
                return lock

            # 计算在此票据之前有多少个票据
            items_before_this = ticket - (lock.now_serving or 0)

            logger.debug(
                f"Failed to acquire lock for conversation ID '{conversation_id}' "
                f"because {items_before_this} other item(s) for this "
                f"conversation ID have to be finished processing first. "
                f"Retrying in {wait_time_in_seconds} seconds ..."
            )

            # 休眠并更新锁
            await asyncio.sleep(wait_time_in_seconds)
            self.update_lock(conversation_id)

        # 如果无法获取锁，抛出异常
        raise LockError(
            f"Could not acquire lock for conversation_id '{conversation_id}'."
        )

    def update_lock(self, conversation_id: Text) -> None:
        """获取`conversation_id`的锁，移除过期票据并保存锁。
        
        Args:
            conversation_id: 对话ID
        """
        lock = self.get_lock(conversation_id)
        if lock:
            # 移除过期票据
            lock.remove_expired_tickets()
            # 保存更新后的锁
            self.save_lock(lock)

    def get_or_create_lock(self, conversation_id: Text) -> TicketLock:
        """获取`conversation_id`的现有锁。

        如果不存在，则创建一个新的。

        Args:
            conversation_id: 对话ID
            
        Returns:
            现有的或新创建的票据锁
        """
        # 尝试获取现有锁
        existing_lock = self.get_lock(conversation_id)

        if existing_lock:
            return existing_lock

        # 如果不存在，创建新锁
        return self.create_lock(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        """返回是否有人正在等待此`conversation_id`的锁。
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            如果有人等待则返回True，否则返回False
        """
        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def finish_serving(self, conversation_id: Text, ticket_number: int) -> None:
        """完成对`conversation_id`的`ticket_number`票据的服务。

        从锁中移除票据并保存锁。

        Args:
            conversation_id: 对话ID
            ticket_number: 票据编号
        """
        lock = self.get_lock(conversation_id)
        if lock:
            # 从锁中移除票据
            lock.remove_ticket_for(ticket_number)
            # 保存更新后的锁
            self.save_lock(lock)

    def cleanup(self, conversation_id: Text, ticket_number: int) -> None:
        """如果没有人等待，则移除`conversation_id`的锁。
        
        Args:
            conversation_id: 对话ID
            ticket_number: 票据编号
        """
        # 完成票据服务
        self.finish_serving(conversation_id, ticket_number)
        # 如果没有人在等待，删除锁
        if not self.is_someone_waiting(conversation_id):
            self.delete_lock(conversation_id)

    @staticmethod
    def _log_deletion(conversation_id: Text, deletion_successful: bool) -> None:
        """记录删除操作的日志。
        
        Args:
            conversation_id: 对话ID
            deletion_successful: 删除是否成功
        """
        if deletion_successful:
            logger.debug(f"Deleted lock for conversation '{conversation_id}'.")
        else:
            logger.debug(f"Could not delete lock for conversation '{conversation_id}'.")


class RedisLockStore(LockStore):
    """票据锁的Redis存储。"""

    def __init__(
        self,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 1,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        use_ssl: bool = False,
        ssl_certfile: Optional[Text] = None,
        ssl_keyfile: Optional[Text] = None,
        ssl_ca_certs: Optional[Text] = None,
        key_prefix: Optional[Text] = None,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_IN_SECONDS,
    ) -> None:
        """创建使用Redis进行持久化的锁存储。

        Args:
            host: Redis服务器的主机地址。
            port: Redis服务器的端口。
            db: Rasa开源版应该使用的Redis数据库名称。
            username: 用于Redis数据库身份验证的用户名。
            password: 用于Redis数据库身份验证的密码。
            use_ssl: 如果应该使用SSL连接到Redis则为`True`。
            ssl_certfile: SSL证书文件的路径。
            ssl_keyfile: SSL私钥文件的路径。
            ssl_ca_certs: SSL CA证书文件的路径。
            key_prefix: 要添加到锁存储使用的所有键的前缀。必须是字母数字。
            socket_timeout: 超时时间（秒），如果Redis在`socket_timeout`秒内没有响应则抛出异常。
        """
        import redis

        # 创建Redis连接
        self.red = redis.StrictRedis(
            host=host,
            port=int(port),
            db=int(db),
            username=username,
            password=password,
            ssl=use_ssl,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_ca_certs=ssl_ca_certs,
            socket_timeout=socket_timeout,
        )

        # 设置键前缀
        self.key_prefix = DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX
        if key_prefix:
            logger.debug(f"Setting non-default redis key prefix: '{key_prefix}'.")
            self._set_key_prefix(key_prefix)

        # 调用父类初始化
        super().__init__()

    def _set_key_prefix(self, key_prefix: Text) -> None:
        """设置键前缀。
        
        Args:
            key_prefix: 要设置的键前缀
        """
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            # 如果前缀是字母数字，则使用它
            self.key_prefix = key_prefix + ":" + DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX
        else:
            # 否则发出警告并使用默认前缀
            logger.warning(
                f"Omitting provided non-alphanumeric redis key prefix: '{key_prefix}'. "
                f"Using default '{self.key_prefix}' instead."
            )

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """检索锁（更多信息请参见父类文档字符串）。
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            找到的票据锁，如果不存在则返回None
        """
        # 从Redis获取序列化的锁
        serialised_lock = self.red.get(self.key_prefix + conversation_id)
        if serialised_lock:
            # 反序列化为票据锁对象
            return TicketLock.from_dict(json.loads(serialised_lock))

        return None

    def delete_lock(self, conversation_id: Text) -> None:
        """删除对话ID的锁。
        
        Args:
            conversation_id: 对话ID
        """
        # 从Redis删除锁
        deletion_successful = self.red.delete(self.key_prefix + conversation_id)
        # 记录删除操作
        self._log_deletion(conversation_id, deletion_successful)

    def save_lock(self, lock: TicketLock) -> None:
        """保存锁到Redis。
        
        Args:
            lock: 要保存的票据锁
        """
        # 将锁序列化并保存到Redis
        self.red.set(self.key_prefix + lock.conversation_id, lock.dumps())


class InMemoryLockStore(LockStore):
    """票据锁的内存存储。"""

    def __init__(self) -> None:
        """初始化锁字典。"""
        # 创建对话锁字典
        self.conversation_locks: Dict[Text, TicketLock] = {}
        # 调用父类初始化
        super().__init__()

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """如果存在则获取对话的锁。
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            找到的票据锁，如果不存在则返回None
        """
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text) -> None:
        """删除对话的锁。
        
        Args:
            conversation_id: 对话ID
        """
        # 从字典中弹出锁
        deleted_lock = self.conversation_locks.pop(conversation_id, None)
        # 记录删除操作
        self._log_deletion(
            conversation_id, deletion_successful=deleted_lock is not None
        )

    def save_lock(self, lock: TicketLock) -> None:
        """在存储中保存锁。
        
        Args:
            lock: 要保存的票据锁
        """
        # 将锁保存到字典中
        self.conversation_locks[lock.conversation_id] = lock


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
) -> LockStore:
    """给定端点配置，创建适当的`LockStore`对象。
    
    Args:
        endpoint_config: 端点配置
        
    Returns:
        创建的锁存储实例
    """
    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        # 如果没有设置锁存储类型，这是默认类型
        lock_store: LockStore = InMemoryLockStore()
    elif endpoint_config.type == "redis":
        # 创建Redis锁存储
        lock_store = RedisLockStore(host=endpoint_config.url, **endpoint_config.kwargs)
    else:
        # 从模块名加载锁存储
        lock_store = _load_from_module_name_in_endpoint_config(endpoint_config)

    logger.debug(f"Connected to lock store '{lock_store.__class__.__name__}'.")

    return lock_store


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> LockStore:
    """基于类名检索`LockStore`。
    
    Args:
        endpoint_config: 端点配置
        
    Returns:
        创建的锁存储实例
        
    Raises:
        Exception: 如果无法找到或创建锁存储类
    """
    try:
        # 从模块路径获取锁存储类
        lock_store_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return lock_store_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        # 抛出异常
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a `LockStore` "
            f"instance. Error: {e}"
        )
