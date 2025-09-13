# 导入JSON处理模块，用于序列化和反序列化票据数据
import json
# 导入日志模块，用于记录日志信息
import logging
# 导入双端队列，用于管理票据队列
from collections import deque

# 导入时间模块，用于处理票据过期时间
import time
# 导入类型提示相关的类型
from typing import Text, Optional, Union, Deque, Dict, Any

# 创建日志记录器
logger = logging.getLogger(__name__)

# 当没有票据存在时，最新发行票据的索引值
NO_TICKET_ISSUED = -1


class Ticket:
    """票据类，表示一个访问对话ID的票据。"""
    
    def __init__(self, number: int, expires: float) -> None:
        """初始化票据。
        
        Args:
            number: 票据编号
            expires: 票据过期时间（时间戳）
        """
        self.number = number      # 票据编号
        self.expires = expires    # 票据过期时间

    def has_expired(self) -> bool:
        """检查票据是否已过期。
        
        Returns:
            如果当前时间超过过期时间则返回True，否则返回False
        """
        return time.time() > self.expires

    def as_dict(self) -> Dict[Text, Any]:
        """将票据转换为字典格式。
        
        Returns:
            包含票据编号和过期时间的字典
        """
        return dict(number=self.number, expires=self.expires)

    def dumps(self) -> Text:
        """返回票据字典的JSON字符串。
        
        Returns:
            票据字典的JSON序列化字符串
        """
        return json.dumps(self.as_dict())

    @classmethod
    def from_dict(cls, data: Dict[Text, Union[int, float]]) -> "Ticket":
        """从字典创建票据。
        
        Args:
            data: 包含票据数据的字典
            
        Returns:
            新创建的票据实例
        """
        return cls(number=data["number"], expires=data["expires"])

    def __repr__(self) -> Text:
        """返回票据的字符串表示。"""
        return f"Ticket(number: {self.number}, expires: {self.expires})"


class TicketLock:
    """票据锁机制，通过发行票据来管理对对话ID的访问。

    票据按照请求的顺序发行。票据锁算法的详细解释可以在以下链接找到：
    http://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf#page=13
    """

    def __init__(
        self, conversation_id: Text, tickets: Optional[Deque[Ticket]] = None
    ) -> None:
        """初始化票据锁。
        
        Args:
            conversation_id: 对话ID
            tickets: 票据队列，如果为None则创建空队列
        """
        self.conversation_id = conversation_id  # 对话ID
        self.tickets = tickets or deque()       # 票据队列

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "TicketLock":
        """从字典创建票据锁。
        
        Args:
            data: 包含票据锁数据的字典
            
        Returns:
            新创建的票据锁实例
        """
        # 从JSON字符串列表创建票据对象列表
        tickets = [Ticket.from_dict(json.loads(d)) for d in data.get("tickets", [])]
        return cls(data.get("conversation_id"), deque(tickets))

    def dumps(self) -> Text:
        """返回票据锁的JSON字符串。
        
        Returns:
            票据锁的JSON序列化字符串
        """
        # 将所有票据转换为JSON字符串
        tickets = [ticket.dumps() for ticket in self.tickets]
        return json.dumps(dict(conversation_id=self.conversation_id, tickets=tickets))

    def is_locked(self, ticket_number: int) -> bool:
        """检查指定票据编号是否被锁定。

        Args:
            ticket_number: 要检查的票据编号

        Returns:
            如果当前服务的票据不等于指定票据则返回True（表示被锁定）
        """
        return self.now_serving != ticket_number

    def issue_ticket(self, lifetime: float) -> int:
        """发行新票据并返回其编号。
        
        Args:
            lifetime: 票据生存时间（秒）
            
        Returns:
            新发行票据的编号
        """
        # 先移除过期的票据
        self.remove_expired_tickets()
        # 生成新的票据编号（比最后发行的票据编号大1）
        number = self.last_issued + 1
        # 创建新票据，过期时间为当前时间加上生存时间
        ticket = Ticket(number, time.time() + lifetime)
        # 将新票据添加到队列末尾
        self.tickets.append(ticket)

        return number

    def remove_expired_tickets(self) -> None:
        """移除过期的票据。"""
        # 遍历票据列表的副本，这样可以在遍历时安全地移除元素
        for ticket in list(self.tickets):
            if ticket.has_expired():
                self.tickets.remove(ticket)

    @property
    def last_issued(self) -> int:
        """返回最后添加的票据编号。

        Returns:
            最后添加的票据编号。如果没有票据存在则返回`NO_TICKET_ISSUED`。
        """
        # 获取队列中最后一个票据的编号（索引-1）
        ticket_number = self._ticket_number_for(-1)

        return ticket_number if ticket_number is not None else NO_TICKET_ISSUED

    @property
    def now_serving(self) -> Optional[int]:
        """获取下一个要服务的票据编号。

        Returns:
            下一个要服务的票据编号。如果没有票据存在则返回0。
        """
        return self._ticket_number_for(0) or 0

    def _ticket_number_for(self, ticket_index: int) -> Optional[int]:
        """获取指定索引位置的票据编号。

        Args:
            ticket_index: 票据在队列中的索引位置

        Returns:
            指定索引位置的票据编号。如果没有票据或索引超出范围则返回None。
        """
        # 先移除过期的票据
        self.remove_expired_tickets()

        try:
            # 尝试获取指定索引位置的票据编号
            return self.tickets[ticket_index].number
        except IndexError:
            # 如果索引超出范围，返回None
            return None

    def _ticket_for_ticket_number(self, ticket_number: int) -> Optional[Ticket]:
        """根据票据编号查找对应的票据。
        
        Args:
            ticket_number: 要查找的票据编号
            
        Returns:
            找到的票据对象，如果不存在则返回None
        """
        # 先移除过期的票据
        self.remove_expired_tickets()

        # 在票据队列中查找指定编号的票据
        return next((t for t in self.tickets if t.number == ticket_number), None)

    def is_someone_waiting(self) -> bool:
        """检查是否有人在等待锁变为可用。

        Returns:
            如果票据队列长度大于0则返回True（表示有人在等待）
        """
        return len(self.tickets) > 0

    def remove_ticket_for(self, ticket_number: int) -> None:
        """移除指定编号的票据。
        
        Args:
            ticket_number: 要移除的票据编号
        """
        # 查找指定编号的票据
        ticket = self._ticket_for_ticket_number(ticket_number)
        # 如果找到票据，则从队列中移除
        if ticket:
            self.tickets.remove(ticket)
