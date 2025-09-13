# =============================================================================
# 对话管理模块 - 定义对话和轮次的数据结构
# =============================================================================
# 此模块定义了对话的基本数据结构，包括对话（Dialogue）和轮次（Turn）
# 的概念。对话是多个轮次的集合，每个轮次包含用户输入和机器人响应。

# 类型提示导入
from typing import Dict, List, Text, Any, TYPE_CHECKING

# 事件模块导入
import rasa.shared.core.events


# 类型检查时的导入（避免循环导入）
if TYPE_CHECKING:
    from rasa.shared.core.events import Event  # 事件类型


# =============================================================================
# 对话类 - 表示一个完整的对话会话
# =============================================================================

class Dialogue:
    """对话类，包含一系列轮次（Turn）对象。
    
    对话是用户与机器人之间的一次完整交互会话，
    由多个轮次组成，每个轮次包含用户输入和机器人响应。
    """

    def __init__(self, name: Text, events: List["Event"]) -> None:
        """初始化对话对象。
        
        此函数使用对话名称和事件列表初始化对话对象。
        对话名称用于标识特定的对话会话，事件列表记录了
        整个对话过程中发生的所有事件。
        
        Args:
            name: 对话的名称标识符
            events: 对话中发生的事件列表
        """
        self.name = name        # 对话名称
        self.events = events    # 事件列表

    def __str__(self) -> Text:
        """返回对话的字符串表示。
        
        此函数返回对话和轮次的字符串表示，
        用于调试和日志记录。
        
        Returns:
            格式化的对话字符串，包含对话名称和所有事件
        """
        return "Dialogue with name '{}' and turns:\n{}".format(
            self.name,  # 对话名称
            "\n\n".join([f"\t{t}" for t in self.events])  # 格式化的事件列表
        )

    def as_dict(self) -> Dict:
        """将对话转换为字典格式。
        
        此函数返回对话的字典表示，用于序列化。
        字典包含对话名称和所有事件的序列化表示。
        
        Returns:
            包含对话名称和事件列表的字典
        """
        return {
            "events": [event.as_dict() for event in self.events],  # 序列化所有事件
            "name": self.name  # 对话名称
        }

    @classmethod
    def from_parameters(cls, parameters: Dict[Text, Any]) -> "Dialogue":
        """从参数字典创建对话对象。
        
        此方法用于从序列化的对话数据中重建对话对象，
        支持从数据库或文件中恢复对话历史。

        Args:
            parameters: 序列化的对话数据，应包含 'name' 和 'events' 键
                - name: 对话名称
                - events: 事件列表（序列化格式）

        Returns:
            反序列化的对话对象
        """
        return cls(
            parameters.get("name"),  # 获取对话名称
            rasa.shared.core.events.deserialise_events(  # 反序列化事件列表
                parameters.get("events", [])  # 获取事件列表，默认为空列表
            ),
        )
