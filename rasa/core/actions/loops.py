# =============================================================================
# Rasa Core Loops 循环动作模块
# 本模块定义了循环动作的抽象基类，用于实现需要循环执行的动作
# =============================================================================

# 导入标准库模块
from abc import ABC  # 抽象基类
from typing import List, TYPE_CHECKING  # 类型注解和类型检查

# 导入 Rasa 核心模块
from rasa.core.actions.action import Action  # 动作基类
from rasa.shared.core.events import Event, ActiveLoop  # 事件类和活跃循环事件

# 类型检查时的导入（避免循环导入）
if TYPE_CHECKING:
    from rasa.core.channels import OutputChannel  # 输出通道接口
    from rasa.shared.core.domain import Domain  # 领域模型
    from rasa.core.nlg import NaturalLanguageGenerator  # 自然语言生成器
    from rasa.shared.core.trackers import DialogueStateTracker  # 对话状态跟踪器


class LoopAction(Action, ABC):
    """循环动作抽象基类。
    
    循环动作是 Rasa 中用于实现需要循环执行的动作的基类，如表单动作。
    它提供了完整的循环生命周期管理，包括激活、执行、完成检查和停用。
    """

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """运行循环动作的主要方法。
        
        此方法实现了循环动作的完整生命周期：
        1. 检查是否已激活，如果没有则激活
        2. 检查是否完成，如果没有则执行循环逻辑
        3. 如果完成则停用循环
        
        Args:
            output_channel: 输出通道，用于发送消息给用户
            nlg: 自然语言生成器，用于生成响应
            tracker: 对话状态跟踪器，跟踪对话状态
            domain: 领域模型，包含对话配置
            
        Returns:
            执行循环动作产生的事件列表
        """
        events: List[Event] = []  # 初始化事件列表

        # 检查循环是否已激活
        if not await self.is_activated(output_channel, nlg, tracker, domain):
            # 如果未激活，则激活循环
            events += await self._activate_loop(
                output_channel,
                nlg,
                tracker,
                domain,
            )

        # 检查循环是否完成
        if not await self.is_done(output_channel, nlg, tracker, domain, events):
            # 如果未完成，则执行循环逻辑
            events += await self.do(output_channel, nlg, tracker, domain, events)

        # 再次检查循环是否完成
        if await self.is_done(output_channel, nlg, tracker, domain, events):
            # 如果完成，则停用循环
            events += self._default_deactivation_events()  # 添加默认停用事件
            events += await self.deactivate(  # 执行停用逻辑
                output_channel, nlg, tracker, domain, events
            )

        return events

    async def is_activated(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> bool:
        """检查循环是否已激活。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            如果循环已激活则返回 True，否则返回 False
        """
        return tracker.active_loop_name == self.name()  # 检查当前活跃循环名称是否匹配

    def _default_activation_events(self) -> List[Event]:
        """获取默认的激活事件。
        
        默认实现检查表单是否活跃
        
        Returns:
            默认激活事件列表
        """
        return [ActiveLoop(self.name())]  # 创建活跃循环事件

    async def activate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """激活循环。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            激活循环产生的事件列表
            
        Note:
            此方法可以被子类重写以提供自定义激活逻辑
        """
        # 可以被重写
        return []  # 默认返回空事件列表

    async def do(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """执行循环的主要逻辑。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            执行循环逻辑产生的事件列表
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError()  # 子类必须实现此方法

    async def is_done(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> bool:
        """检查循环是否完成。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            如果循环完成则返回 True，否则返回 False
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError()  # 子类必须实现此方法

    def _default_deactivation_events(self) -> List[Event]:
        """获取默认的停用事件。
        
        Returns:
            默认停用事件列表
        """
        return [ActiveLoop(None)]  # 创建停用循环事件

    async def deactivate(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        events_so_far: List[Event],
    ) -> List[Event]:
        """停用循环。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            events_so_far: 到目前为止的事件列表
            
        Returns:
            停用循环产生的事件列表
            
        Note:
            此方法可以被子类重写以提供自定义停用逻辑
        """
        # 可以被重写
        return []  # 默认返回空事件列表

    async def _activate_loop(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """激活循环的内部方法。
        
        Args:
            output_channel: 输出通道
            nlg: 自然语言生成器
            tracker: 对话状态跟踪器
            domain: 领域模型
            
        Returns:
            激活循环产生的事件列表
        """
        # 获取默认激活事件
        events = self._default_activation_events()
        # 执行自定义激活逻辑
        events += await self.activate(output_channel, nlg, tracker, domain)

        return events
