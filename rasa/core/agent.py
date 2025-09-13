# =============================================================================
# Rasa Agent 模块 - 对话代理的核心实现
# =============================================================================
# 此模块实现了 Rasa 的 Agent 类，提供了对话机器人的核心功能，
# 包括模型加载、消息处理、动作预测、对话管理等。

# 标准库导入
from __future__ import annotations  # 启用延迟注解评估，允许在类型提示中使用前向引用
from asyncio import AbstractEventLoop, CancelledError  # 异步事件循环和取消异常处理
import functools  # 函数工具，用于装饰器等高级函数操作
import logging  # 日志记录模块，用于记录程序运行状态
import os  # 操作系统接口，用于文件系统操作
from pathlib import Path  # 路径处理，提供面向对象的文件系统路径操作
from typing import Any, Callable, Dict, List, Optional, Text, Union  # 类型提示，用于静态类型检查
import uuid  # UUID 生成，用于创建唯一标识符

# 第三方库导入
import aiohttp  # 异步 HTTP 客户端库，用于异步网络请求
from aiohttp import ClientError  # HTTP 客户端错误异常类

# Rasa 内部模块导入
from rasa.core import jobs  # 任务调度模块，用于管理后台任务
from rasa.core.channels.channel import OutputChannel, UserMessage  # 通道和用户消息类
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT  # 默认请求超时时间常量
from rasa.core.http_interpreter import RasaNLUHttpInterpreter  # HTTP NLU 解释器，用于远程NLU服务
from rasa.shared.core.domain import Domain  # 对话域类，定义对话系统的所有组件
from rasa.core.exceptions import AgentNotReady  # 代理未就绪异常类
from rasa.shared.constants import DEFAULT_SENDER_ID  # 默认发送者ID常量
from rasa.core.lock_store import InMemoryLockStore, LockStore  # 锁存储相关类，用于并发控制
from rasa.core.nlg import NaturalLanguageGenerator, TemplatedNaturalLanguageGenerator  # 自然语言生成器
from rasa.core.policies.policy import PolicyPrediction  # 策略预测类，用于动作预测
from rasa.core.processor import MessageProcessor  # 消息处理器，核心消息处理逻辑
from rasa.core.tracker_store import FailSafeTrackerStore, InMemoryTrackerStore  # 跟踪器存储实现
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity  # 对话状态跟踪器和事件详细程度
from rasa.exceptions import ModelNotFound  # 模型未找到异常类
from rasa.nlu.utils import is_url  # URL 检查工具函数
from rasa.shared.exceptions import RasaException  # Rasa 基础异常类
import rasa.shared.utils.io  # 共享 IO 工具模块
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name  # 临时目录工具
from rasa.utils.endpoints import EndpointConfig  # 端点配置类

# 重复导入（用于类型提示）
from rasa.core.tracker_store import TrackerStore  # 跟踪器存储基类
from rasa.core.utils import AvailableEndpoints  # 可用端点配置类

# 日志记录器
logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


# =============================================================================
# 模型加载函数
# =============================================================================

async def load_from_server(agent: Agent, model_server: EndpointConfig) -> Agent:
    """从服务器加载持久化模型。
    
    此函数负责从远程服务器加载模型，支持自动更新机制。
    首先拉取一次模型，然后安排定期拉取任务。
    
    Args:
        agent: 要更新的代理实例
        model_server: 模型服务器端点配置
        
    Returns:
        更新后的代理实例
    """
    # 我们先拉取一次模型，然后安排定期任务。
    # 这种方法的优势是我们可以确保在此函数完成后
    # 有一个模型 -> 允许在启动服务器的 `/status` 端点上
    # 进行适当的"存活"检查。如果服务器已启动，
    # 我们可以确保它也已经加载（或尝试加载）了一个模型。
    await _update_model_from_server(model_server, agent)  # 立即从服务器更新模型

    # 获取拉取间隔时间，默认为100秒
    wait_time_between_pulls = model_server.kwargs.get("wait_time_between_pulls", 100)  # 从配置中获取拉取间隔时间

    if wait_time_between_pulls:  # 如果设置了拉取间隔时间
        # 每 `wait_time_between_pulls` 秒持续拉取模型
        await _schedule_model_pulling(model_server, int(wait_time_between_pulls), agent)  # 安排定期拉取任务

    return agent  # 返回更新后的代理实例


def _load_and_set_updated_model(
    agent: Agent, model_directory: Text, fingerprint: Text
) -> None:
    """将持久化模型加载到内存中并在代理上设置模型。

    Args:
        agent: 要用新模型更新的 `Agent` 实例
        model_directory: Rasa 模型目录
        fingerprint: 在 `model_directory` 处提供的模型的指纹
    """
    logger.debug(f"发现具有指纹 {fingerprint} 的新模型。正在加载...")  # 记录发现新模型的日志
    # 加载模型到代理
    agent.load_model(model_directory, fingerprint)  # 调用代理的模型加载方法

    logger.debug("完成将代理更新为新模型。")  # 记录模型更新完成的日志


async def _update_model_from_server(model_server: EndpointConfig, agent: Agent) -> None:
    """从URL加载压缩的Rasa Core模型并更新传递的代理。
    
    此函数负责从远程服务器下载模型文件，检查是否有新版本，
    如果有则更新代理的模型。
    
    Args:
        model_server: 模型服务器端点配置
        agent: 要更新的代理实例
    """
    # 验证URL格式
    if not is_url(model_server.url):  # 检查URL格式是否有效
        raise aiohttp.InvalidURL(model_server.url)  # 如果URL无效则抛出异常

    # 使用临时目录下载模型
    with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:  # 创建临时目录上下文管理器
        try:
            # 拉取模型并获取指纹
            new_fingerprint = await _pull_model_and_fingerprint(  # 异步拉取模型并获取指纹
                model_server, agent.fingerprint, temporary_directory
            )

            if new_fingerprint:  # 如果获取到新的模型指纹
                # 如果有新模型，加载并设置
                _load_and_set_updated_model(agent, temporary_directory, new_fingerprint)  # 加载新模型到代理
            else:
                logger.debug(f"在URL {model_server.url} 处未找到新模型")  # 记录未找到新模型的日志
        except Exception:  # skipcq: PYL-W0703
            # TODO: 使此异常更具体，可能为每个异常打印不同的日志
            logger.exception(  # 记录异常信息
                "更新模型失败。将保持加载之前的模型。"
            )


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, fingerprint: Optional[Text], model_directory: Text
) -> Optional[Text]:
    """查询模型服务器。

    Args:
        model_server: 模型服务器端点信息
        fingerprint: 当前模型指纹
        model_directory: 下载模型到的目录

    Returns:
        响应 <ETag> 头的值，包含模型哈希。
        如果未找到新模型则返回 `None`
    """
    # 设置条件请求头，如果指纹匹配则不返回内容
    headers = {"If-None-Match": fingerprint}  # 设置条件请求头，用于检查模型是否有更新

    logger.debug(f"从服务器 {model_server.url} 请求模型...")  # 记录请求模型的日志

    # 使用模型服务器的会话
    async with model_server.session() as session:  # 创建HTTP会话上下文管理器
        try:
            # 组合请求参数
            params = model_server.combine_parameters()  # 组合请求参数
            # 发送GET请求
            async with session.request(  # 发送异步HTTP请求
                "GET",
                model_server.url,
                timeout=DEFAULT_REQUEST_TIMEOUT,  # 设置请求超时时间
                headers=headers,  # 设置请求头
                params=params,  # 设置请求参数
            ) as resp:

                # 检查响应状态码
                if resp.status in [204, 304]:  # 如果服务器返回无内容或未修改状态码
                    logger.debug(
                        "模型服务器返回 {} 状态码，"
                        "表示没有新模型可用。 "
                        "当前指纹: {}"
                        "".format(resp.status, fingerprint)
                    )
                    return None  # 没有新模型，返回None
                elif resp.status == 404:  # 如果服务器返回未找到状态码
                    logger.debug(
                        "模型服务器在请求的端点 '{}' 处找不到模型。 "
                        "可能没有训练模型，或者请求的标签尚未分配。"
                        .format(model_server.url)
                    )
                    return None  # 模型不存在，返回None
                elif resp.status != 200:  # 如果服务器返回其他错误状态码
                    logger.debug(
                        "尝试从服务器获取模型，但服务器响应 "
                        "状态码为 {}。稍后重试..."
                        "".format(resp.status)
                    )
                    return None  # 请求失败，返回None

                # 获取模型文件名，默认为 model.tar.gz
                model_path = Path(model_directory) / resp.headers.get(  # 构建模型文件路径
                    "filename", "model.tar.gz"  # 从响应头获取文件名，默认为model.tar.gz
                )
                # 将模型内容写入文件
                with open(model_path, "wb") as file:  # 以二进制写入模式打开文件
                    file.write(await resp.read())  # 异步读取响应内容并写入文件

                logger.debug("模型已保存到 '{}'".format(os.path.abspath(model_path)))  # 记录模型保存路径

                # 返回新指纹
                return resp.headers.get("ETag")  # 返回ETag头作为模型指纹

        except aiohttp.ClientError as e:  # 捕获HTTP客户端错误
            logger.debug(
                "尝试从服务器获取模型，但无法连接到服务器。稍后重试... "
                "错误: {}.".format(e)
            )
            return None  # 连接失败，返回None


async def _run_model_pulling_worker(model_server: EndpointConfig, agent: Agent) -> None:
    """模型拉取工作器。
    
    此函数在后台任务中运行，定期从服务器拉取模型更新。
    
    Args:
        model_server: 模型服务器端点配置
        agent: 要更新的代理实例
    """
    # noinspection PyBroadException
    try:
        # 从服务器更新模型
        await _update_model_from_server(model_server, agent)  # 异步更新模型
    except CancelledError:  # 捕获任务取消异常
        # 任务被取消
        logger.warning("停止模型拉取（已取消）。")  # 记录任务取消的警告日志
    except ClientError:  # 捕获客户端错误异常
        # 客户端错误
        logger.exception(  # 记录异常信息
            "获取模型时发生异常。无论如何继续..."
        )


async def _schedule_model_pulling(
    model_server: EndpointConfig, wait_time_between_pulls: int, agent: Agent
) -> None:
    """安排模型拉取任务。
    
    此函数将模型拉取任务添加到调度器中，
    定期从服务器检查并拉取模型更新。
    
    Args:
        model_server: 模型服务器端点配置
        wait_time_between_pulls: 拉取间隔时间（秒）
        agent: 要更新的代理实例
    """
    # 获取调度器并添加定期任务
    (await jobs.scheduler()).add_job(  # 获取任务调度器并添加任务
        _run_model_pulling_worker,  # 要执行的工作函数
        "interval",                  # 间隔任务类型
        seconds=wait_time_between_pulls,  # 间隔时间（秒）
        args=[model_server, agent],  # 传递给工作函数的参数
        id="pull-model-from-server", # 任务唯一标识符
        replace_existing=True,       # 替换现有同名任务
    )


# =============================================================================
# 代理加载函数
# =============================================================================

async def load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    loop: Optional[AbstractEventLoop] = None,
) -> Agent:
    """从服务器、远程存储或磁盘加载代理。

    Args:
        model_path: 如果模型在磁盘上，则为模型路径
        model_server: 提供模型的潜在服务器配置
        remote_storage: 模型的远程存储URL
        endpoints: 端点配置
        loop: 传递给代理创建的可选异步循环

    Returns:
        实例化的 `Agent` 或 `None`
    """
    # 导入必要的类
    from rasa.core.tracker_store import TrackerStore  # 导入跟踪器存储基类
    from rasa.core.brokers.broker import EventBroker  # 导入事件代理类

    # 初始化组件变量
    tracker_store = None  # 跟踪器存储实例
    lock_store = None  # 锁存储实例
    generator = None  # 自然语言生成器实例
    action_endpoint = None  # 动作端点配置
    http_interpreter = None  # HTTP解释器实例

    # 如果提供了端点配置，创建相应的组件
    if endpoints:  # 如果提供了端点配置
        # 创建事件代理
        broker = await EventBroker.create(endpoints.event_broker, loop=loop)  # 异步创建事件代理
        # 创建跟踪器存储
        tracker_store = TrackerStore.create(  # 创建跟踪器存储
            endpoints.tracker_store, event_broker=broker  # 传入跟踪器存储配置和事件代理
        )
        # 创建锁存储
        lock_store = LockStore.create(endpoints.lock_store)  # 创建锁存储
        # 设置自然语言生成器
        generator = endpoints.nlg  # 设置自然语言生成器
        # 设置动作端点
        action_endpoint = endpoints.action  # 设置动作端点配置
        # 设置模型服务器（优先使用端点配置）
        model_server = endpoints.model if endpoints.model else model_server  # 优先使用端点配置中的模型服务器
        # 如果配置了NLU端点，创建HTTP解释器
        if endpoints.nlu:  # 如果配置了NLU端点
            http_interpreter = RasaNLUHttpInterpreter(endpoints.nlu)  # 创建HTTP NLU解释器

    # 创建代理实例
    agent = Agent(  # 创建Agent实例
        generator=generator,  # 传入自然语言生成器
        tracker_store=tracker_store,  # 传入跟踪器存储
        lock_store=lock_store,  # 传入锁存储
        action_endpoint=action_endpoint,  # 传入动作端点配置
        model_server=model_server,  # 传入模型服务器配置
        remote_storage=remote_storage,  # 传入远程存储配置
        http_interpreter=http_interpreter,  # 传入HTTP解释器
    )

    try:
        # 根据配置加载模型
        if model_server is not None:  # 如果配置了模型服务器
            # 从服务器加载
            return await load_from_server(agent, model_server)  # 从服务器加载模型

        elif remote_storage is not None:  # 如果配置了远程存储
            # 从远程存储加载
            agent.load_model_from_remote_storage(model_path)  # 从远程存储加载模型

        elif model_path is not None and os.path.exists(model_path):  # 如果提供了本地模型路径且文件存在
            # 从本地路径加载
            try:
                agent.load_model(model_path)  # 从本地路径加载模型
            except ModelNotFound:  # 捕获模型未找到异常
                rasa.shared.utils.io.raise_warning(  # 发出警告
                    f"在 {model_path} 处未找到有效模型！"
                )
        else:
            # 没有有效配置
            rasa.shared.utils.io.raise_warning(  # 发出警告
                "没有提供有效配置来加载代理。 "
                "代理已加载但没有模型！"
            )
        return agent  # 返回代理实例

    except Exception as e:  # 捕获所有异常
        # 记录错误并返回代理（即使没有模型）
        logger.error(f"由于 {e} 无法加载模型。", exc_info=True)  # 记录错误日志
        return agent  # 返回代理实例（即使没有模型）


# =============================================================================
# 装饰器函数
# =============================================================================

def agent_must_be_ready(f: Callable[..., Any]) -> Callable[..., Any]:
    """任何用此装饰器装饰的代理方法在代理未就绪时会抛出异常。
    
    此装饰器确保代理在使用前已正确初始化，
    包括设置了处理器和跟踪器存储。
    
    Args:
        f: 要装饰的方法
        
    Returns:
        装饰后的方法
    """

    @functools.wraps(f)  # 保持原函数的元数据
    def decorated(self: Agent, *args: Any, **kwargs: Any) -> Any:  # 定义装饰后的方法
        # 检查代理是否就绪
        if not self.is_ready():  # 如果代理未就绪
            raise AgentNotReady(  # 抛出代理未就绪异常
                "代理在使用前需要准备。您需要设置 "
                "处理器和跟踪器存储。"
            )
        # 调用原始方法
        return f(self, *args, **kwargs)  # 调用原始方法并返回结果

    return decorated  # 返回装饰后的方法


# =============================================================================
# Agent 类定义 - Rasa 对话代理的核心实现
# =============================================================================

class Agent:
    """代理类提供最重要的 Rasa 功能接口。

    这包括训练、处理消息、加载对话模型、
    获取下一个动作和处理通道。
    
    Agent 是 Rasa 对话系统的核心组件，负责：
    - 管理对话状态和跟踪器
    - 处理用户消息和意图识别
    - 预测和执行下一个动作
    - 生成自然语言响应
    - 与外部系统集成
    """

    def __init__(
        self,
        domain: Optional[Domain] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ):
        """初始化一个 `Agent` 实例。"""
        self.domain = domain  # 对话域，定义对话系统的所有组件
        self.processor: Optional[MessageProcessor] = None  # 消息处理器，初始为None

        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)  # 创建自然语言生成器
        self.tracker_store = self._create_tracker_store(tracker_store, self.domain)  # 创建跟踪器存储
        self.lock_store = self._create_lock_store(lock_store)  # 创建锁存储
        self.action_endpoint = action_endpoint  # 动作端点配置
        self.http_interpreter = http_interpreter  # HTTP解释器

        self._set_fingerprint(fingerprint)  # 设置模型指纹
        self.model_server = model_server  # 模型服务器配置
        self.remote_storage = remote_storage  # 远程存储配置

    @classmethod
    def load(
        cls,
        model_path: Union[Text, Path],
        domain: Optional[Domain] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ) -> Agent:
        """构造一个新的代理并加载处理器和模型。"""
        agent = Agent(  # 创建Agent实例
            domain=domain,  # 传入对话域
            generator=generator,  # 传入自然语言生成器
            tracker_store=tracker_store,  # 传入跟踪器存储
            lock_store=lock_store,  # 传入锁存储
            action_endpoint=action_endpoint,  # 传入动作端点配置
            fingerprint=fingerprint,  # 传入模型指纹
            model_server=model_server,  # 传入模型服务器配置
            remote_storage=remote_storage,  # 传入远程存储配置
            http_interpreter=http_interpreter,  # 传入HTTP解释器
        )
        agent.load_model(model_path=model_path, fingerprint=fingerprint)  # 加载模型
        return agent  # 返回代理实例

    def load_model(
        self, model_path: Union[Text, Path], fingerprint: Optional[Text] = None
    ) -> None:
        """根据新的模型路径加载代理的模型和处理器。"""
        self.processor = MessageProcessor(  # 创建消息处理器
            model_path=model_path,  # 传入模型路径
            tracker_store=self.tracker_store,  # 传入跟踪器存储
            lock_store=self.lock_store,  # 传入锁存储
            action_endpoint=self.action_endpoint,  # 传入动作端点配置
            generator=self.nlg,  # 传入自然语言生成器
            http_interpreter=self.http_interpreter,  # 传入HTTP解释器
        )
        self.domain = self.processor.domain  # 从处理器获取对话域

        self._set_fingerprint(fingerprint)  # 设置模型指纹

        # update domain on all instances
        self.tracker_store.domain = self.domain  # 更新跟踪器存储的对话域
        if isinstance(self.nlg, TemplatedNaturalLanguageGenerator):  # 如果是模板化自然语言生成器
            self.nlg.responses = self.domain.responses if self.domain else {}  # 更新响应模板

    @property
    def model_id(self) -> Optional[Text]:
        """从处理器的模型元数据中返回模型ID。"""
        return self.processor.model_metadata.model_id if self.processor else None  # 如果处理器存在则返回模型ID，否则返回None

    @property
    def model_name(self) -> Optional[Text]:
        """从处理器的模型路径中返回模型名称。"""
        return self.processor.model_path.name if self.processor else None  # 如果处理器存在则返回模型名称，否则返回None

    def is_ready(self) -> bool:
        """检查所有必要的组件是否已实例化以使用代理。"""
        return self.tracker_store is not None and self.processor is not None  # 检查跟踪器存储和处理器是否都已初始化

    @agent_must_be_ready
    async def parse_message(self, message_data: Text) -> Dict[Text, Any]:
        """处理消息文本和意图载荷输入消息。

        此函数的返回值是解析后的数据。

        Args:
            message_data (Text): 包含接收到的消息，格式为文本或意图载荷。

        Returns:
            解析后的消息。

        Example:
                {\
                    "text": '/greet{"name":"Rasa"}',\
                    "intent": {"name": "greet", "confidence": 1.0},\
                    "intent_ranking": [{"name": "greet", "confidence": 1.0}],\
                    "entities": [{"entity": "name", "start": 6,\
                                  "end": 21, "value": "Rasa"}],\
                }

        """
        message = UserMessage(message_data)  # 创建用户消息对象

        return await self.processor.parse_message(message)  # type: ignore[union-attr]  # 异步解析消息并返回结果

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """处理单个消息。"""
        if not self.is_ready():  # 如果代理未就绪
            logger.info("Ignoring message as there is no agent to handle it.")  # 记录忽略消息的日志
            return None  # 返回None

        # todo : 这确保了针对同一对话的多条消息按顺序处理，防止对同一追踪器进行并发修改时出现的问题。
        async with self.lock_store.lock(message.sender_id):  # 使用锁确保同一发送者的消息串行处理
            return await self.processor.handle_message(  # type: ignore[union-attr]  # 异步处理消息并返回结果
                message
            )

    @agent_must_be_ready
    async def predict_next_for_sender_id(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """为发送者ID预测下一个动作。"""
        return await self.processor.predict_next_for_sender_id(  # type: ignore[union-attr] # noqa:E501  # 异步预测下一个动作
            sender_id
        )

    @agent_must_be_ready
    def predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART,
    ) -> Optional[Dict[Text, Any]]:
        """使用跟踪器预测下一个动作。"""
        return self.processor.predict_next_with_tracker(  # type: ignore[union-attr]  # 使用跟踪器预测下一个动作
            tracker, verbosity
        )

    @agent_must_be_ready
    async def log_message(self, message: UserMessage) -> DialogueStateTracker:
        """将消息添加到对话中 - 不预测动作。"""
        return await self.processor.log_message(message)  # type: ignore[union-attr]  # 异步记录消息

    @agent_must_be_ready
    async def execute_action(
        self,
        sender_id: Text,
        action: Text,
        output_channel: OutputChannel,
        policy: Optional[Text],
        confidence: Optional[float],
    ) -> Optional[DialogueStateTracker]:
        """执行一个动作。"""
        prediction = PolicyPrediction.for_action_name(  # 为动作名称创建策略预测
            self.domain, action, policy, confidence or 0.0
        )
        return await self.processor.execute_action(  # type: ignore[union-attr]  # 异步执行动作
            sender_id, action, output_channel, self.nlg, prediction
        )

    @agent_must_be_ready
    async def trigger_intent(
        self,
        intent_name: Text,
        entities: List[Dict[Text, Any]],
        output_channel: OutputChannel,
        tracker: DialogueStateTracker,
    ) -> None:
        """触发用户意图，例如由外部事件触发。"""
        await self.processor.trigger_external_user_uttered(  # type: ignore[union-attr]  # 异步触发外部用户话语
            intent_name, entities, tracker, output_channel
        )

    @agent_must_be_ready
    async def handle_text(
        self,
        text_message: Union[Text, Dict[Text, Any]],
        output_channel: Optional[OutputChannel] = None,
        sender_id: Optional[Text] = DEFAULT_SENDER_ID,
    ) -> Optional[List[Dict[Text, Any]]]:
        """处理单个消息。

        如果传递了消息预处理器，消息将首先传递给该函数，
        然后返回值用作对话引擎的输入。

        此函数的返回值取决于 ``output_channel``。如果
        输出通道未设置、设置为 ``None`` 或设置为
        ``CollectingOutputChannel``，此函数将返回机器人想要响应的消息。

        :Example:

            >>> from rasa.core.agent import Agent
            >>> agent = Agent.load("examples/moodbot/models")
            >>> await agent.handle_text("hello")
            [u'how can I help you?']

        """
        if isinstance(text_message, str):  # 如果消息是字符串
            text_message = {"text": text_message}  # 转换为字典格式

        msg = UserMessage(text_message.get("text"), output_channel, sender_id)  # 创建用户消息对象

        return await self.handle_message(msg)  # 异步处理消息并返回结果

    def _set_fingerprint(self, fingerprint: Optional[Text] = None) -> None:
        """设置模型指纹。"""
        if fingerprint:  # 如果提供了指纹
            self.fingerprint = fingerprint  # 使用提供的指纹
        else:
            self.fingerprint = uuid.uuid4().hex  # 生成新的UUID作为指纹

    @staticmethod
    def _create_tracker_store(
        store: Optional[TrackerStore], domain: Domain
    ) -> TrackerStore:
        """创建跟踪器存储。"""
        if store is not None:  # 如果提供了存储
            store.domain = domain  # 设置对话域
            tracker_store = store  # 使用提供的存储
        else:
            tracker_store = InMemoryTrackerStore(domain)  # 创建内存跟踪器存储

        return FailSafeTrackerStore(tracker_store)  # 包装为故障安全存储

    @staticmethod
    def _create_lock_store(store: Optional[LockStore]) -> LockStore:
        """创建锁存储。"""
        if store is not None:  # 如果提供了锁存储
            return store  # 返回提供的锁存储

        return InMemoryLockStore()  # 创建内存锁存储

    def load_model_from_remote_storage(self, model_name: Text) -> None:
        """从远程存储加载代理。"""
        from rasa.nlu.persistor import get_persistor  # 导入持久化器

        persistor = get_persistor(self.remote_storage)  # 获取持久化器

        if persistor is not None:  # 如果持久化器存在
            with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:  # 创建临时目录
                persistor.retrieve(model_name, temporary_directory)  # 从远程存储检索模型
                self.load_model(temporary_directory)  # 加载模型

        else:
            raise RasaException(  # 抛出异常
                f"Persistor not found for remote storage: '{self.remote_storage}'."
            )
