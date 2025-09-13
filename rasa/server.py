# =============================================================================
# Rasa HTTP 服务器模块 - 提供 RESTful API 接口
# =============================================================================
# 此模块实现了 Rasa 的 HTTP 服务器，提供完整的 RESTful API 接口，
# 包括对话管理、模型训练、测试、预测等功能。

# 标准库导入
import asyncio                    # 异步编程支持
import concurrent.futures         # 并发执行支持
import logging                    # 日志记录
import multiprocessing            # 多进程支持
import os                         # 操作系统接口
import traceback                  # 异常跟踪
from collections import defaultdict  # 默认字典
from functools import reduce, wraps  # 函数工具
from inspect import isawaitable   # 检查是否可等待
from pathlib import Path          # 路径处理
from http import HTTPStatus       # HTTP 状态码
from typing import (              # 类型提示
    Any,                          # 任意类型
    Callable,                     # 可调用类型
    DefaultDict,                  # 默认字典类型
    List,                         # 列表类型
    Optional,                     # 可选类型
    Text,                         # 文本类型
    Union,                        # 联合类型
    Dict,                         # 字典类型
    TYPE_CHECKING,                # 类型检查标志
    NoReturn,                     # 无返回值类型
    Coroutine,                    # 协程类型
)

# 第三方库导入
import aiohttp                    # 异步 HTTP 客户端
import jsonschema                 # JSON 模式验证
from sanic import Sanic, response # Sanic Web 框架
from sanic.request import Request # Sanic 请求对象
from sanic.response import HTTPResponse  # Sanic 响应对象
from sanic_cors import CORS       # CORS 支持
from sanic_jwt import Initialize, exceptions  # JWT 认证

# Rasa 内部模块导入
import rasa  # Rasa 主模块
import rasa.core.utils  # Core 工具函数
from rasa.nlu.emulators.emulator import Emulator  # NLU 模拟器基类
import rasa.utils.common  # 通用工具函数
import rasa.shared.utils.common  # 共享通用工具
import rasa.shared.utils.io  # 共享 IO 工具
import rasa.shared.utils.validation  # 共享验证工具
import rasa.shared.nlu.training_data.schemas.data_schema  # NLU 数据模式
import rasa.utils.endpoints  # 端点工具
import rasa.utils.io  # IO 工具
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,  # YAML 故事写入器
)
from rasa.shared.importers.importer import TrainingDataImporter  # 训练数据导入器
from rasa.shared.nlu.training_data.formats import RasaYAMLReader  # Rasa YAML 读取器
from rasa.core.constants import DEFAULT_RESPONSE_TIMEOUT  # 默认响应超时
from rasa.constants import MINIMUM_COMPATIBLE_VERSION  # 最小兼容版本
from rasa.shared.constants import (  # 共享常量
    DOCS_URL_TRAINING_DATA,  # 训练数据文档URL
    DOCS_BASE_URL,           # 文档基础URL
    DEFAULT_SENDER_ID,       # 默认发送者ID
    DEFAULT_MODELS_PATH,     # 默认模型路径
    TEST_STORIES_FILE_PREFIX,  # 测试故事文件前缀
)
from rasa.shared.core.domain import InvalidDomain, Domain  # 域相关类
from rasa.core.agent import Agent  # 代理类
from rasa.core.channels.channel import (  # 通道相关类
    CollectingOutputChannel,  # 收集输出通道
    OutputChannel,            # 输出通道基类
    UserMessage,              # 用户消息类
)
import rasa.shared.core.events  # 共享事件模块
from rasa.shared.core.events import Event  # 事件基类
from rasa.core.test import test  # 测试函数
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name  # 临时目录工具
from rasa.shared.core.trackers import (  # 跟踪器相关类
    DialogueStateTracker,  # 对话状态跟踪器
    EventVerbosity,        # 事件详细程度
)
from rasa.core.utils import AvailableEndpoints  # 可用端点
from rasa.nlu.emulators.no_emulator import NoEmulator  # 无模拟器
import rasa.nlu.test  # NLU 测试模块
from rasa.nlu.test import CVEvaluationResult  # 交叉验证评估结果
from rasa.shared.utils.schemas.events import EVENTS_SCHEMA  # 事件模式
from rasa.utils.endpoints import EndpointConfig  # 端点配置

# 类型检查导入（避免循环导入）
if TYPE_CHECKING:
    from ssl import SSLContext  # SSL 上下文
    from rasa.core.processor import MessageProcessor  # 消息处理器
    from mypy_extensions import Arg, VarArg, KwArg  # MyPy 扩展

    # Sanic 响应类型定义
    SanicResponse = Union[
        response.HTTPResponse, Coroutine[Any, Any, response.HTTPResponse]
    ]
    # Sanic 视图类型定义
    SanicView = Callable[
        [Arg(Request, "request"), VarArg(), KwArg()],
        Coroutine[Any, Any, SanicResponse],
    ]


# 日志记录器
logger = logging.getLogger(__name__)

# =============================================================================
# 常量定义
# =============================================================================

# 内容类型常量
JSON_CONTENT_TYPE = "application/json"  # JSON 内容类型
YAML_CONTENT_TYPE = "application/x-yaml"  # YAML 内容类型

# 查询参数键常量
OUTPUT_CHANNEL_QUERY_KEY = "output_channel"  # 输出通道查询键
USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL = "latest"  # 使用最新输入通道作为输出通道
EXECUTE_SIDE_EFFECTS_QUERY_KEY = "execute_side_effects"  # 执行副作用查询键


# =============================================================================
# 异常类定义
# =============================================================================

class ErrorResponse(Exception):
    """用于处理失败 API 请求的通用异常。
    
    此类提供了统一的错误响应格式，包含错误信息、状态码、
    帮助链接等详细信息。
    """

    def __init__(
        self,
        status: Union[int, HTTPStatus],  # HTTP 状态码
        reason: Text,                    # 错误原因
        message: Text,                   # 错误消息
        details: Any = None,             # 错误详情
        help_url: Optional[Text] = None, # 帮助链接
    ) -> None:
        """创建错误响应。

        Args:
            status: 要返回的 HTTP 状态码
            reason: 错误的简短摘要
            message: 错误的详细说明
            details: 描述错误的附加详情。必须可序列化
            help_url: 用户可以获取进一步帮助的URL（如文档）
        """
        # 构建错误信息字典
        self.error_info = {
            "version": rasa.__version__,  # Rasa 版本
            "status": "failure",          # 状态
            "message": message,           # 错误消息
            "reason": reason,             # 错误原因
            "details": details or {},     # 错误详情
            "help": help_url,             # 帮助链接
            "code": status,               # 状态码
        }
        self.status = status  # 设置状态码
        logger.error(message)  # 记录错误日志
        super(ErrorResponse, self).__init__()  # 调用父类构造函数


# =============================================================================
# 工具函数
# =============================================================================

def _docs(sub_url: Text) -> Text:
    """创建指向文档子部分的URL。
    
    Args:
        sub_url: 文档子路径
        
    Returns:
        完整的文档URL
    """
    return DOCS_BASE_URL + sub_url  # 拼接文档基础URL和子路径


def ensure_loaded_agent(
    app: Sanic, require_core_is_ready: bool = False
) -> Callable[[Callable], Callable[..., Any]]:
    """包装请求处理器，确保有已加载且可用的代理。

    如果 `require_core_is_ready` 为 `True`，
    则要求代理具有已加载的 Core 模型。
    
    Args:
        app: Sanic 应用实例
        require_core_is_ready: 是否需要 Core 模型就绪
        
    Returns:
        装饰器函数
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            # 检查代理是否已加载且就绪
            # noinspection PyUnresolvedReferences
            if not app.ctx.agent or not app.ctx.agent.is_ready():
                raise ErrorResponse(
                    HTTPStatus.CONFLICT,
                    "Conflict",
                    "No agent loaded. To continue processing, a "
                    "model of a trained agent needs to be loaded.",
                    help_url=_docs("/user-guide/configuring-http-api/"),
                )  # 抛出冲突错误

            return f(*args, **kwargs)  # 调用原始函数

        return decorated  # 返回装饰后的函数

    return decorator  # 返回装饰器


def ensure_conversation_exists() -> Callable[["SanicView"], "SanicView"]:
    """Wraps a request handler ensuring the conversation exists."""

    def decorator(f: "SanicView") -> "SanicView":
        @wraps(f)
        async def decorated(
            request: Request, *args: Any, **kwargs: Any
        ) -> "SanicResponse":
            conversation_id = kwargs["conversation_id"]
            if await request.app.ctx.agent.tracker_store.exists(conversation_id):
                return await f(request, *args, **kwargs)
            else:
                raise ErrorResponse(
                    HTTPStatus.NOT_FOUND, "Not found", "Conversation ID not found."
                )

        return decorated

    return decorator


def requires_auth(
    app: Sanic, token: Optional[Text] = None
) -> Callable[["SanicView"], "SanicView"]:
    """Wraps a request handler with token authentication."""

    def decorator(f: "SanicView") -> "SanicView":
        def conversation_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.shared.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("conversation_id")
                if "conversation_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["conversation_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        async def sufficient_scope(
            request: Request, *args: Any, **kwargs: Any
        ) -> Optional[bool]:
            # This is a coroutine since `sanic-jwt==1.6`
            jwt_data = await rasa.utils.common.call_potential_coroutine(
                request.app.ctx.auth.extract_payload(request)
            )

            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                conversation_id = conversation_id_from_args(args, kwargs)
                return conversation_id is not None and username == conversation_id
            else:
                return False

        @wraps(f)
        async def decorated(
            request: Request, *args: Any, **kwargs: Any
        ) -> response.HTTPResponse:

            provided = request.args.get("token", None)

            # noinspection PyProtectedMember
            if token is not None and provided == token:
                result = f(request, *args, **kwargs)
                return await result if isawaitable(result) else result
            elif app.config.get(
                "USE_JWT"
            ) and await rasa.utils.common.call_potential_coroutine(
                # This is a coroutine since `sanic-jwt==1.6`
                request.app.ctx.auth.is_authenticated(request)
            ):
                if await sufficient_scope(request, *args, **kwargs):
                    result = f(request, *args, **kwargs)
                    return await result if isawaitable(result) else result
                raise ErrorResponse(
                    HTTPStatus.FORBIDDEN,
                    "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs(
                        "/user-guide/configuring-http-api/#security-considerations"
                    ),
                )
            elif token is None and app.config.get("USE_JWT") is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                return await result if isawaitable(result) else result
            raise ErrorResponse(
                HTTPStatus.UNAUTHORIZED,
                "NotAuthenticated",
                "User is not authenticated.",
                help_url=_docs(
                    "/user-guide/configuring-http-api/#security-considerations"
                ),
            )

        return decorated

    return decorator


def event_verbosity_parameter(
    request: Request, default_verbosity: EventVerbosity
) -> EventVerbosity:
    """如果存在请求参数，则使用请求参数创建 `EventVerbosity` 对象。"""
    event_verbosity_str = request.args.get(
        "include_events", default_verbosity.name
    ).upper()
    try:
        return EventVerbosity[event_verbosity_str]
    except KeyError:
        enum_values = ", ".join([e.name for e in EventVerbosity])
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "Invalid parameter value for 'include_events'. "
            "Should be one of {}".format(enum_values),
            {"parameter": "include_events", "in": "query"},
        )


async def get_test_stories(
    processor: "MessageProcessor",
    conversation_id: Text,
    until_time: Optional[float],
    fetch_all_sessions: bool = False,
) -> Text:
    """从 `processor` 检索 `conversation_id` 的所有对话会话的测试故事。

    Args:
        processor: `MessageProcessor` 的实例
        conversation_id: 要获取故事的对话ID
        until_time: 包含事件的时间戳
        fetch_all_sessions: 是否获取所有对话会话的故事。
            如果为 `False`，则只检索最后一个对话会话

    Returns:
        `conversation_id` 的测试格式故事
    """
    if fetch_all_sessions:
        trackers = await processor.get_trackers_for_all_conversation_sessions(
            conversation_id
        )
    else:
        trackers = [await processor.get_tracker(conversation_id)]

    if until_time is not None:
        trackers = [tracker.travel_back_in_time(until_time) for tracker in trackers]
        # keep only non-empty trackers
        trackers = [tracker for tracker in trackers if len(tracker.events)]

    logger.debug(
        f"Fetched trackers for {len(trackers)} conversation sessions "
        f"for conversation ID {conversation_id}."
    )

    story_steps = []

    more_than_one_story = len(trackers) > 1

    for i, tracker in enumerate(trackers, 1):
        tracker.sender_id = conversation_id

        if more_than_one_story:
            tracker.sender_id += f", story {i}"

        story_steps += tracker.as_story().story_steps

    return YAMLStoryWriter().dumps(story_steps, is_test_story=True)


async def update_conversation_with_events(
    conversation_id: Text,
    processor: "MessageProcessor",
    domain: Domain,
    events: List[Event],
) -> DialogueStateTracker:
    """获取或创建 `conversation_id` 的跟踪器并将 `events` 追加到其中。

    Args:
        conversation_id: 要更新跟踪器的对话ID
        processor: `MessageProcessor` 的实例
        domain: 与当前 `Agent` 关联的域
        events: 要追加到跟踪器的事件

    Returns:
        具有更新事件的 `conversation_id` 跟踪器
    """
    if rasa.shared.core.events.do_events_begin_with_session_start(events):
        tracker = await processor.get_tracker(conversation_id)
    else:
        tracker = await processor.fetch_tracker_with_initial_session(conversation_id)

    for event in events:
        tracker.update(event, domain)

    return tracker


def validate_request_body(request: Request, error_message: Text) -> None:
    """检查 `request` 是否有请求体。"""
    if not request.body:
        raise ErrorResponse(HTTPStatus.BAD_REQUEST, "BadRequest", error_message)


def validate_events_in_request_body(request: Request) -> None:
    """验证请求体中的事件格式。"""
    if not isinstance(request.json, list):
        events = [request.json]
    else:
        events = request.json

    try:
        jsonschema.validate(events, EVENTS_SCHEMA)
    except jsonschema.ValidationError as error:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"Failed to validate the events format. "
            f"For more information about the format visit the docs. Error: {error}",
            help_url=_docs("/pages/http-api"),
        ) from error


async def authenticate(_: Request) -> NoReturn:
    """认证失败的回调函数。"""
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Rasa will just make "
        "sure that the token is valid, but not issue new tokens."
    )


def create_ssl_context(
    ssl_certificate: Optional[Text],
    ssl_keyfile: Optional[Text],
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
) -> Optional["SSLContext"]:
    """如果传递了适当的证书，则创建SSL上下文。

    Args:
        ssl_certificate: SSL客户端证书的路径
        ssl_keyfile: SSL密钥文件的路径
        ssl_ca_file: 用于验证的SSL CA文件路径（可选）
        ssl_password: SSL私钥密码（可选）

    Returns:
        如果可以加载有效的证书链则返回SSL上下文，否则返回 `None`

    """
    if ssl_certificate:
        import ssl

        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.CLIENT_AUTH, cafile=ssl_ca_file
        )
        ssl_context.load_cert_chain(
            ssl_certificate, keyfile=ssl_keyfile, password=ssl_password
        )
        return ssl_context
    else:
        return None


def _create_emulator(mode: Optional[Text]) -> Emulator:
    """为指定模式创建模拟器。

    如果没有指定模拟器，我们将使用Rasa NLU格式。
    """
    if mode is None:
        return NoEmulator()
    elif mode.lower() == "wit":
        from rasa.nlu.emulators.wit import WitEmulator

        return WitEmulator()
    elif mode.lower() == "luis":
        from rasa.nlu.emulators.luis import LUISEmulator

        return LUISEmulator()
    elif mode.lower() == "dialogflow":
        from rasa.nlu.emulators.dialogflow import DialogflowEmulator

        return DialogflowEmulator()
    else:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "Invalid parameter value for 'emulation_mode'. "
            "Should be one of 'WIT', 'LUIS', 'DIALOGFLOW'.",
            {"parameter": "emulation_mode", "in": "query"},
        )


async def _load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
) -> Agent:
    """加载代理。
    
    Args:
        model_path: 模型路径
        model_server: 模型服务器配置
        remote_storage: 远程存储
        endpoints: 可用端点
        
    Returns:
        加载的代理
    """
    try:
        loaded_agent = await rasa.core.agent.load_agent(
            model_path=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            endpoints=endpoints,
        )
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise ErrorResponse(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "LoadingError",
            f"An unexpected error occurred. Error: {e}",
        )

    if not loaded_agent.is_ready():
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"Agent with name '{model_path}' could not be loaded.",
            {"parameter": "model", "in": "query"},
        )

    return loaded_agent


def configure_cors(
    app: Sanic, cors_origins: Union[Text, List[Text], None] = ""
) -> None:
    """为给定的应用配置CORS源。"""
    # Workaround so that socketio works with requests from other origins.
    # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
    app.config.CORS_AUTOMATIC_OPTIONS = True
    app.config.CORS_SUPPORTS_CREDENTIALS = True
    app.config.CORS_EXPOSE_HEADERS = "filename"

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )


def add_root_route(app: Sanic) -> None:
    """添加 '/' 路由以返回问候语。"""

    @app.get("/")
    async def hello(request: Request) -> HTTPResponse:
        """检查服务器是否正在运行并响应版本信息。"""
        return response.text("Hello from Rasa: " + rasa.__version__)


def async_if_callback_url(f: Callable[..., Coroutine]) -> Callable:
    """启用异步请求处理的装饰器。

    如果传入的HTTP请求指定了 `callback_url` 查询参数，请求
    将立即返回204，而实际的请求响应将
    发送到 `callback_url`。如果发生错误，错误负载也将
    发送到 `callback_url`。

    Args:
        f: 应该被装饰的请求处理函数

    Returns:
        装饰后的函数
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        callback_url = request.args.get("callback_url")
        # Only process request asynchronously if the user specified a `callback_url`
        # query parameter.
        if not callback_url:
            return await f(request, *args, **kwargs)

        async def wrapped() -> None:
            try:
                result: HTTPResponse = await f(request, *args, **kwargs)
                payload: Dict[Text, Any] = dict(
                    data=result.body, headers={"Content-Type": result.content_type}
                )
                logger.debug(
                    "Asynchronous processing of request was successful. "
                    "Sending result to callback URL."
                )

            except Exception as e:
                if not isinstance(e, ErrorResponse):
                    logger.error(e)
                    e = ErrorResponse(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "UnexpectedError",
                        f"An unexpected error occurred. Error: {e}",
                    )
                # If an error happens, we send the error payload to the `callback_url`
                payload = dict(json=e.error_info)
                logger.error(
                    "Error happened when processing request asynchronously. "
                    "Sending error to callback URL."
                )
            async with aiohttp.ClientSession() as session:
                await session.post(callback_url, raise_for_status=True, **payload)

        # Run the request in the background on the event loop
        request.app.add_task(wrapped())

        # The incoming request will return immediately with a 204
        return response.empty()

    return decorated_function


def run_in_thread(f: Callable[..., Coroutine]) -> Callable:
    """在单独线程上运行请求的装饰器。

    某些请求（例如训练或交叉验证）是计算密集型请求。
    这意味着它们将阻塞事件循环，从而阻塞其他
    请求的处理。此装饰器可用于在单独线程上处理这些请求
    以避免阻塞传入请求的处理。

    Args:
        f: 应该被装饰的请求处理函数

    Returns:
        装饰后的函数
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        # Use a sync wrapper for our `async` function as `run_in_executor` only supports
        # sync functions
        def run() -> HTTPResponse:
            return asyncio.run(f(request, *args, **kwargs))

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await request.app.loop.run_in_executor(pool, run)

    return decorated_function


def inject_temp_dir(f: Callable[..., Coroutine]) -> Callable:
    """在请求前注入临时目录并在之后清理的装饰器。

    Args:
        f: 应该被装饰的请求处理函数

    Returns:
        装饰后的函数
    """

    @wraps(f)
    async def decorated_function(*args: Any, **kwargs: Any) -> HTTPResponse:
        with TempDirectoryPath(get_temp_dir_name()) as directory:
            # Decorated request handles need to have a parameter `temporary_directory`
            return await f(*args, temporary_directory=Path(directory), **kwargs)

    return decorated_function


# =============================================================================
# 主应用创建函数 - 创建和配置 Rasa HTTP 服务器
# =============================================================================

def create_app(
    agent: Optional["Agent"] = None,                    # 可选的代理实例
    cors_origins: Union[Text, List[Text], None] = "*",  # CORS 源配置，默认为允许所有源
    auth_token: Optional[Text] = None,                  # 认证令牌
    response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,   # 响应超时时间
    jwt_secret: Optional[Text] = None,                  # JWT 密钥
    jwt_private_key: Optional[Text] = None,             # JWT 私钥
    jwt_method: Text = "HS256",                         # JWT 算法方法
    endpoints: Optional[AvailableEndpoints] = None,     # 可用端点配置
) -> Sanic:
    """创建和配置 Rasa HTTP 服务器应用。
    
    此函数是 Rasa 服务器的核心创建函数，负责初始化 Sanic 应用、
    配置认证、CORS、错误处理、注册所有 API 端点等。
    
    Args:
        agent: 可选的代理实例，用于处理对话请求
        cors_origins: CORS 源配置，控制跨域访问
        auth_token: 简单的认证令牌
        response_timeout: HTTP 响应超时时间（秒）
        jwt_secret: JWT 对称密钥
        jwt_private_key: JWT 非对称私钥
        jwt_method: JWT 签名算法（HS256、RS256等）
        endpoints: 可用端点配置
        
    Returns:
        配置完成的 Sanic 应用实例
    """
    # 创建 Sanic 应用实例
    app = Sanic("rasa_server")
    
    # 设置响应超时配置
    app.config.RESPONSE_TIMEOUT = response_timeout
    
    # 配置 CORS（跨域资源共享）
    configure_cors(app, cors_origins)

    # =============================================================================
    # JWT 认证配置
    # =============================================================================
    # 设置 Sanic-JWT 扩展
    if jwt_secret and jwt_method:
        # `sanic-jwt` 在调用 `Initialize` 时需要有一个可用的事件循环。
        # 如果没有，服务器启动将失败并显示错误：
        # `There is no current event loop in thread 'MainThread'`
        try:
            # 尝试获取当前运行的事件循环
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行的事件循环，创建一个新的并设置为当前循环
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        # 由于我们只想检查签名，我们实际上不关心
        # JWT 方法，并将传递的密钥设置为对称或非对称密钥。
        # jwt 库将根据方法选择正确的密钥
        app.config["USE_JWT"] = True  # 启用 JWT 认证
        
        # 初始化 JWT 扩展
        Initialize(
            app,                        # Sanic 应用实例
            secret=jwt_secret,          # JWT 密钥
            private_key=jwt_private_key, # JWT 私钥
            authenticate=authenticate,   # 认证回调函数
            algorithm=jwt_method,        # JWT 算法
            user_id="username",          # 用户ID字段名
        )

    # =============================================================================
    # 应用上下文初始化
    # =============================================================================
    # 将代理实例存储到应用上下文中
    app.ctx.agent = agent
    
    # 初始化用于跟踪活跃训练进程数量的共享对象
    # 使用无符号整数类型，初始值为 0
    app.ctx.active_training_processes = multiprocessing.Value("I", 0)

    # =============================================================================
    # 错误处理配置
    # =============================================================================
    # 注册全局错误处理器，处理 ErrorResponse 异常
    @app.exception(ErrorResponse)
    async def handle_error_response(
        request: Request, exception: ErrorResponse
    ) -> HTTPResponse:
        # 返回 JSON 格式的错误信息，包含状态码
        return response.json(exception.error_info, status=exception.status)

    # =============================================================================
    # 路由注册
    # =============================================================================
    # 添加根路由（健康检查）
    add_root_route(app)

    # =============================================================================
    # 系统信息端点
    # =============================================================================
    
    # 版本信息端点 - 获取 Rasa 版本信息
    @app.get("/version")
    async def version(request: Request) -> HTTPResponse:
        """响应已安装Rasa的版本号。
        
        此端点返回当前安装的 Rasa 版本号和最小兼容版本号，
        用于客户端检查版本兼容性。
        """
        return response.json(
            {
                "version": rasa.__version__,                    # 当前 Rasa 版本
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,  # 最小兼容版本
            }
        )

    # 服务器状态端点 - 获取服务器和模型状态信息
    @app.get("/status")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    async def status(request: Request) -> HTTPResponse:
        """响应模型名称和该模型的指纹。
        
        此端点返回当前加载的模型信息、模型ID和活跃训练任务数量，
        用于监控服务器状态和模型状态。
        """
        return response.json(
            {
                "model_file": app.ctx.agent.processor.model_filename,  # 模型文件名
                "model_id": app.ctx.agent.model_id,                     # 模型ID
                "num_active_training_jobs": app.ctx.active_training_processes.value,  # 活跃训练任务数
            }
        )

    # =============================================================================
    # 对话管理端点
    # =============================================================================
    
    # 获取对话跟踪器端点 - 获取指定对话的跟踪器状态
    @app.get("/conversations/<conversation_id:path>/tracker")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    async def retrieve_tracker(request: Request, conversation_id: Text) -> HTTPResponse:
        """获取对话跟踪器的转储，包括其事件。
        
        此端点返回指定对话的完整跟踪器状态，包括所有事件、
        槽位值、活跃循环等信息。支持时间过滤和详细程度控制。
        """
        # 获取事件详细程度参数，默认为重启后
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        # 获取时间过滤参数
        until_time = rasa.utils.endpoints.float_arg(request, "until")

        # 获取完整的对话跟踪器（包含初始会话）
        tracker = await app.ctx.agent.processor.fetch_full_tracker_with_initial_session(
            conversation_id,
            output_channel=CollectingOutputChannel(),  # 使用收集输出通道
        )

        try:
            # 如果指定了时间过滤，则回退到指定时间
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            # 获取当前状态并返回
            state = tracker.current_state(verbosity)
            return response.json(state)
        except Exception as e:
            # 记录调试信息
            logger.debug(traceback.format_exc())
            # 抛出内部服务器错误
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    # 追加事件端点 - 向对话状态追加事件列表
    @app.post("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    async def append_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """将事件列表追加到对话状态。
        
        此端点允许向指定对话的跟踪器追加新的事件，
        支持执行副作用和保存跟踪器状态。
        """
        # 验证请求体中的事件格式
        validate_events_in_request_body(request)

        # 获取事件详细程度参数
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            # 使用锁确保对话状态的原子性操作
            async with app.ctx.agent.lock_store.lock(conversation_id):
                processor = app.ctx.agent.processor
                # 从请求体中提取事件
                events = _get_events_from_request_body(request)

                # 更新对话状态并追加事件
                tracker = await update_conversation_with_events(
                    conversation_id, processor, app.ctx.agent.domain, events
                )

                # 获取输出通道
                output_channel = _get_output_channel(request, tracker)

                # 如果请求指定执行副作用，则执行
                if rasa.utils.endpoints.bool_arg(
                    request, EXECUTE_SIDE_EFFECTS_QUERY_KEY, False
                ):
                    await processor.execute_side_effects(
                        events, tracker, output_channel
                    )
                # 保存更新后的跟踪器状态
                await app.ctx.agent.tracker_store.save(tracker)
            # 返回更新后的跟踪器状态
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            # 记录调试信息
            logger.debug(traceback.format_exc())
            # 抛出内部服务器错误
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    # =============================================================================
    # 辅助函数
    # =============================================================================
    
    def _get_events_from_request_body(request: Request) -> List[Event]:
        """从请求体中提取事件列表。
        
        此函数负责解析请求体中的事件数据，支持单个事件或事件列表，
        并验证事件的有效性。
        """
        # 获取请求体中的JSON数据
        events = request.json

        # 如果事件不是列表，则转换为列表
        if not isinstance(events, list):
            events = [events]

        # 将每个事件字典转换为Event对象
        events = [Event.from_parameters(event) for event in events]
        # 过滤掉无效的事件（None值）
        events = [event for event in events if event]

        # 如果没有有效的事件，抛出错误
        if not events:
            rasa.shared.utils.io.raise_warning(
                f"Append event called, but could not extract a valid event. "
                f"Request JSON: {request.json}"
            )
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Couldn't extract a proper event from the request body.",
                {"parameter": "", "in": "body"},
            )

        return events

    # 替换事件端点 - 使用事件列表完全替换对话跟踪器状态
    @app.put("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    async def replace_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """使用事件列表将对话跟踪器设置为状态。
        
        此端点允许完全替换指定对话的跟踪器状态，
        而不是追加事件。会覆盖现有的跟踪器。
        """
        # 验证请求体中的事件格式
        validate_events_in_request_body(request)

        # 获取事件详细程度参数
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            # 使用锁确保对话状态的原子性操作
            async with app.ctx.agent.lock_store.lock(conversation_id):
                # 从请求体中的事件列表创建新的跟踪器
                tracker = DialogueStateTracker.from_dict(
                    conversation_id, request.json, app.ctx.agent.domain.slots
                )

                # 保存新的跟踪器状态（会覆盖具有相同ID的现有跟踪器！）
                await app.ctx.agent.tracker_store.save(tracker)

            # 返回新跟踪器的当前状态
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            # 记录调试信息
            logger.debug(traceback.format_exc())
            # 抛出内部服务器错误
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    # 获取对话故事端点 - 获取对话的端到端故事
    @app.get("/conversations/<conversation_id:path>/story")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    @ensure_conversation_exists()        # 确保对话存在
    async def retrieve_story(request: Request, conversation_id: Text) -> HTTPResponse:
        """获取与此对话对应的端到端故事。
        
        此端点返回指定对话的端到端故事格式，
        支持时间过滤和会话选择。
        """
        # 获取时间过滤参数
        until_time = rasa.utils.endpoints.float_arg(request, "until")
        # 获取是否获取所有会话的参数
        fetch_all_sessions = rasa.utils.endpoints.bool_arg(
            request, "all_sessions", default=False
        )

        try:
            # 获取测试故事
            stories = await get_test_stories(
                app.ctx.agent.processor,
                conversation_id,
                until_time,
                fetch_all_sessions=fetch_all_sessions,
            )
            # 返回文本格式的故事
            return response.text(stories)
        except Exception as e:
            # 记录调试信息
            logger.debug(traceback.format_exc())
            # 抛出内部服务器错误
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def execute_action(request: Request, conversation_id: Text) -> HTTPResponse:
        rasa.shared.utils.io.raise_warning(
            'The "POST /conversations/<conversation_id>/execute"'
            " endpoint is deprecated. Inserting actions to the tracker externally"
            " should be avoided. Actions should be predicted by the policies only.",
            category=FutureWarning,
        )
        request_params = request.json

        action_to_execute = request_params.get("name", None)

        if not action_to_execute:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the action not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.ctx.agent.processor.fetch_tracker_and_update_session(
                        conversation_id
                    )
                )

                output_channel = _get_output_channel(request, tracker)
                await app.ctx.agent.execute_action(
                    conversation_id,
                    action_to_execute,
                    output_channel,
                    policy,
                    confidence,
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id:path>/trigger_intent")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def trigger_intent(request: Request, conversation_id: Text) -> HTTPResponse:
        request_params = request.json

        intent_to_trigger = request_params.get("name")
        entities = request_params.get("entities", [])

        if not intent_to_trigger:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the intent not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.ctx.agent.processor.fetch_tracker_and_update_session(
                        conversation_id
                    )
                )
                output_channel = _get_output_channel(request, tracker)
                if intent_to_trigger not in app.ctx.agent.domain.intents:
                    raise ErrorResponse(
                        HTTPStatus.NOT_FOUND,
                        "NotFound",
                        f"The intent {trigger_intent} does not exist in the domain.",
                    )
                await app.ctx.agent.trigger_intent(
                    intent_name=intent_to_trigger,
                    entities=entities,
                    output_channel=output_channel,
                    tracker=tracker,
                )
        except ErrorResponse:
            raise
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    # 预测下一个动作端点 - 预测对话的下一个动作
    @app.post("/conversations/<conversation_id:path>/predict")
    @requires_auth(app, auth_token)      # 需要认证
    @ensure_loaded_agent(app)            # 确保代理已加载
    @ensure_conversation_exists()        # 确保对话存在
    async def predict(request: Request, conversation_id: Text) -> HTTPResponse:
        """预测对话的下一个动作。
        
        此端点基于当前对话状态预测下一个应该执行的动作，
        返回动作列表和对应的置信度分数。
        """
        try:
            # 以JSON格式获取适当的机器人响应
            responses = await app.ctx.agent.predict_next_for_sender_id(conversation_id)
            # 按分数降序和动作名称排序响应
            responses["scores"] = sorted(
                responses["scores"], key=lambda k: (-k["score"], k["action"])
            )
            return response.json(responses)
        except Exception as e:
            # 记录调试信息
            logger.debug(traceback.format_exc())
            # 抛出内部服务器错误
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text) -> HTTPResponse:
        validate_request_body(
            request,
            "No message defined in request body. Add a message to the request body in "
            "order to add it to the tracker.",
        )

        request_params = request.json

        message = request_params.get("text")
        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        # TODO: implement for agent / bot
        if sender != "user":
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Currently, only user messages can be passed to this endpoint. "
                "Messages of sender '{}' cannot be handled.".format(sender),
                {"parameter": "sender", "in": "body"},
            )

        # TODO: 入口用户发送消息  User -> Send message -> Input Channel -> Rasa Server -> Agent
        user_message = UserMessage(message, None, conversation_id, parse_data)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                # cf. processor.handle_message (ignoring prediction loop run)
                tracker = await app.ctx.agent.processor.log_message(
                    user_message, should_save_tracker=False
                )
                tracker = await app.ctx.agent.processor.run_action_extract_slots(
                    user_message.output_channel, tracker
                )
                await app.ctx.agent.processor.save_tracker(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def train(request: Request, temporary_directory: Path) -> HTTPResponse:
        validate_request_body(
            request,
            "You must provide training data in the request body in order to "
            "train your model.",
        )

        training_payload = _training_payload_from_yaml(request, temporary_directory)

        try:
            with app.ctx.active_training_processes.get_lock():
                app.ctx.active_training_processes.value += 1

            from rasa.model_training import train

            # pass `None` to run in default executor
            training_result = train(**training_payload)

            if training_result.model:
                filename = os.path.basename(training_result.model)

                return await response.file(
                    training_result.model,
                    filename=filename,
                    headers={"filename": filename},
                )
            else:
                raise ErrorResponse(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "TrainingError",
                    "Ran training, but it finished without a trained model.",
                )
        except ErrorResponse as e:
            raise e
        except InvalidDomain as e:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "InvalidDomainError",
                f"Provided domain file is invalid. Error: {e}",
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TrainingError",
                f"An unexpected error occurred during training. Error: {e}",
            )
        finally:
            with app.ctx.active_training_processes.get_lock():
                app.ctx.active_training_processes.value -= 1

    @app.post("/model/test/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    @inject_temp_dir
    async def evaluate_stories(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate stories against the currently loaded model."""
        validate_request_body(
            request,
            "You must provide some stories in the request body in order to "
            "evaluate your model.",
        )

        test_data = _test_data_file_from_payload(request, temporary_directory)

        e2e = rasa.utils.endpoints.bool_arg(request, "e2e", default=False)

        try:
            evaluation = await test(
                test_data, app.ctx.agent, e2e=e2e, disable_plotting=True
            )
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    @app.post("/model/test/intents")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def evaluate_intents(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate intents against a Rasa model."""
        validate_request_body(
            request,
            "You must provide some nlu data in the request body in order to "
            "evaluate your model.",
        )

        cross_validation_folds = request.args.get("cross_validation_folds")
        is_yaml_payload = request.headers.get("Content-type") == YAML_CONTENT_TYPE
        test_coroutine = None

        if is_yaml_payload:
            payload = _training_payload_from_yaml(request, temporary_directory)
            config_file = payload.get("config")
            test_data = payload.get("training_files")

            if cross_validation_folds:
                test_coroutine = _cross_validate(
                    test_data, config_file, int(cross_validation_folds)
                )
        else:
            payload = _nlu_training_payload_from_json(request, temporary_directory)
            test_data = payload.get("training_files")

            if cross_validation_folds:
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "TestingError",
                    "Cross-validation is only supported for YAML data.",
                )

        if not cross_validation_folds:
            test_coroutine = _evaluate_model_using_test_set(
                request.args.get("model"), test_data
            )

        try:
            if test_coroutine is not None:
                evaluation = await test_coroutine
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    async def _evaluate_model_using_test_set(
        model_path: Text, test_data_file: Text
    ) -> Dict:
        logger.info("Starting model evaluation using test set.")

        eval_agent = app.ctx.agent

        if model_path:
            model_server = app.ctx.agent.model_server
            if model_server is not None:
                model_server = model_server.copy()
                model_server.url = model_path
                # Set wait time between pulls to `0` so that the agent does not schedule
                # a job to pull the model from the server
                model_server.kwargs["wait_time_between_pulls"] = 0
            eval_agent = await _load_agent(
                model_path=model_path,
                model_server=model_server,
                remote_storage=app.ctx.agent.remote_storage,
            )

        data_path = os.path.abspath(test_data_file)

        if not eval_agent.is_ready():
            raise ErrorResponse(
                HTTPStatus.CONFLICT, "Conflict", "Loaded model file not found."
            )

        return await rasa.nlu.test.run_evaluation(
            data_path, eval_agent.processor, disable_plotting=True, report_as_dict=True
        )

    async def _cross_validate(data_file: Text, config_file: Text, folds: int) -> Dict:
        logger.info(f"Starting cross-validation with {folds} folds.")
        importer = TrainingDataImporter.load_from_dict(
            config=None, config_path=config_file, training_data_paths=[data_file]
        )
        config = importer.get_config()
        nlu_data = importer.get_nlu_data()

        evaluations = await rasa.nlu.test.cross_validate(
            data=nlu_data,
            n_folds=folds,
            nlu_config=config,
            disable_plotting=True,
            errors=True,
            report_as_dict=True,
        )
        evaluation_results = _get_evaluation_results(*evaluations)

        return evaluation_results

    def _get_evaluation_results(
        intent_report: CVEvaluationResult,
        entity_report: CVEvaluationResult,
        response_selector_report: CVEvaluationResult,
    ) -> Dict[Text, Any]:
        eval_name_mapping = {
            "intent_evaluation": intent_report,
            "entity_evaluation": entity_report,
            "response_selection_evaluation": response_selector_report,
        }

        result: DefaultDict[Text, Any] = defaultdict(dict)
        for evaluation_name, evaluation in eval_name_mapping.items():
            report = evaluation.evaluation.get("report", {})
            averages = report.get("weighted avg", {})
            result[evaluation_name]["report"] = report
            result[evaluation_name]["precision"] = averages.get("precision")
            result[evaluation_name]["f1_score"] = averages.get("1-score")
            result[evaluation_name]["errors"] = evaluation.evaluation.get("errors", [])

        return result

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    async def tracker_predict(request: Request) -> HTTPResponse:
        """给定事件列表，预测下一个动作。"""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        request_params = request.json
        try:
            tracker = DialogueStateTracker.from_dict(
                DEFAULT_SENDER_ID, request_params, app.ctx.agent.domain.slots
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                f"Supplied events are not valid. {e}",
                {"parameter": "", "in": "body"},
            )

        try:
            result = app.ctx.agent.predict_next_with_tracker(tracker, verbosity)

            return response.json(result)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "PredictionError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def parse(request: Request) -> HTTPResponse:
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        emulation_mode = request.args.get("emulation_mode")
        emulator = _create_emulator(emulation_mode)

        try:
            data = emulator.normalise_request_json(request.json)
            try:
                # todo Rasa Server -> Agent
                parsed_data = await app.ctx.agent.parse_message(data.get("text"))
            except Exception as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "ParsingError",
                    f"An unexpected error occurred. Error: {e}",
                )
            response_data = emulator.normalise_response_json(parsed_data)

            return response.json(response_data)

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ParsingError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.put("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request) -> HTTPResponse:
        validate_request_body(request, "No path to model file defined in request_body.")

        model_path = request.json.get("model_file", None)
        model_server = request.json.get("model_server", None)
        remote_storage = request.json.get("remote_storage", None)

        if model_server:
            try:
                model_server = EndpointConfig.from_dict(model_server)
            except TypeError as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "BadRequest",
                    f"Supplied 'model_server' is not valid. Error: {e}",
                    {"parameter": "model_server", "in": "body"},
                )

        new_agent = await _load_agent(
            model_path=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            endpoints=endpoints,
        )
        new_agent.lock_store = app.ctx.agent.lock_store
        app.ctx.agent = new_agent

        logger.debug(f"Successfully loaded model '{model_path}'.")
        return response.json(None, status=HTTPStatus.NO_CONTENT)

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request) -> HTTPResponse:
        model_file = app.ctx.agent.model_name

        app.ctx.agent = Agent(lock_store=app.ctx.agent.lock_store)

        logger.debug(f"Successfully unloaded model '{model_file}'.")
        return response.json(None, status=HTTPStatus.NO_CONTENT)

    @app.get("/domain")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def get_domain(request: Request) -> HTTPResponse:
        """以yaml或json格式获取当前域。"""
        # FIXME: this is a false positive mypy error after upgrading to 0.931
        accepts = request.headers.get("Accept", default=JSON_CONTENT_TYPE)
        if accepts.endswith("json"):
            domain = app.ctx.agent.domain.as_dict()
            return response.json(domain)
        elif accepts.endswith("yml") or accepts.endswith("yaml"):
            domain_yaml = app.ctx.agent.domain.as_yaml()
            return response.text(
                domain_yaml, status=HTTPStatus.OK, content_type=YAML_CONTENT_TYPE
            )
        else:
            raise ErrorResponse(
                HTTPStatus.NOT_ACCEPTABLE,
                "NotAcceptable",
                f"Invalid Accept header. Domain can be "
                f"provided as "
                f'json ("Accept: {JSON_CONTENT_TYPE}") or'
                f'yml ("Accept: {YAML_CONTENT_TYPE}"). '
                f"Make sure you've set the appropriate Accept "
                f"header.",
            )

    # =============================================================================
    # 函数结束 - 返回配置完成的 Sanic 应用
    # =============================================================================
    # 此时应用已完全配置，包括：
    # 1. 基础配置（超时、CORS等）
    # 2. 认证配置（JWT、令牌认证）
    # 3. 错误处理配置
    # 4. 所有API端点注册
    # 5. 应用上下文初始化
    # 
    # 返回的应用可以立即用于启动HTTP服务器
    return app


def _get_output_channel(
    request: Request, tracker: Optional[DialogueStateTracker]
) -> OutputChannel:
    """Returns the `OutputChannel` which should be used for the bot's responses.

    Args:
        request: HTTP request whose query parameters can specify which `OutputChannel`
                 should be used.
        tracker: Tracker for the conversation. Used to get the latest input channel.

    Returns:
        `OutputChannel` which should be used to return the bot's responses to.
    """
    requested_output_channel = request.args.get(OUTPUT_CHANNEL_QUERY_KEY)

    if (
        requested_output_channel == USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL
        and tracker
    ):
        requested_output_channel = tracker.get_latest_input_channel()

    # Interactive training does not set `input_channels`, hence we have to be cautious
    registered_input_channels = getattr(request.app.ctx, "input_channels", None) or []
    matching_channels = [
        channel
        for channel in registered_input_channels
        if channel.name() == requested_output_channel
    ]

    # Check if matching channels can provide a valid output channel,
    # otherwise use `CollectingOutputChannel`
    return reduce(
        lambda output_channel_created_so_far, input_channel: (
            input_channel.get_output_channel() or output_channel_created_so_far
        ),
        matching_channels,
        CollectingOutputChannel(),
    )


def _test_data_file_from_payload(request: Request, temporary_directory: Path) -> Text:
    return str(
        _training_payload_from_yaml(
            request,
            temporary_directory,
            # test stories have to prefixed with `test_`
            file_name=f"{TEST_STORIES_FILE_PREFIX}data.yml",
        )["training_files"]
    )


def _training_payload_from_yaml(
    request: Request, temp_dir: Path, file_name: Text = "data.yml"
) -> Dict[Text, Any]:
    logger.debug("Extracting YAML training data from request body.")

    decoded = request.body.decode(rasa.shared.utils.io.DEFAULT_ENCODING)
    _validate_yaml_training_payload(decoded)

    training_data = temp_dir / file_name
    rasa.shared.utils.io.write_text_file(decoded, training_data)

    model_output_directory = str(temp_dir)
    if rasa.utils.endpoints.bool_arg(request, "save_to_default_model_directory", True):
        model_output_directory = DEFAULT_MODELS_PATH

    return dict(
        domain=str(training_data),
        config=str(training_data),
        training_files=str(temp_dir),
        output=model_output_directory,
        force_training=rasa.utils.endpoints.bool_arg(request, "force_training", False),
        core_additional_arguments=_extract_core_additional_arguments(request),
        nlu_additional_arguments=_extract_nlu_additional_arguments(request),
    )


def _nlu_training_payload_from_json(
    request: Request, temp_dir: Path, file_name: Text = "data.json"
) -> Dict[Text, Any]:
    logger.debug("Extracting JSON training data from request body.")

    rasa.shared.utils.validation.validate_training_data(
        request.json,
        rasa.shared.nlu.training_data.schemas.data_schema.rasa_nlu_data_schema(),
    )
    training_data = temp_dir / file_name
    rasa.shared.utils.io.dump_obj_as_json_to_file(training_data, request.json)

    model_output_directory = str(temp_dir)
    if rasa.utils.endpoints.bool_arg(request, "save_to_default_model_directory", True):
        model_output_directory = DEFAULT_MODELS_PATH

    return dict(
        domain=str(training_data),
        config=str(training_data),
        training_files=str(temp_dir),
        output=model_output_directory,
        force_training=rasa.utils.endpoints.bool_arg(request, "force_training", False),
        core_additional_arguments=_extract_core_additional_arguments(request),
        nlu_additional_arguments=_extract_nlu_additional_arguments(request),
    )


def _validate_yaml_training_payload(yaml_text: Text) -> None:
    try:
        RasaYAMLReader().validate(yaml_text)
    except Exception as e:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"The request body does not contain valid YAML. Error: {e}",
            help_url=DOCS_URL_TRAINING_DATA,
        )


def _extract_core_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {
        "augmentation_factor": rasa.utils.endpoints.int_arg(request, "augmentation", 50)
    }


def _extract_nlu_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {"num_threads": rasa.utils.endpoints.int_arg(request, "num_threads", 1)}
