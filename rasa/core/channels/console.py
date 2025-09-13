# 这个内置模块是必需的，这样我们可以在测试中重写它
import asyncio
# 导入JSON处理模块，用于序列化和反序列化
import json
# 导入日志模块，用于记录日志信息
import logging
# 导入操作系统接口，用于环境变量等操作
import os

# 导入类型提示相关的类型
from typing import (
    Any,           # 任意类型
    AsyncGenerator, # 异步生成器类型
    Dict,          # 字典类型
    List,          # 列表类型
    Optional,      # 可选类型
    Text,          # 文本类型（字符串的别名）
    overload,      # 函数重载装饰器
)

# 导入异步HTTP客户端
import aiohttp
# 导入交互式命令行工具
import questionary
# 导入HTTP客户端超时配置
from aiohttp import ClientTimeout
# 导入提示工具包样式
from prompt_toolkit.styles import Style

# 导入Rasa共享CLI工具
import rasa.shared.utils.cli
# 导入Rasa共享IO工具
import rasa.shared.utils.io
# 导入Rasa CLI工具
from rasa.cli import utils as cli_utils
# 导入Rasa核心工具
from rasa.core import utils
# 导入REST输入通道
from rasa.core.channels.rest import RestInput
# 导入默认服务器URL和流读取超时常量
from rasa.core.constants import DEFAULT_SERVER_URL, DEFAULT_STREAM_READING_TIMEOUT
# 导入意图消息前缀常量
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
# 导入默认编码常量
from rasa.shared.utils.io import DEFAULT_ENCODING

# 创建日志记录器
logger = logging.getLogger(__name__)

# 流读取超时环境变量名
STREAM_READING_TIMEOUT_ENV = "RASA_SHELL_STREAM_READING_TIMEOUT_IN_SECONDS"


def print_buttons(
    message: Dict[Text, Any],
    is_latest_message: bool = False,
    color: Text = rasa.shared.utils.io.bcolors.OKBLUE,
) -> Optional[questionary.Question]:
    """从消息数据创建CLI按钮。
    
    Args:
        message: 包含按钮数据的消息字典
        is_latest_message: 是否为最新消息（决定是否显示交互式选择）
        color: 打印颜色
        
    Returns:
        如果是最新消息则返回交互式问题对象，否则返回None
    """
    if is_latest_message:
        # 从消息数据创建按钮选择项，允许自由文本输入
        choices = cli_utils.button_choices_from_message_data(
            message, allow_free_text_input=True
        )
        # 创建交互式选择问题
        question = questionary.select(
            message.get("text"),  # 问题文本
            choices,              # 选择项列表
            style=Style([("qmark", "#6d91d3"), ("", "#6d91d3"), ("answer", "#b373d6")]),  # 样式配置
        )
        return question
    else:
        # 打印按钮标题
        rasa.shared.utils.cli.print_color("Buttons:", color=color)
        # 遍历并打印每个按钮
        for idx, button in enumerate(message.get("buttons")):
            rasa.shared.utils.cli.print_color(
                cli_utils.button_to_string(button, idx), color=color
            )
        return None


def _print_bot_output(
    message: Dict[Text, Any],
    is_latest_message: bool = False,
    color: Text = rasa.shared.utils.io.bcolors.OKBLUE,
) -> Optional[questionary.Question]:
    """打印机器人输出消息。
    
    Args:
        message: 机器人消息字典
        is_latest_message: 是否为最新消息
        color: 打印颜色
        
    Returns:
        如果消息包含按钮且为最新消息则返回交互式问题对象，否则返回None
    """
    # 处理按钮消息
    if "buttons" in message:
        question = print_buttons(message, is_latest_message, color)
        if question:
            return question

    # 处理文本消息
    if "text" in message:
        rasa.shared.utils.cli.print_color(message["text"], color=color)

    # 处理图片消息
    if "image" in message:
        rasa.shared.utils.cli.print_color("Image: " + message["image"], color=color)

    # 处理附件消息
    if "attachment" in message:
        rasa.shared.utils.cli.print_color(
            "Attachment: " + message["attachment"], color=color
        )

    # 处理元素消息（如卡片、轮播等）
    if "elements" in message:
        rasa.shared.utils.cli.print_color("Elements:", color=color)
        for idx, element in enumerate(message["elements"]):
            rasa.shared.utils.cli.print_color(
                cli_utils.element_to_string(element, idx), color=color
            )

    # 处理快速回复消息
    if "quick_replies" in message:
        rasa.shared.utils.cli.print_color("Quick Replies:", color=color)
        for idx, element in enumerate(message["quick_replies"]):
            rasa.shared.utils.cli.print_color(
                cli_utils.button_to_string(element, idx), color=color
            )

    # 处理自定义JSON消息
    if "custom" in message:
        rasa.shared.utils.cli.print_color("Custom json:", color=color)
        rasa.shared.utils.cli.print_color(
            rasa.shared.utils.io.json_to_string(message["custom"]), color=color
        )

    return None


@overload
async def _get_user_input(previous_response: None) -> Text:
    """获取用户输入的重载函数（无前一个响应）。"""
    ...


@overload
async def _get_user_input(previous_response: Dict[str, Any]) -> Optional[Text]:
    """获取用户输入的重载函数（有前一个响应）。"""
    ...


async def _get_user_input(
    previous_response: Optional[Dict[str, Any]]
) -> Optional[Text]:
    """获取用户输入。
    
    Args:
        previous_response: 前一个机器人响应，如果为None则直接提示用户输入
        
    Returns:
        用户输入的文本，如果用户取消输入则返回None
    """
    button_response = None
    # 如果有前一个响应，先显示它并获取按钮响应
    if previous_response is not None:
        button_response = _print_bot_output(previous_response, is_latest_message=True)

    if button_response is not None:
        # 从按钮问题获取响应
        response = await cli_utils.payload_from_button_question(button_response)
        if response == cli_utils.FREE_TEXT_INPUT_PROMPT:
            # 如果用户选择自由文本输入，重新提示用户
            response = await _get_user_input(None)
    else:
        # 创建文本输入问题
        question = questionary.text(
            "",  # 空提示文本
            qmark="Your input ->",  # 问题标记
            style=Style([("qmark", "#b373d6"), ("", "#b373d6")]),  # 样式配置
        )
        # 异步获取用户输入
        response = await question.ask_async()
    # 返回去除首尾空白的响应，如果为None则返回None
    return response.strip() if response is not None else None


async def send_message_receive_block(
    server_url: Text, auth_token: Text, sender_id: Text, message: Text
) -> List[Dict[Text, Any]]:
    """发送消息并返回响应（阻塞模式）。
    
    Args:
        server_url: 服务器URL
        auth_token: 认证令牌
        sender_id: 发送者ID
        message: 消息内容
        
    Returns:
        机器人响应列表
    """
    # 构建请求负载
    payload = {"sender": sender_id, "message": message}

    # 构建请求URL
    url = f"{server_url}/webhooks/rest/webhook?token={auth_token}"
    # 创建HTTP会话并发送POST请求
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:
            return await resp.json()


async def _send_message_receive_stream(
    server_url: Text,
    auth_token: Text,
    sender_id: Text,
    message: Text,
    request_timeout: Optional[int] = None,
) -> AsyncGenerator[Dict[Text, Any], None]:
    """发送消息并接收流式响应。
    
    Args:
        server_url: 服务器URL
        auth_token: 认证令牌
        sender_id: 发送者ID
        message: 消息内容
        request_timeout: 请求超时时间
        
    Yields:
        流式响应中的每个消息字典
    """
    # 构建请求负载
    payload = {"sender": sender_id, "message": message}

    # 构建流式请求URL
    url = f"{server_url}/webhooks/rest/webhook?stream=true&token={auth_token}"

    # 定义超时以防止服务器崩溃时继续读取
    timeout = _get_stream_reading_timeout(request_timeout)

    # 创建HTTP会话并发送POST请求
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:
            # 逐行读取流式响应
            async for line in resp.content:
                if line:
                    # 解码并解析JSON响应
                    yield json.loads(line.decode(DEFAULT_ENCODING))


def _get_stream_reading_timeout(request_timeout: Optional[int] = None) -> ClientTimeout:
    """定义带有回退机制的ClientTimeout。

    首先使用`request_timeout`函数参数（如果可用），这来自`--request-timeout`命令行参数。
    如果失败，则回退到`STREAM_READING_TIMEOUT_ENV`环境变量。
    最后回退到`rasa.core.constants`中的`DEFAULT_STREAM_READING_TIMEOUT`。

    Args:
        request_timeout: 请求超时时间（秒）
        
    Returns:
        配置好的ClientTimeout对象
    """
    # 按优先级获取超时时间
    timeout_str = (
        request_timeout
        if request_timeout is not None
        else os.environ.get(STREAM_READING_TIMEOUT_ENV, DEFAULT_STREAM_READING_TIMEOUT)
    )

    return ClientTimeout(int(timeout_str))


async def record_messages(
    sender_id: Text,
    server_url: Text = DEFAULT_SERVER_URL,
    auth_token: Text = "",
    max_message_limit: Optional[int] = None,
    use_response_stream: bool = True,
    request_timeout: Optional[int] = None,
) -> int:
    """从命令行读取消息并打印机器人响应。
    
    Args:
        sender_id: 发送者ID
        server_url: 服务器URL
        auth_token: 认证令牌
        max_message_limit: 最大消息数量限制
        use_response_stream: 是否使用流式响应
        request_timeout: 请求超时时间
        
    Returns:
        处理的消息数量
    """
    # 定义退出文本
    exit_text = INTENT_MESSAGE_PREFIX + "stop"

    # 打印成功消息和提示
    rasa.shared.utils.cli.print_success(
        "Bot loaded. Type a message and press enter "
        "(use '{}' to exit): ".format(exit_text)
    )

    num_messages = 0
    previous_response = None
    # 等待服务器启动
    await asyncio.sleep(0.5)
    # 主消息循环
    while not utils.is_limit_reached(num_messages, max_message_limit):
        # 获取用户输入
        text = await _get_user_input(previous_response)

        # 检查退出条件
        if text == exit_text or text is None:
            break

        if use_response_stream:
            # 使用流式响应模式
            bot_responses_stream = _send_message_receive_stream(
                server_url, auth_token, sender_id, text, request_timeout=request_timeout
            )
            previous_response = None
            # 处理流式响应
            async for response in bot_responses_stream:
                if previous_response is not None:
                    _print_bot_output(previous_response)
                previous_response = response
        else:
            # 使用阻塞响应模式
            bot_responses = await send_message_receive_block(
                server_url, auth_token, sender_id, text
            )
            previous_response = None
            # 处理阻塞响应
            for response in bot_responses:
                if previous_response is not None:
                    _print_bot_output(previous_response)
                previous_response = response

        num_messages += 1
        # 让出事件循环给其他协程
        await asyncio.sleep(0)
    return num_messages


class CmdlineInput(RestInput):
    """命令行输入通道类，继承自RestInput。"""
    
    @classmethod
    def name(cls) -> Text:
        """返回通道名称。
        
        Returns:
            通道名称"cmdline"
        """
        return "cmdline"

    def url_prefix(self) -> Text:
        """返回URL前缀。
        
        Returns:
            REST输入通道的名称作为URL前缀
        """
        return RestInput.name()
