import sys
from typing import Any, Text, NoReturn

import rasa.shared.utils.io

# =============================================================================
# 命令行界面工具模块 - 提供彩色输出和错误处理功能
# =============================================================================

def print_color(*args: Any, color: Text) -> None:
    """以指定颜色将给定参数打印到 STDOUT。

    Args:
        args: 要打印的对象列表
        color: 颜色的文本表示
    """
    output = rasa.shared.utils.io.wrap_with_color(*args, color=color)
    stream = sys.stdout
    if sys.platform == "win32":
        # 使用 colorama 修复 Windows 上无法打印颜色的回归问题
        # https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
    try:
        print(output, file=stream)
    except BlockingIOError:
        rasa.shared.utils.io.handle_print_blocking(output)


def print_success(*args: Any) -> None:
    """以绿色将给定参数打印到 STDOUT，表示成功。

    Args:
        args: 要打印的对象列表
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKGREEN)


def print_info(*args: Any) -> None:
    """以蓝色将给定参数打印到 STDOUT。

    Args:
        args: 要打印的对象列表
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKBLUE)


def print_warning(*args: Any) -> None:
    """以警告颜色将给定参数打印到 STDOUT。

    Args:
        args: 要打印的对象列表
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.WARNING)


def print_error(*args: Any) -> None:
    """以错误颜色将给定参数打印到 STDOUT。

    Args:
        args: 要打印的对象列表
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """打印错误消息并退出应用程序。

    Args:
        message: 要打印的错误消息
        exit_code: 程序退出代码，默认为 1
    """
    print_error(message)
    sys.exit(exit_code)
