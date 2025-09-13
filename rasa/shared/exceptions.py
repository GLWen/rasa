import json
from typing import Optional, Text

import jsonschema
from ruamel.yaml.error import (
    MarkedYAMLError,
    MarkedYAMLWarning,
    MarkedYAMLFutureWarning,
)

# =============================================================================
# Rasa 异常层次结构定义
# =============================================================================

class RasaException(Exception):
    """Rasa 开源版本中所有错误的基础异常类。
    
    这些异常是由于无效的使用情况而产生的，会向用户报告，
    但在遥测中会被忽略。
    """


class RasaCoreException(RasaException):
    """Rasa Core 模块错误的基础异常类。"""


class InvalidParameterException(RasaException, ValueError):
    """当使用无效参数时抛出的异常。"""


# =============================================================================
# YAML 处理异常系统
# =============================================================================

class YamlException(RasaException):
    """当读取 YAML 文件时发生错误时抛出的异常。"""

    def __init__(self, filename: Optional[Text] = None) -> None:
        """创建异常实例。

        Args:
            filename: 发生错误的可选文件名"""
        self.filename = filename


class YamlSyntaxException(YamlException):
    """当 YAML 文件由于语法错误而无法正确解析时抛出的异常。"""

    def __init__(
        self,
        filename: Optional[Text] = None,
        underlying_yaml_exception: Optional[Exception] = None,
    ) -> None:
        super(YamlSyntaxException, self).__init__(filename)

        self.underlying_yaml_exception = underlying_yaml_exception

    def __str__(self) -> Text:
        """返回格式化的异常信息，包含用户友好的错误描述和修复建议。"""
        if self.filename:
            exception_text = f"读取文件 '{self.filename}' 失败。"
        else:
            exception_text = "读取 YAML 文件失败。"

        if self.underlying_yaml_exception:
            # 处理 ruamel.yaml 的特定异常类型，清理冗余信息
            if isinstance(
                self.underlying_yaml_exception,
                (MarkedYAMLError, MarkedYAMLWarning, MarkedYAMLFutureWarning),
            ):
                self.underlying_yaml_exception.note = None
            if isinstance(
                self.underlying_yaml_exception,
                (MarkedYAMLWarning, MarkedYAMLFutureWarning),
            ):
                self.underlying_yaml_exception.warn = None
            exception_text += f" {self.underlying_yaml_exception}"

        if self.filename:
            # 将通用错误信息替换为具体的文件名
            exception_text = exception_text.replace(
                'in "<unicode string>"', f'in "{self.filename}"'
            )

        # 添加用户友好的修复建议
        exception_text += (
            "\n\n您可以使用 https://yamlchecker.com/ 来验证 "
            "YAML 文件的语法。"
        )
        return exception_text


# =============================================================================
# 文件系统异常
# =============================================================================

class FileNotFoundException(RasaException, FileNotFoundError):
    """当预期存在的文件不存在时抛出的异常。"""


class FileIOException(RasaException):
    """当进行文件 IO 操作时发生错误时抛出的异常。"""


# =============================================================================
# 配置和功能支持异常
# =============================================================================

class InvalidConfigException(ValueError, RasaException):
    """当遇到无效配置时抛出的异常。"""


class UnsupportedFeatureException(RasaCoreException):
    """当请求的功能不受支持时抛出的异常。"""


# =============================================================================
# 数据验证异常
# =============================================================================

class SchemaValidationError(RasaException, jsonschema.ValidationError):
    """当通过 `jsonschema` 进行 schema 验证失败时抛出的异常。"""


class InvalidEntityFormatException(RasaException, json.JSONDecodeError):
    """当实体格式无效时抛出的异常。"""

    @classmethod
    def create_from(
        cls, other: json.JSONDecodeError, msg: Text
    ) -> "InvalidEntityFormatException":
        """从 `JSONDecodeError` 创建 `InvalidEntityFormatException`。
        
        Args:
            other: 原始的 JSONDecodeError 异常
            msg: 自定义错误消息
            
        Returns:
            新的 InvalidEntityFormatException 实例
        """
        return cls(msg, other.doc, other.pos)


# =============================================================================
# 连接异常
# =============================================================================

class ConnectionException(RasaException):
    """当连接到第三方服务失败时抛出的异常。

    它被我们的代理和跟踪器存储类使用，当它们无法连接到
    postgres、dynamoDB、mongo 等服务时。
    """
