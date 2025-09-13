import logging
import os
from typing import Text, Dict, List, Optional, Any

from packaging import version
from packaging.version import LegacyVersion
from pykwalify.errors import SchemaError

from ruamel.yaml.constructor import DuplicateKeyError

import rasa.shared
from rasa.shared.exceptions import (
    YamlException,
    YamlSyntaxException,
    SchemaValidationError,
)
import rasa.shared.utils.io
from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA,
    PACKAGE_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    SCHEMA_EXTENSIONS_FILE,
    RESPONSES_SCHEMA_FILE,
)

# =============================================================================
# 数据验证模块 - 提供 YAML 和 JSON Schema 验证功能
# =============================================================================

logger = logging.getLogger(__name__)

KEY_TRAINING_DATA_FORMAT_VERSION = "version"  # 训练数据格式版本键名


class YamlValidationException(YamlException, ValueError):
    """当 YAML 文件不符合预期模式时抛出的异常。"""

    def __init__(
        self,
        message: Text,
        validation_errors: Optional[List[SchemaError.SchemaErrorEntry]] = None,
        filename: Optional[Text] = None,
        content: Any = None,
    ) -> None:
        """创建错误实例。

        Args:
            message: 错误消息
            validation_errors: 验证错误列表
            filename: 被验证的文件名
            content: 从文件加载的 YAML 内容（用于行信息）
        """
        super(YamlValidationException, self).__init__(filename)

        self.message = message
        self.validation_errors = validation_errors
        self.content = content

    def __str__(self) -> Text:
        """返回格式化的错误消息。"""
        msg = ""
        if self.filename:
            msg += f"验证 '{self.filename}' 失败。 "
        else:
            msg += "验证 YAML 失败。 "
        msg += self.message
        if self.validation_errors:
            unique_errors = {}
            for error in self.validation_errors:
                line_number = self._line_number_for_path(self.content, error.path)

                if line_number and self.filename:
                    error_representation = f"  在 {self.filename}:{line_number}:\n"
                elif line_number:
                    error_representation = f"  在第 {line_number} 行:\n"
                else:
                    error_representation = ""

                error_representation += f"      {error}"
                unique_errors[str(error)] = error_representation
            error_msg = "\n".join(unique_errors.values())
            msg += f":\n{error_msg}"
        return msg

    def _line_number_for_path(self, current: Any, path: Text) -> Optional[int]:
        """获取当前内容中 YAML 路径的行号。

        使用递归实现：算法沿着路径导航到 YAML 树中的叶子节点。
        不幸的是，并非所有从 ruamel yaml 解析器返回的节点都有附加的行号
        （数组有，字典有），例如字符串没有附加的行号。
        如果我们到达一个没有附加行号的节点，我们将返回父节点的行号 - 
        这是最接近的。

        Args:
            current: 当前内容
            path: 在内容中遍历的路径

        Returns:
            内容中路径的行号
        """
        if not current:
            return None

        this_line = current.lc.line + 1 if hasattr(current, "lc") else None

        if not path:
            return this_line

        if "/" in path:
            head, tail = path.split("/", 1)
        else:
            head, tail = path, ""

        if head:
            if isinstance(current, dict) and head in current:
                return self._line_number_for_path(current[head], tail) or this_line
            elif isinstance(current, list) and head.isdigit():
                return self._line_number_for_path(current[int(head)], tail) or this_line
            else:
                return this_line
        return self._line_number_for_path(current, tail) or this_line


# =============================================================================
# YAML Schema 验证函数
# =============================================================================

def validate_yaml_schema(
    yaml_file_content: Text, schema_path: Text, package_name: Text = PACKAGE_NAME
) -> None:
    """验证 YAML 内容。

    Args:
        yaml_file_content: 要验证的 YAML 文件内容
        schema_path: YAML 文件的模式
        package_name: 模式所在包的名称，默认为 `rasa`
    """
    from pykwalify.core import Core
    from pykwalify.errors import SchemaError
    from ruamel.yaml import YAMLError
    import pkg_resources
    import logging

    log = logging.getLogger("pykwalify")
    log.setLevel(logging.CRITICAL)

    try:
        # 我们需要 "rt" 因为它会向解析输出添加元信息。
        # 这个元信息将包括例如对象被解析的行号。
        # 当我们稍后验证文件并想要指向用户正确的行时，这非常有用
        source_data = rasa.shared.utils.io.read_yaml(
            yaml_file_content, reader_type=["safe", "rt"]
        )
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    schema_file = pkg_resources.resource_filename(package_name, schema_path)
    schema_utils_file = pkg_resources.resource_filename(
        PACKAGE_NAME, RESPONSES_SCHEMA_FILE
    )
    schema_extensions = pkg_resources.resource_filename(
        PACKAGE_NAME, SCHEMA_EXTENSIONS_FILE
    )

    # 使用我们的 YAML 加载器加载模式内容，因为 `pykwalify` 使用全局实例
    # 在并发使用时可能会失败
    schema_content = rasa.shared.utils.io.read_yaml_file(schema_file)
    schema_utils_content = rasa.shared.utils.io.read_yaml_file(schema_utils_file)
    schema_content = dict(schema_content, **schema_utils_content)

    c = Core(
        source_data=source_data,
        schema_data=schema_content,
        extensions=[schema_extensions],
    )

    try:
        c.validate(raise_exception=True)
    except SchemaError:
        raise YamlValidationException(
            "请确保文件正确且所有必需参数都已指定。以下是验证过程中发现的错误",
            c.errors,
            content=source_data,
        )


def validate_training_data(json_data: Dict[Text, Any], schema: Dict[Text, Any]) -> None:
    """Validate rasa training data format to ensure proper training.

    Args:
        json_data: the data to validate
        schema: the schema

    Raises:
        SchemaValidationError if validation fails.
    """
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(json_data, schema)
    except ValidationError as e:
        e.message += (
            f". Failed to validate data, make sure your data "
            f"is valid. For more information about the format visit "
            f"{DOCS_URL_TRAINING_DATA}."
        )
        raise SchemaValidationError.create_from(e) from e


def validate_training_data_format_version(
    yaml_file_content: Dict[Text, Any], filename: Optional[Text]
) -> bool:
    """Validates version on the training data content using `version` field
       and warns users if the file is not compatible with the current version of
       Rasa Open Source.

    Args:
        yaml_file_content: Raw content of training data file as a dictionary.
        filename: Name of the validated file.

    Returns:
        `True` if the file can be processed by current version of Rasa Open Source,
        `False` otherwise.
    """
    if filename:
        filename = os.path.abspath(filename)

    if not isinstance(yaml_file_content, dict):
        raise YamlValidationException(
            "YAML content in is not a mapping, can not validate training "
            "data schema version.",
            filename=filename,
        )

    version_value = yaml_file_content.get(KEY_TRAINING_DATA_FORMAT_VERSION)

    if not version_value:
        # not raising here since it's not critical
        logger.info(
            f"The '{KEY_TRAINING_DATA_FORMAT_VERSION}' key is missing in "
            f"the training data file {filename}. "
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file. "
            f"See {DOCS_URL_TRAINING_DATA}."
        )
        return True

    try:
        if isinstance(version_value, str):
            version_value = version_value.strip("\"'")
        parsed_version = version.parse(version_value)
        latest_version = version.parse(LATEST_TRAINING_DATA_FORMAT_VERSION)

        if isinstance(parsed_version, LegacyVersion):
            raise TypeError

        if parsed_version < latest_version:
            rasa.shared.utils.io.raise_warning(
                f"Training data file {filename} has a lower "
                f"format version than your Rasa Open Source installation: "
                f"{version_value} < {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
                f"Rasa Open Source will read the file as a version "
                f"{LATEST_TRAINING_DATA_FORMAT_VERSION} file. "
                f"Please update your version key to "
                f"{LATEST_TRAINING_DATA_FORMAT_VERSION}. "
                f"See {DOCS_URL_TRAINING_DATA}."
            )

        if latest_version >= parsed_version:

            return True

    except TypeError:
        rasa.shared.utils.io.raise_warning(
            f"Training data file {filename} must specify "
            f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' as string, for example:\n"
            f"{KEY_TRAINING_DATA_FORMAT_VERSION}: "
            f"'{LATEST_TRAINING_DATA_FORMAT_VERSION}'\n"
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file.",
            docs=DOCS_URL_TRAINING_DATA,
        )
        return True

    rasa.shared.utils.io.raise_warning(
        f"Training data file {filename} has a greater "
        f"format version than your Rasa Open Source installation: "
        f"{version_value} > {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
        f"Please consider updating to the latest version of Rasa Open Source."
        f"This file will be skipped.",
        docs=DOCS_URL_TRAINING_DATA,
    )
    return False
