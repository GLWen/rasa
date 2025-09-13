from collections import OrderedDict
import errno
import glob
from hashlib import md5
from io import StringIO
import json
import os
import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Text, Type, Union
import warnings
import random
import string

import portalocker

from ruamel import yaml as yaml
from ruamel.yaml import RoundTripRepresenter, YAMLError
from ruamel.yaml.constructor import DuplicateKeyError, BaseConstructor, ScalarNode

from rasa.shared.constants import (
    DEFAULT_LOG_LEVEL,
    ENV_LOG_LEVEL,
    NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
    CONFIG_SCHEMA_FILE,
    MODEL_CONFIG_SCHEMA_FILE,
)
from rasa.shared.exceptions import (
    FileIOException,
    FileNotFoundException,
    YamlSyntaxException,
    RasaException,
)
import rasa.shared.utils.validation

# =============================================================================
# 文件 IO 操作模块 - 提供文件读写、YAML/JSON 处理、目录操作等功能
# =============================================================================

DEFAULT_ENCODING = "utf-8"  # 默认文件编码
YAML_VERSION = (1, 2)  # 支持的 YAML 版本


# =============================================================================
# 颜色输出相关类
# =============================================================================

class bcolors:
    """ANSI 颜色代码常量类，用于终端彩色输出。"""
    HEADER = "\033[95m"      # 紫色标题
    OKBLUE = "\033[94m"      # 蓝色（信息）
    OKGREEN = "\033[92m"     # 绿色（成功）
    WARNING = "\033[93m"     # 黄色（警告）
    FAIL = "\033[91m"        # 红色（错误）
    ENDC = "\033[0m"         # 结束颜色
    BOLD = "\033[1m"         # 粗体
    UNDERLINE = "\033[4m"    # 下划线


def wrap_with_color(*args: Any, color: Text) -> Text:
    """用指定颜色包装文本。
    
    Args:
        *args: 要包装的文本参数
        color: 颜色代码

    Returns:
        带颜色代码的文本字符串
    """
    return color + " ".join(str(s) for s in args) + bcolors.ENDC


# =============================================================================
# 警告和日志相关函数
# =============================================================================

def raise_warning(
    message: Text,
    category: Optional[Type[Warning]] = None,
    docs: Optional[Text] = None,
    **kwargs: Any,
) -> None:
    """发出带有合理默认值和彩色警告消息的 `warnings.warn`。
    
    Args:
        message: 警告消息
        category: 警告类别
        docs: 相关文档链接
        **kwargs: 其他警告参数
    """
    original_formatter = warnings.formatwarning

    def should_show_source_line() -> bool:
        if "stacklevel" not in kwargs:
            if category == UserWarning or category is None:
                return False
            if category == FutureWarning:
                return False
        return True

    def formatwarning(
        message: Union[Warning, Text],
        category: Type[Warning],
        filename: Text,
        lineno: int,
        line: Optional[Text] = None,
    ) -> Text:
        """以标准方式格式化警告的函数。"""
        if not should_show_source_line():
            if docs:
                line = f"更多信息请访问 {docs}"
            else:
                line = ""

        formatted_message = original_formatter(
            message, category, filename, lineno, line
        )
        return wrap_with_color(formatted_message, color=bcolors.WARNING)

    if "stacklevel" not in kwargs:
        # 为最常见的警告类别设置有用的默认值
        if category == DeprecationWarning:
            kwargs["stacklevel"] = 3
        elif category in (UserWarning, FutureWarning):
            kwargs["stacklevel"] = 2

    warnings.formatwarning = formatwarning
    warnings.warn(message, category=category, **kwargs)
    warnings.formatwarning = original_formatter


# =============================================================================
# 基础文件操作函数
# =============================================================================

def write_text_file(
    content: Text,
    file_path: Union[Text, Path],
    encoding: Text = DEFAULT_ENCODING,
    append: bool = False,
) -> None:
    """将文本写入文件。

    Args:
        content: 要写入的内容
        file_path: 内容应写入的路径
        encoding: 应使用的编码
        append: 是否追加到文件或截断文件
    """
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)


def read_file(filename: Union[Text, Path], encoding: Text = DEFAULT_ENCODING) -> Any:
    """从文件读取文本。
    
    Args:
        filename: 要读取的文件路径
        encoding: 文件编码
        
    Returns:
        文件内容
        
    Raises:
        FileNotFoundException: 文件不存在时
        FileIOException: 文件读取错误时
    """
    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundException(
            f"读取文件失败，'{os.path.abspath(filename)}' 不存在。"
        )
    except UnicodeDecodeError:
        raise FileIOException(
            f"读取文件 '{os.path.abspath(filename)}' 失败，"
            f"无法使用 {encoding} 编码解码文件。请确保文件使用此编码存储。"
        )


def read_json_file(filename: Union[Text, Path]) -> Any:
    """从文件读取 JSON 数据。
    
    Args:
        filename: JSON 文件路径
        
    Returns:
        解析后的 JSON 数据
        
    Raises:
        FileIOException: JSON 解析错误时
    """
    content = read_file(filename)
    try:
        return json.loads(content)
    except ValueError as e:
        raise FileIOException(
            f"从 '{os.path.abspath(filename)}' 读取 JSON 失败。错误: {e}"
        )


# =============================================================================
# 目录操作函数
# =============================================================================

def list_directory(path: Text) -> List[Text]:
    """返回所有文件和文件夹，排除隐藏文件。

    如果路径指向文件，则返回该文件。这是一个递归实现，
    返回路径中任何深度的文件。
    
    Args:
        path: 要列出的路径
        
    Returns:
        文件和目录路径列表
        
    Raises:
        ValueError: 路径类型错误或资源不存在时
    """
    if not isinstance(path, str):
        raise ValueError(
            f"`resource_name` 必须是字符串类型。得到 `{type(path)}` 而不是字符串"
        )

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results: List[Text] = []
        for base, dirs, files in os.walk(path, followlinks=True):
            # 对文件进行排序，确保跨运行的一致性顺序
            files = sorted(files, key=_filename_without_prefix)
            # 添加非隐藏文件
            good_files = filter(lambda x: not x.startswith("."), files)
            results.extend(os.path.join(base, f) for f in good_files)
            # 添加非隐藏目录
            good_directories = filter(lambda x: not x.startswith("."), dirs)
            results.extend(os.path.join(base, f) for f in good_directories)
        return results
    else:
        raise ValueError(f"无法定位资源 '{os.path.abspath(path)}'。")


def list_files(path: Text) -> List[Text]:
    """返回所有文件，排除隐藏文件。

    如果路径指向文件，则返回该文件。
    
    Args:
        path: 要列出的路径
        
    Returns:
        文件路径列表
    """
    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def _filename_without_prefix(file: Text) -> Text:
    """分割文件名的前缀，直到第一个 ``_`` 之后。
    
    Args:
        file: 文件名
        
    Returns:
        去除前缀后的文件名
    """
    return "_".join(file.split("_")[1:])


def list_subdirectories(path: Text) -> List[Text]:
    """返回所有文件夹，排除隐藏文件。

    如果路径指向文件，则返回空列表。
    
    Args:
        path: 要列出的路径
        
    Returns:
        子目录路径列表
    """
    return [fn for fn in glob.glob(os.path.join(path, "*")) if os.path.isdir(fn)]


# =============================================================================
# 哈希和指纹计算函数
# =============================================================================

def deep_container_fingerprint(
    obj: Union[List[Any], Dict[Any, Any], Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """计算稳定的哈希值。

    适用于列表和字典。对于键和值，我们递归调用 `hash(...)`。
    对于字典，哈希值与容器的键顺序无关。请记住，
    不同顺序的列表不会创建相同的哈希！

    Args:
        obj: 要哈希的字典或列表
        encoding: 用于将对象转储为字符串的编码

    Returns:
        容器的哈希值
    """
    if isinstance(obj, dict):
        return get_dictionary_fingerprint(obj, encoding)
    elif isinstance(obj, list):
        return get_list_fingerprint(obj, encoding)
    elif hasattr(obj, "fingerprint") and callable(obj.fingerprint):
        return obj.fingerprint()
    else:
        return get_text_hash(str(obj), encoding)


def get_dictionary_fingerprint(
    dictionary: Dict[Any, Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """计算字典的指纹。

    字典可以包含任何键和值，这些键和值可以是字典、
    列表或可以转储为字符串的元素。

    Args:
        dictionary: 要哈希的字典
        encoding: 用于将对象转储为字符串的编码

    Returns:
        字典的哈希值
    """
    stringified = json.dumps(
        {
            deep_container_fingerprint(k, encoding): deep_container_fingerprint(
                v, encoding
            )
            for k, v in dictionary.items()
        },
        sort_keys=True,
    )
    return get_text_hash(stringified, encoding)


def get_list_fingerprint(
    elements: List[Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """计算无序列表的指纹。

    Args:
        elements: 无序列表
        encoding: 用于将对象转储为字符串的编码

    Returns:
        列表的指纹
    """
    stringified = json.dumps(
        [deep_container_fingerprint(element, encoding) for element in elements]
    )
    return get_text_hash(stringified, encoding)


def get_text_hash(text: Text, encoding: Text = DEFAULT_ENCODING) -> Text:
    """计算文本的 md5 哈希值。
    
    Args:
        text: 要哈希的文本
        encoding: 文本编码
        
    Returns:
        文本的 MD5 哈希值
    """
    return md5(text.encode(encoding)).hexdigest()  # nosec


def json_to_string(obj: Any, **kwargs: Any) -> Text:
    """Dumps a JSON-serializable object to string.

    Args:
        obj: JSON-serializable object.
        kwargs: serialization options. Defaults to 2 space indentation
                and disable escaping of non-ASCII characters.

    Returns:
        The objects serialized to JSON, as a string.
    """
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def fix_yaml_loader() -> None:
    """Ensure that any string read by yaml is represented as unicode."""

    def construct_yaml_str(self: BaseConstructor, node: ScalarNode) -> Any:
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    yaml.Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)


def replace_environment_variables() -> None:
    """Enable yaml loader to process the environment variables in the yaml."""
    # eg. ${USER_NAME}, ${PASSWORD}
    env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
    yaml.Resolver.add_implicit_resolver("!env_var", env_var_pattern, None)

    def env_var_constructor(loader: BaseConstructor, node: ScalarNode) -> Text:
        """Process environment variables found in the YAML."""
        value = loader.construct_scalar(node)
        expanded_vars = os.path.expandvars(value)
        not_expanded = [
            w for w in expanded_vars.split() if w.startswith("$") and w in value
        ]
        if not_expanded:
            raise RasaException(
                f"Error when trying to expand the "
                f"environment variables in '{value}'. "
                f"Please make sure to also set these "
                f"environment variables: '{not_expanded}'."
            )
        return expanded_vars

    yaml.SafeConstructor.add_constructor("!env_var", env_var_constructor)


fix_yaml_loader()
replace_environment_variables()


def read_yaml(content: Text, reader_type: Union[Text, List[Text]] = "safe") -> Any:
    """Parses yaml from a text.

    Args:
        content: A text containing yaml content.
        reader_type: Reader type to use. By default "safe" will be used.

    Raises:
        ruamel.yaml.parser.ParserError: If there was an error when parsing the YAML.
    """
    if _is_ascii(content):
        # Required to make sure emojis are correctly parsed
        content = (
            content.encode("utf-8")
            .decode("raw_unicode_escape")
            .encode("utf-16", "surrogatepass")
            .decode("utf-16")
        )

    yaml_parser = yaml.YAML(typ=reader_type)
    yaml_parser.version = YAML_VERSION  # type: ignore[assignment]
    yaml_parser.preserve_quotes = True  # type: ignore[assignment]

    return yaml_parser.load(content) or {}


def _is_ascii(text: Text) -> bool:
    return all(ord(character) < 128 for character in text)


def read_yaml_file(
    filename: Union[Text, Path], reader_type: Union[Text, List[Text]] = "safe"
) -> Union[List[Any], Dict[Text, Any]]:
    """Parses a yaml file.

    Raises an exception if the content of the file can not be parsed as YAML.

    Args:
        filename: The path to the file which should be read.
        reader_type: Reader type to use. By default "safe" will be used.

    Returns:
        Parsed content of the file.
    """
    try:
        return read_yaml(read_file(filename, DEFAULT_ENCODING), reader_type)
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(filename, e)


def write_yaml(
    data: Any,
    target: Union[Text, Path, StringIO],
    should_preserve_key_order: bool = False,
) -> None:
    """Writes a yaml to the file or to the stream.

    Args:
        data: The data to write.
        target: The path to the file which should be written or a stream object
        should_preserve_key_order: Whether to force preserve key order in `data`.
    """
    _enable_ordered_dict_yaml_dumping()

    if should_preserve_key_order:
        data = convert_to_ordered_dict(data)

    dumper = yaml.YAML()
    # no wrap lines
    dumper.width = YAML_LINE_MAX_WIDTH  # type: ignore[assignment]

    # use `null` to represent `None`
    dumper.representer.add_representer(
        type(None),
        lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
    )

    if isinstance(target, StringIO):
        dumper.dump(data, target)
        return

    with Path(target).open("w", encoding=DEFAULT_ENCODING) as outfile:
        dumper.dump(data, outfile)


YAML_LINE_MAX_WIDTH = 4096


def is_key_in_yaml(file_path: Union[Text, Path], *keys: Text) -> bool:
    """Checks if any of the keys is contained in the root object of the yaml file.

    Arguments:
        file_path: path to the yaml file
        keys: keys to look for

    Returns:
          `True` if at least one of the keys is found, `False` otherwise.

    Raises:
        FileNotFoundException: if the file cannot be found.
    """
    try:
        with open(file_path, encoding=DEFAULT_ENCODING) as file:
            return any(
                any(line.lstrip().startswith(f"{key}:") for key in keys)
                for line in file
            )
    except FileNotFoundError:
        raise FileNotFoundException(
            f"Failed to read file, " f"'{os.path.abspath(file_path)}' does not exist."
        )


def convert_to_ordered_dict(obj: Any) -> Any:
    """Convert object to an `OrderedDict`.

    Args:
        obj: Object to convert.

    Returns:
        An `OrderedDict` with all nested dictionaries converted if `obj` is a
        dictionary, otherwise the object itself.
    """
    if isinstance(obj, OrderedDict):
        return obj
    # use recursion on lists
    if isinstance(obj, list):
        return [convert_to_ordered_dict(element) for element in obj]

    if isinstance(obj, dict):
        out = OrderedDict()
        # use recursion on dictionaries
        for k, v in obj.items():
            out[k] = convert_to_ordered_dict(v)

        return out

    # return all other objects
    return obj


def _enable_ordered_dict_yaml_dumping() -> None:
    """Ensure that `OrderedDict`s are dumped so that the order of keys is respected."""
    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )


def is_logging_disabled() -> bool:
    """Returns `True` if log level is set to WARNING or ERROR, `False` otherwise."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    return log_level in ("ERROR", "WARNING")


def create_directory_for_file(file_path: Union[Text, Path]) -> None:
    """Creates any missing parent directories of this file path."""
    create_directory(os.path.dirname(file_path))


def dump_obj_as_json_to_file(filename: Union[Text, Path], obj: Any) -> None:
    """Dump an object as a json string to a file."""
    write_text_file(json.dumps(obj, ensure_ascii=False, indent=2), filename)


def dump_obj_as_yaml_to_string(
    obj: Any, should_preserve_key_order: bool = False
) -> Text:
    """Writes data (python dict) to a yaml string.

    Args:
        obj: The object to dump. Has to be serializable.
        should_preserve_key_order: Whether to force preserve key order in `data`.

    Returns:
        The object converted to a YAML string.
    """
    buffer = StringIO()

    write_yaml(obj, buffer, should_preserve_key_order=should_preserve_key_order)

    return buffer.getvalue()


def create_directory(directory_path: Text) -> None:
    """Creates a directory and its super paths.

    Succeeds even if the path already exists.
    """
    try:
        os.makedirs(directory_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def raise_deprecation_warning(
    message: Text,
    warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
    docs: Optional[Text] = None,
    **kwargs: Any,
) -> None:
    """Thin wrapper around `raise_warning()` to raise a deprecation warning. It requires
    a version until which we'll warn, and after which the support for the feature will
    be removed.
    """
    if warn_until_version not in message:
        message = f"{message} (will be removed in {warn_until_version})"

    # need the correct stacklevel now
    kwargs.setdefault("stacklevel", 3)
    # we're raising a `FutureWarning` instead of a `DeprecationWarning` because
    # we want these warnings to be visible in the terminal of our users
    # https://docs.python.org/3/library/warnings.html#warning-categories
    raise_warning(message, FutureWarning, docs, **kwargs)


def read_validated_yaml(
    filename: Union[Text, Path],
    schema: Text,
    reader_type: Union[Text, List[Text]] = "safe",
) -> Any:
    """Validates YAML file content and returns parsed content.

    Args:
        filename: The path to the file which should be read.
        schema: The path to the schema file which should be used for validating the
            file content.
        reader_type: Reader type to use. By default "safe" will be used.

    Returns:
        The parsed file content.

    Raises:
        YamlValidationException: In case the model configuration doesn't match the
            expected schema.
    """
    content = read_file(filename)

    rasa.shared.utils.validation.validate_yaml_schema(content, schema)
    return read_yaml(content, reader_type)


def read_config_file(
    filename: Union[Path, Text], reader_type: Union[Text, List[Text]] = "safe"
) -> Dict[Text, Any]:
    """Parses a yaml configuration file. Content needs to be a dictionary.

    Args:
        filename: The path to the file which should be read.
        reader_type: Reader type to use. By default "safe" will be used.

    Raises:
        YamlValidationException: In case file content is not a `Dict`.

    Returns:
        Parsed config file.
    """
    return read_validated_yaml(filename, CONFIG_SCHEMA_FILE, reader_type)


def read_model_configuration(filename: Union[Path, Text]) -> Dict[Text, Any]:
    """Parses a model configuration file.

    Args:
        filename: The path to the file which should be read.

    Raises:
        YamlValidationException: In case the model configuration doesn't match the
            expected schema.

    Returns:
        Parsed config file.
    """
    return read_validated_yaml(filename, MODEL_CONFIG_SCHEMA_FILE)


def is_subdirectory(path: Text, potential_parent_directory: Text) -> bool:
    """Checks if `path` is a subdirectory of `potential_parent_directory`.

    Args:
        path: Path to a file or directory.
        potential_parent_directory: Potential parent directory.

    Returns:
        `True` if `path` is a subdirectory of `potential_parent_directory`.
    """
    if path is None or potential_parent_directory is None:
        return False

    path = os.path.abspath(path)
    potential_parent_directory = os.path.abspath(potential_parent_directory)

    return potential_parent_directory in path


def random_string(length: int) -> Text:
    """Returns a random string of given length."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def handle_print_blocking(output: Text) -> None:
    """Handle print blocking (BlockingIOError) by getting the STDOUT lock.

    Args:
        output: Text to be printed to STDOUT.
    """
    # Locking again to obtain STDOUT with a lock.
    with portalocker.Lock(sys.stdout) as lock:
        if sys.platform == "win32":
            # colorama is used to fix a regression where colors can not be printed on
            # windows. https://github.com/RasaHQ/rasa/issues/7053
            from colorama import AnsiToWin32

            lock = AnsiToWin32(lock).stream

        print(output, file=lock, flush=True)
