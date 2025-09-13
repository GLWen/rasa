import os
import shutil
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Text, Optional, Union, List, Callable, Set, Iterable

# =============================================================================
# 文件扩展名常量定义
# =============================================================================

YAML_FILE_EXTENSIONS = [".yml", ".yaml"]  # 支持的 YAML 文件扩展名
JSON_FILE_EXTENSIONS = [".json"]  # 支持的 JSON 文件扩展名
TRAINING_DATA_EXTENSIONS = set(JSON_FILE_EXTENSIONS + YAML_FILE_EXTENSIONS)  # 所有支持的训练数据文件扩展名


# =============================================================================
# 文件类型检测函数
# =============================================================================

def yaml_file_extension() -> Text:
    """返回 YAML 文件扩展名。
    
    Returns:
        默认的 YAML 文件扩展名（.yml）
    """
    return YAML_FILE_EXTENSIONS[0]


def is_likely_yaml_file(file_path: Union[Text, Path]) -> bool:
    """检查文件是否可能包含 YAML 格式的数据。
    
    基于文件扩展名进行快速检测，不进行内容验证。

    Arguments:
        file_path: 要检查的文件路径

    Returns:
        如果文件可能包含 YAML 格式数据则返回 `True`，否则返回 `False`
    """
    return Path(file_path).suffix in set(YAML_FILE_EXTENSIONS)


def is_likely_json_file(file_path: Text) -> bool:
    """检查文件是否可能包含 JSON 格式的数据。
    
    基于文件扩展名进行快速检测，不进行内容验证。

    Arguments:
        file_path: 要检查的文件路径

    Returns:
        如果文件可能包含 JSON 格式数据则返回 `True`，否则返回 `False`
    """
    return Path(file_path).suffix in set(JSON_FILE_EXTENSIONS)


# =============================================================================
# 训练数据目录收集函数
# =============================================================================

def get_core_directory(paths: Optional[Union[Text, List[Text]]]) -> Text:
    """递归收集指定路径下的所有 Core 训练文件。
    
    将找到的所有 Core 训练文件复制到一个临时目录中，避免文件名冲突。

    Args:
        paths: 训练文件或包含训练文件的文件夹路径列表

    Returns:
        包含所有找到的 Core 训练文件的临时目录路径
    """
    from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
        YAMLStoryReader,
    )

    # 使用故事文件检测器收集 Core 训练文件
    core_files = get_data_files(paths, YAMLStoryReader.is_stories_file)
    return _copy_files_to_new_dir(core_files)


def get_nlu_directory(paths: Optional[Union[Text, List[Text]]]) -> Text:
    """递归收集指定路径下的所有 NLU 训练文件。
    
    将找到的所有 NLU 训练文件复制到一个临时目录中，避免文件名冲突。

    Args:
        paths: 训练文件或包含训练文件的文件夹路径列表

    Returns:
        包含所有找到的 NLU 训练文件的临时目录路径
    """
    # 使用 NLU 文件检测器收集 NLU 训练文件
    nlu_files = get_data_files(paths, is_nlu_file)
    return _copy_files_to_new_dir(nlu_files)


# =============================================================================
# 通用数据文件收集函数
# =============================================================================

def get_data_files(
    paths: Optional[Union[Text, List[Text]]], filter_predicate: Callable[[Text], bool]
) -> List[Text]:
    """递归收集指定路径下的所有训练文件。
    
    支持单个文件、文件列表或目录路径。使用提供的过滤谓词来确定哪些文件应该被包含。

    Args:
        paths: 训练文件或包含训练文件的文件夹路径列表
        filter_predicate: 用于过滤路径的属性函数，例如 `is_nlu_file`

    Returns:
        训练数据文件的路径列表（已排序）
    """
    data_files = set()

    # 处理路径参数：None 转为空列表，字符串转为单元素列表
    if paths is None:
        paths = []
    elif isinstance(paths, str):
        paths = [paths]

    # 遍历所有路径，去重处理
    for path in set(paths):
        if not path:
            continue

        # 如果是有效文件类型，直接检查过滤条件
        if is_valid_filetype(path):
            if filter_predicate(path):
                data_files.add(os.path.abspath(path))
        else:
            # 如果是目录，递归查找其中的文件
            new_data_files = _find_data_files_in_directory(path, filter_predicate)
            data_files.update(new_data_files)

    return sorted(data_files)  # 返回排序后的文件列表，确保结果一致性


# =============================================================================
# 目录遍历辅助函数
# =============================================================================

def _find_data_files_in_directory(
    directory: Text, filter_property: Callable[[Text], bool]
) -> Set[Text]:
    """在指定目录中递归查找符合过滤条件的数据文件。
    
    使用 os.walk 遍历目录树，对每个文件应用过滤条件。

    Args:
        directory: 要搜索的目录路径
        filter_property: 用于过滤文件的属性函数

    Returns:
        符合过滤条件的文件路径集合
    """
    filtered_files = set()

    # 递归遍历目录，支持符号链接
    for root, _, files in os.walk(directory, followlinks=True):
        # 对文件进行排序，确保训练结果的可重复性
        for f in sorted(files):
            full_path = os.path.join(root, f)

            # 跳过不支持的文件类型
            if not is_valid_filetype(full_path):
                continue

            # 应用过滤条件
            if filter_property(full_path):
                filtered_files.add(full_path)

    return filtered_files


# =============================================================================
# 文件类型验证函数
# =============================================================================

def is_valid_filetype(path: Union[Path, Text]) -> bool:
    """检查给定文件是否具有支持的扩展名。
    
    验证文件是否存在且扩展名在支持的训练数据扩展名列表中。

    Args:
        path: 源文件的路径

    Returns:
        如果文件具有支持的扩展名则返回 `True`，否则返回 `False`
    """
    return Path(path).is_file() and Path(path).suffix in TRAINING_DATA_EXTENSIONS


def is_nlu_file(file_path: Text) -> bool:
    """检查文件是否为 Rasa 兼容的 NLU 文件。
    
    通过 NLU 加载器进行内容格式验证，比简单的扩展名检查更准确。

    Args:
        file_path: 要检查的文件路径

    Returns:
        如果是 NLU 文件则返回 `True`，否则返回 `False`
    """
    from rasa.shared.nlu.training_data import loading as nlu_loading

    # 使用 NLU 加载器猜测文件格式，如果不是未知格式则认为是 NLU 文件
    return nlu_loading.guess_format(file_path) != nlu_loading.UNK


def is_config_file(file_path: Text) -> bool:
    """检查给定文件路径是否为 Rasa 配置文件。
    
    基于文件名进行精确匹配，支持 .yml 和 .yaml 扩展名。

    Args:
        file_path: 要检查的文件路径

    Returns:
        如果是 Rasa 配置文件则返回 `True`，否则返回 `False`
    """
    file_name = os.path.basename(file_path)

    return file_name in ["config.yml", "config.yaml"]


# =============================================================================
# 文件复制和临时目录管理函数
# =============================================================================

def _copy_files_to_new_dir(files: Iterable[Text]) -> Text:
    """将文件复制到新的临时目录中。
    
    使用 UUID 前缀确保文件名唯一性，避免文件覆盖冲突。

    Args:
        files: 要复制的文件路径迭代器

    Returns:
        临时目录的路径
    """
    directory = tempfile.mkdtemp()  # 创建临时目录
    for f in files:
        # 使用 UUID 前缀确保文件不会相互覆盖
        unique_prefix = uuid.uuid4().hex
        unique_file_name = unique_prefix + "_" + os.path.basename(f)
        # 复制文件并保留元数据
        shutil.copy2(f, os.path.join(directory, unique_file_name))

    return directory


# =============================================================================
# 训练类型枚举
# =============================================================================

class TrainingType(Enum):
    """训练类型枚举类，明确定义所有存在的训练类型。"""

    NLU = 1  # 仅 NLU 训练
    CORE = 2  # 仅 Core 训练
    BOTH = 3  # 同时进行 NLU 和 Core 训练
    END_TO_END = 4  # 端到端训练

    @property
    def model_type(self) -> Text:
        """返回此训练类型产生的模型类型。
        
        Returns:
            模型类型字符串：'nlu'、'core' 或 'rasa'
        """
        if self == TrainingType.NLU:
            return "nlu"
        if self == TrainingType.CORE:
            return "core"
        return "rasa"  # 对于 BOTH 和 END_TO_END 类型
