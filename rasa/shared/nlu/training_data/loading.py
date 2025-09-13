import json
import logging
import os
import typing
from typing import Optional, Text, Callable, Dict, Any, List

import rasa.shared.utils.io
from rasa.shared.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT,
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_ENTITY_ENTRIES,
    DIALOGFLOW_INTENT,
    DIALOGFLOW_INTENT_EXAMPLES,
    DIALOGFLOW_PACKAGE,
)
from rasa.shared.nlu.training_data.training_data import TrainingData

# =============================================================================
# 训练数据加载模块 - 提供多种格式训练数据的加载和格式识别功能
# =============================================================================

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataReader

logger = logging.getLogger(__name__)

# 不同支持的文件格式及其标识符
WIT = "wit"  # Wit.ai 格式
LUIS = "luis"  # Microsoft LUIS 格式
RASA = "rasa_nlu"  # Rasa NLU JSON 格式
RASA_YAML = "rasa_yml"  # Rasa YAML 格式
UNK = "unk"  # 未知格式
DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}  # Dialogflow 相关格式

_json_format_heuristics: Dict[Text, Callable[[Any, Text], bool]] = {
    # JSON 格式启发式规则字典，用于识别不同的训练数据格式
    WIT: lambda js, fn: "utterances" in js and "luis_schema_version" not in js,
    LUIS: lambda js, fn: "luis_schema_version" in js,
    RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn,
}


def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData":
    """从磁盘加载训练数据。

    如果从磁盘加载并找到多个文件，则合并它们。
    
    Args:
        resource_name: 资源名称（文件或目录路径）
        language: 语言代码，默认为 "en"
        
    Returns:
        合并后的训练数据对象
        
    Raises:
        ValueError: 当文件不存在时
    """
    if not os.path.exists(resource_name):
        raise ValueError(f"文件 '{resource_name}' 不存在。")

    if os.path.isfile(resource_name):
        files = [resource_name]
    else:
        files = rasa.shared.utils.io.list_files(resource_name)

    data_sets = [_load(f, language) for f in files]
    training_data_sets: List[TrainingData] = [ds for ds in data_sets if ds]
    if len(training_data_sets) == 0:
        training_data = TrainingData()
    elif len(training_data_sets) == 1:
        training_data = training_data_sets[0]
    else:
        training_data = training_data_sets[0].merge(*training_data_sets[1:])

    return training_data


def _reader_factory(fformat: Text) -> Optional["TrainingDataReader"]:
    """根据文件格式生成适当的读取器类。
    
    Args:
        fformat: 文件格式标识符
        
    Returns:
        对应的训练数据读取器实例，如果格式不支持则返回 None
    """
    from rasa.shared.nlu.training_data.formats import (
        RasaYAMLReader,
        WitReader,
        LuisReader,
        RasaReader,
        DialogflowReader,
    )

    reader: Optional["TrainingDataReader"] = None
    if fformat == LUIS:
        reader = LuisReader()
    elif fformat == WIT:
        reader = WitReader()
    elif fformat in DIALOGFLOW_RELEVANT:
        reader = DialogflowReader()
    elif fformat == RASA:
        reader = RasaReader()
    elif fformat == RASA_YAML:
        reader = RasaYAMLReader()
    return reader


def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
    """Loads a single training data file from disk."""

    fformat = guess_format(filename)
    if fformat == UNK:
        raise ValueError(f"Unknown data format for file '{filename}'.")

    reader = _reader_factory(fformat)

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def guess_format(filename: Text) -> Text:
    """Applies heuristics to guess the data format of a file.

    Args:
        filename: file whose type should be guessed

    Returns:
        Guessed file format.
    """
    from rasa.shared.nlu.training_data.formats import RasaYAMLReader

    guess = UNK

    if not os.path.isfile(filename):
        return guess

    try:
        content = rasa.shared.utils.io.read_file(filename)
        js = json.loads(content)
    except ValueError:
        if RasaYAMLReader.is_yaml_nlu_file(filename):
            guess = RASA_YAML
    else:
        for file_format, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = file_format
                break

    logger.debug(f"Training data format of '{filename}' is '{guess}'.")

    return guess
