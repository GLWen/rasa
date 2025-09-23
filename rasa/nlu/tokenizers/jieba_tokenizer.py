# 启用类型注解的前向引用功能，允许在类型注解中使用字符串形式的类型
from __future__ import annotations
# 导入文件路径匹配模块
import glob
# 导入日志记录模块
import logging
# 导入操作系统接口模块
import os
# 导入文件操作模块
import shutil
# 导入类型注解相关的类型
from typing import Any, Dict, List, Optional, Text

# 导入图引擎相关模块
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

# 导入分词器基类和Token类
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# 导入消息类
from rasa.shared.nlu.training_data.message import Message
# 导入训练数据类
from rasa.shared.nlu.training_data.training_data import TrainingData

# 创建日志记录器
logger = logging.getLogger(__name__)


# 注册为默认配方中的消息分词器组件，可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=True
)
# 定义Jieba分词器类
class JiebaTokenizer(Tokenizer):
    """Jieba分词器，这是Jieba库的包装器 (https://github.com/fxsjy/jieba)。

    Jieba是一个优秀的中文分词库，支持精确模式、全模式和搜索引擎模式。
    该分词器专门为中文文本设计，能够准确识别中文词汇边界，支持自定义词典。
    """

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """返回支持的语言列表（参见父类的完整文档字符串）。

        返回:
            支持的语言代码列表，目前只支持中文。
        """
        return ["zh"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回默认配置（参见父类的完整文档字符串）。

        定义了Jieba分词器的默认参数设置。

        返回:
            包含默认配置的字典。
        """
        return {
            # 默认不加载自定义词典
            "dictionary_path": None,
            # 标志位：是否对意图进行分词
            "intent_tokenization_flag": False,
            # 意图分词使用的分隔符
            "intent_split_symbol": "_",
            # 用于检测词汇的正则表达式模式
            "token_pattern": None,
            # 前缀分割使用的分隔符
            "prefix_separator_symbol": None,
        }

    def __init__(
        self, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource
    ) -> None:
        """初始化分词器。

        设置分词器的配置、模型存储和资源信息。

        Args:
            config: 分词器配置字典。
            model_storage: 模型存储接口。
            resource: 资源定位器。
        """
        # 调用父类初始化方法
        super().__init__(config)
        # 存储模型存储接口
        self._model_storage = model_storage
        # 存储资源定位器
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> JiebaTokenizer:
        """创建一个新的组件（参见父类的完整文档字符串）。

        实现GraphComponent接口的create方法，用于创建Jieba分词器实例。
        如果配置了自定义词典路径，会加载自定义词典。

        Args:
            config: 组件配置。
            model_storage: 模型存储接口。
            resource: 资源定位器。
            execution_context: 执行上下文。

        Returns:
            创建的Jieba分词器实例。
        """
        # 获取本地文件系统上的词典路径
        dictionary_path = config["dictionary_path"]

        # 如果配置了自定义词典路径，则加载自定义词典
        if dictionary_path is not None:
            cls._load_custom_dictionary(dictionary_path)
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """此组件运行所需的额外Python依赖包。

        返回:
            必需的Python包名称列表。
        """
        return ["jieba"]

    @staticmethod
    def _load_custom_dictionary(path: Text) -> None:
        """加载指定路径中存储的所有自定义词典。

        从指定目录加载所有Jieba用户词典文件，用于提高特定领域的分词准确性。
        更多关于词典文件格式的信息可以在Jieba文档中找到。
        https://github.com/fxsjy/jieba#load-dictionary

        Args:
            path: 包含自定义词典文件的目录路径。
        """
        # 导入jieba模块
        import jieba

        # 使用glob模式匹配获取所有词典文件
        jieba_userdicts = glob.glob(f"{path}/*")
        # 遍历每个词典文件并加载
        for jieba_userdict in jieba_userdicts:
            # 记录加载信息
            logger.info(f"Loading Jieba User Dictionary at {jieba_userdict}")
            # 加载用户词典到jieba
            jieba.load_userdict(jieba_userdict)

    def train(self, training_data: TrainingData) -> Resource:
        """将词典复制到模型存储中。

        训练阶段的主要方法，将自定义词典持久化到模型存储中。

        Args:
            training_data: 训练数据对象。

        Returns:
            资源定位器。
        """
        # 持久化自定义词典
        self.persist()
        return self._resource

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """对传入消息的指定属性文本进行分词。

        使用Jieba库对中文文本进行分词，这是Jieba分词器的核心方法。

        Args:
            message: 包含要分词文本的消息对象。
            attribute: 要分词的属性名称。

        Returns:
            分词后的Token对象列表。
        """
        # 导入jieba模块
        import jieba

        # 获取要分词的文本
        text = message.get(attribute)

        # 使用jieba进行分词，返回(词汇, 起始位置, 结束位置)的生成器
        tokenized = jieba.tokenize(text)
        # 将jieba分词结果转换为Token对象列表
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        # 应用分词模式（如果配置了的话）
        return self._apply_token_pattern(tokens)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> JiebaTokenizer:
        """从模型存储中加载自定义词典。

        推理阶段的主要方法，从模型存储中加载之前保存的自定义词典。

        Args:
            config: 组件配置。
            model_storage: 模型存储接口。
            resource: 资源定位器。
            execution_context: 执行上下文。
            **kwargs: 其他参数。

        Returns:
            加载了自定义词典的Jieba分词器实例。
        """
        # 获取自定义词典路径
        dictionary_path = config["dictionary_path"]

        # 如果配置中指定了自定义词典路径，说明它应该已经被保存到模型存储中
        if dictionary_path is not None:
            try:
                # 从模型存储中读取资源目录
                with model_storage.read_from(resource) as resource_directory:
                    # 加载自定义词典
                    cls._load_custom_dictionary(str(resource_directory))
            except ValueError:
                # 记录调试信息，说明资源不存在
                logger.debug(
                    f"Failed to load {cls.__name__} from model storage. "
                    f"Resource '{resource.name}' doesn't exist."
                )
        return cls(config, model_storage, resource)

    @staticmethod
    def _copy_files_dir_to_dir(input_dir: Text, output_dir: Text) -> None:
        """将输入目录中的所有文件复制到输出目录。

        用于将自定义词典文件从源目录复制到目标目录。

        Args:
            input_dir: 源目录路径。
            output_dir: 目标目录路径。
        """
        # 确保目标路径存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取源目录中的所有文件
        target_file_list = glob.glob(f"{input_dir}/*")
        # 复制每个文件到目标目录
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self) -> None:
        """持久化自定义词典。

        将自定义词典文件保存到模型存储中，以便在推理时使用。
        """
        # 获取自定义词典路径
        dictionary_path = self._config["dictionary_path"]
        # 如果配置了自定义词典路径，则进行持久化
        if dictionary_path is not None:
            # 写入到模型存储
            with self._model_storage.write_to(self._resource) as resource_directory:
                # 复制词典文件到资源目录
                self._copy_files_dir_to_dir(dictionary_path, str(resource_directory))
