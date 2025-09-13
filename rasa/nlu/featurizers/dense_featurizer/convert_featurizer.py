# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入标准库模块
import logging  # 日志记录模块，用于记录程序运行状态和调试信息
import os  # 操作系统接口模块，用于文件路径操作
from typing import Any, Dict, List, Optional, Text, Tuple, Type  # 类型注解模块，提供类型提示功能

# 导入深度学习框架
import tensorflow as tf  # TensorFlow 深度学习框架，用于模型推理
from tensorflow.python.eager.wrap_function import WrappedFunction  # TensorFlow 包装函数，用于模型签名
from tqdm import tqdm  # 进度条库，用于显示训练进度
import numpy as np  # 数值计算库，提供多维数组和数学运算功能

# 导入 Rasa 核心模块
from rasa.engine.graph import GraphComponent, ExecutionContext  # 图组件和执行上下文，用于组件管理和执行控制
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
import rasa.shared.utils.io  # 共享工具模块，提供文件读写和序列化功能
import rasa.core.utils  # 核心工具模块，提供核心功能
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer  # 标记化器，用于将文本分割为标记
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer  # 密集特征化器基类，提供密集特征提取的通用接口
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息
from rasa.nlu.constants import (  # NLU 常量导入
    DENSE_FEATURIZABLE_ATTRIBUTES,  # 可密集特征化的属性列表
    TOKENS_NAMES,  # 标记名称常量，定义各种标记的键名
    NUMBER_OF_SUB_TOKENS,  # 子标记数量常量
)
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT  # NLU 常量，定义消息属性的键名
from rasa.exceptions import RasaException  # Rasa 异常类，用于错误处理
import rasa.nlu.utils  # NLU 工具模块，提供 NLU 相关功能
import rasa.utils.train_utils as train_utils  # 训练工具模块，提供训练相关功能

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)

# 原始模型远程位置的 URL，用户可能使用此 URL，但模型已不再托管在此处
ORIGINAL_TF_HUB_MODULE_URL = (
    "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model.tar.gz"
)

# 警告：此 URL 仅用于运行 ConveRT 相关组件的 pytest 测试
# 此 URL 不应被用户使用
RESTRICTED_ACCESS_URL = (
    "https://storage.googleapis.com/continuous-"
    "integration-model-storage/convert_tf2.tar.gz"
)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False  # 注册为消息特征化器组件类型，标记为不可训练组件
)
class ConveRTFeaturizer(DenseFeaturizer, GraphComponent):
    """使用 ConveRT 模型的特征化器。

    该类继承自 DenseFeaturizer 和 GraphComponent，实现了基于 ConveRT 模型的密集特征提取。
    从 TFHub 加载 ConveRT 模型（https://github.com/PolyAI-LDN/polyai-models#convert），
    并为每个消息对象的密集可特征化属性计算句子级别和序列级别的特征表示。
    
    ConveRT 是一个高效的对话表示转换器，专门为对话系统设计，
    能够生成高质量的文本嵌入表示。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """获取此组件运行前必须包含在管道中的组件类型。
        
        该方法定义了组件的依赖关系，确保在特征提取之前文本已经被正确标记化。
        
        Returns:
            必需的组件类型列表，包含标记化器组件
        """
        return [Tokenizer]  # 需要标记化器组件，用于将文本分割为标记

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置参数。
        
        该方法定义了 ConveRTFeaturizer 的所有可配置参数及其默认值。
        配置参数主要控制模型 URL 的设置。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        return {
            **DenseFeaturizer.get_default_config(),  # 继承密集特征化器基类的默认配置
            # 模型文件配置
            "model_url": None,  # 模型 URL 或本地路径，必须由用户指定
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """获取此组件运行所需的额外 Python 依赖项。
        
        该方法定义了组件运行所需的外部包，确保在组件初始化前已安装必要的依赖。
        
        Returns:
            依赖包名称列表，包含 TensorFlow 相关包
        """
        return ["tensorflow_text", "tensorflow_hub"]  # 需要 TensorFlow 文本处理和模型中心包

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """确定此组件可以处理的语言。
        
        该方法返回组件支持的语言列表，用于语言兼容性检查。
        
        Returns:
            支持的语言列表，或 None 表示支持所有语言
        """
        return ["en"]  # 仅支持英语，因为 ConveRT 模型是基于英语训练的

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> ConveRTFeaturizer:
        """创建新的组件实例。
        
        这是一个类方法，用于创建新的 ConveRTFeaturizer 实例。
        通常在组件初始化时调用。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            
        Returns:
            新的 ConveRTFeaturizer 实例
        """
        return cls(name=execution_context.node_name, config=config)

    def __init__(self, name: Text, config: Dict[Text, Any]) -> None:
        """初始化 ConveRTFeaturizer 实例。

        该构造函数初始化 ConveRTFeaturizer 实例，加载 ConveRT 模型，
        并设置模型的各种签名函数用于特征提取。

        Args:
            name: 此特征化器的标识符
            config: 配置参数
        """
        super().__init__(name=name, config=config)  # 调用父类构造函数

        # 获取并处理模型 URL
        model_url = self._config["model_url"]
        self.model_url = (
            model_url
            if rasa.nlu.utils.is_url(model_url)  # 如果是 URL
            else os.path.abspath(model_url)  # 如果是本地路径，转换为绝对路径
        )

        # 加载 TensorFlow Hub 模型
        self.module = train_utils.load_tf_hub_model(self.model_url)

        # 获取模型的各种签名函数
        self.tokenize_signature: WrappedFunction = self._get_signature(
            "tokenize", self.module  # 标记化签名
        )
        self.sequence_encoding_signature: WrappedFunction = self._get_signature(
            "encode_sequence", self.module  # 序列编码签名
        )
        self.sentence_encoding_signature: WrappedFunction = self._get_signature(
            "default", self.module  # 句子编码签名
        )

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """验证组件配置是否正确。
        
        该方法检查配置参数的有效性，确保组件能够正常运行。
        主要验证模型 URL 的有效性。
        
        Args:
            config: 要验证的配置字典
        """
        cls._validate_model_url(config)  # 验证模型 URL

    @staticmethod
    def _validate_model_files_exist(model_directory: Text) -> None:
        """检查模型目录中是否存在必要的模型文件。

        该方法验证本地模型目录是否包含 ConveRT 模型所需的所有文件。
        如果缺少任何必要文件，将抛出异常。

        Args:
            model_directory: 要检查的模型目录路径
        """
        # 定义需要检查的模型文件列表
        files_to_check = [
            os.path.join(model_directory, "saved_model.pb"),  # 保存的模型文件
            os.path.join(model_directory, "variables/variables.index"),  # 变量索引文件
            os.path.join(model_directory, "variables/variables.data-00001-of-00002"),  # 变量数据文件1
            os.path.join(model_directory, "variables/variables.data-00000-of-00002"),  # 变量数据文件2
        ]

        # 检查每个文件是否存在
        for file_path in files_to_check:
            if not os.path.exists(file_path):  # 如果文件不存在
                raise RasaException(
                    f"File {file_path} does not exist. "
                    f"Re-check the files inside the directory {model_directory}. "
                    f"It should contain the following model "
                    f"files - [{', '.join(files_to_check)}]"
                )

    @classmethod
    def _validate_model_url(cls, config: Dict[Text, Any]) -> None:
        """Validates the specified `model_url` parameter.

        The `model_url` parameter cannot be left empty. It can either
        be set to a remote URL where the model is hosted or it can be
        a path to a local directory.

        Args:
            config: a configuration for this graph component
        """
        model_url = config.get("model_url", None)

        if not model_url:
            raise RasaException(
                f"Parameter 'model_url' was not specified in the configuration "
                f"of '{ConveRTFeaturizer.__name__}'. "
                f"It is mandatory to pass a value for this parameter. "
                f"You can either use a community hosted URL of the model "
                f"or if you have a local copy of the model, pass the "
                f"path to the directory containing the model files."
            )

        if model_url == ORIGINAL_TF_HUB_MODULE_URL:
            # Can't use the originally hosted URL
            raise RasaException(
                f"Parameter 'model_url' of "
                f"'{ConveRTFeaturizer.__name__}' was "
                f"set to '{model_url}' which does not contain the model any longer. "
                f"You can either use a community hosted URL or if you have a "
                f"local copy of the model, pass the path to the directory "
                f"containing the model files."
            )

        if model_url == RESTRICTED_ACCESS_URL:
            # Can't use the URL that is reserved for tests only
            raise RasaException(
                f"Parameter 'model_url' of "
                f"'{ConveRTFeaturizer.__name__}' was "
                f"set to '{model_url}' which is strictly reserved for pytests of "
                f"Rasa Open Source only. Due to licensing issues you are "
                f"not allowed to use the model from this URL. "
                f"You can either use a community hosted URL or if you have a "
                f"local copy of the model, pass the path to the directory "
                f"containing the model files."
            )

        if os.path.isfile(model_url):
            # Definitely invalid since the specified path should be a directory
            raise RasaException(
                f"Parameter 'model_url' of "
                f"'{ConveRTFeaturizer.__name__}' was "
                f"set to the path of a file which is invalid. You "
                f"can either use a community hosted URL or if you have a "
                f"local copy of the model, pass the path to the directory "
                f"containing the model files."
            )

        if not rasa.nlu.utils.is_url(model_url) and not os.path.isdir(model_url):
            raise RasaException(
                f"{model_url} is neither a valid remote URL nor a local directory. "
                f"You can either use a community hosted URL or if you have a "
                f"local copy of the model, pass the path to "
                f"the directory containing the model files."
            )

        if os.path.isdir(model_url):
            # Looks like a local directory. Inspect the directory
            # to see if model files exist.
            cls._validate_model_files_exist(model_url)

    @staticmethod
    def _get_signature(signature: Text, module: Any) -> WrappedFunction:
        """Retrieve a signature from a (hopefully loaded) TF model."""
        if not module:
            raise Exception(
                f"{ConveRTFeaturizer.__name__} needs "
                f"a proper loaded tensorflow module when used. "
                f"Make sure to pass a module when training and using the component."
            )

        return module.signatures[signature]

    def _compute_features(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算批量示例的特征。

        该方法计算给定批量消息的句子级别和序列级别特征。
        首先计算句子编码，然后计算序列编码，最后组合成最终特征。

        Args:
            batch_examples: 批量消息示例
            attribute: 要处理的属性，默认为 TEXT

        Returns:
            包含序列特征和句子特征的元组
        """
        # 计算句子级别编码
        sentence_encodings = self._compute_sentence_encodings(batch_examples, attribute)

        # 计算序列级别编码
        (
            sequence_encodings,
            number_of_tokens_in_sentence,
        ) = self._compute_sequence_encodings(batch_examples, attribute)

        # 组合并返回最终特征
        return self._get_features(
            sentence_encodings, sequence_encodings, number_of_tokens_in_sentence
        )

    def _compute_sentence_encodings(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> np.ndarray:
        """计算句子级别编码。

        该方法为批量示例计算句子级别的特征表示。
        使用 ConveRT 模型的句子编码功能生成整个句子的嵌入。

        Args:
            batch_examples: 批量消息示例
            attribute: 要处理的属性，默认为 TEXT

        Returns:
            句子级别编码的 numpy 数组
        """
        # 获取每个示例的指定属性文本
        batch_attribute_text = [ex.get(attribute) for ex in batch_examples]
        # 使用 ConveRT 模型计算句子编码
        sentence_encodings = self._sentence_encoding_of_text(batch_attribute_text)

        # 将编码转换为序列长度为1的格式
        return np.reshape(sentence_encodings, (len(batch_examples), 1, -1))

    def _compute_sequence_encodings(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> Tuple[np.ndarray, List[int]]:
        """计算序列级别编码。

        该方法为批量示例计算序列级别的特征表示。
        首先对文本进行标记化，然后使用 ConveRT 模型计算序列编码，
        最后对齐子标记特征以匹配原始标记。

        Args:
            batch_examples: 批量消息示例
            attribute: 要处理的属性，默认为 TEXT

        Returns:
            包含序列编码和每个句子中标记数量的元组
        """
        # 对每个示例进行标记化
        list_of_tokens = [
            self.tokenize(example, attribute) for example in batch_examples
        ]

        # 计算每个句子中的标记数量
        number_of_tokens_in_sentence = [
            len(sent_tokens) for sent_tokens in list_of_tokens
        ]

        # 将标记连接成文本，确保 ConveRT 返回的嵌入序列长度
        # 与标记长度（包括子标记）匹配
        tokenized_texts = self._tokens_to_text(list_of_tokens)
        # 使用 ConveRT 模型计算序列编码
        token_features = self._sequence_encoding_of_text(tokenized_texts)

        # ConveRT 可能将标记分割为子标记
        # 取子标记向量的平均值作为标记向量
        token_features = train_utils.align_token_features(
            list_of_tokens, token_features
        )

        return token_features, number_of_tokens_in_sentence

    @staticmethod
    def _get_features(
        sentence_encodings: np.ndarray,
        sequence_encodings: np.ndarray,
        number_of_tokens_in_sentence: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the sequence and sentence features."""
        sentence_embeddings = []
        sequence_embeddings = []

        for index in range(len(number_of_tokens_in_sentence)):
            sequence_length = number_of_tokens_in_sentence[index]
            sequence_encoding = sequence_encodings[index][:sequence_length]
            sentence_encoding = sentence_encodings[index]

            sequence_embeddings.append(sequence_encoding)
            sentence_embeddings.append(sentence_encoding)

        return np.array(sequence_embeddings), np.array(sentence_embeddings)

    @staticmethod
    def _tokens_to_text(list_of_tokens: List[List[Token]]) -> List[Text]:
        """Convert list of tokens to text.

        Add a whitespace between two tokens if the end value of the first tokens
        is not the same as the end value of the second token.
        """
        texts = []
        for tokens in list_of_tokens:
            text = ""
            offset = 0
            for token in tokens:
                if offset != token.start:
                    text += " "
                text += token.text

                offset = token.end
            texts.append(text)

        return texts

    def _sentence_encoding_of_text(self, batch: List[Text]) -> np.ndarray:

        return self.sentence_encoding_signature(tf.convert_to_tensor(batch))[
            "default"
        ].numpy()

    def _sequence_encoding_of_text(self, batch: List[Text]) -> np.ndarray:

        return self.sequence_encoding_signature(tf.convert_to_tensor(batch))[
            "sequence_encoding"
        ].numpy()

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """使用 ConveRT 模型对训练数据中的所有消息属性进行特征化。

        该方法遍历所有密集可特征化的属性，对每个属性进行批量特征提取，
        并显示进度条以跟踪处理进度。

        Args:
            training_data: 要进行特征化的训练数据

        Returns:
            特征化后的训练数据
        """
        batch_size = 64  # 批处理大小

        # 遍历所有密集可特征化的属性
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            # 过滤出非空的示例
            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            # 创建进度条
            progress_bar = tqdm(
                range(0, len(non_empty_examples), batch_size),
                desc=attribute.capitalize() + " batches",
            )
            
            # 批量处理示例
            for batch_start_index in progress_bar:
                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )

                # 收集批量示例
                batch_examples = non_empty_examples[batch_start_index:batch_end_index]

                # 计算批量特征
                (
                    batch_sequence_features,
                    batch_sentence_features,
                ) = self._compute_features(batch_examples, attribute)

                # 设置特征到示例中
                self._set_features(
                    batch_examples,
                    batch_sequence_features,
                    batch_sentence_features,
                    attribute,
                )
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Featurize an incoming message with the ConveRT model.

        Args:
            messages: Message to be featurized
        """
        for message in messages:
            for attribute in {TEXT, ACTION_TEXT}:
                if message.get(attribute):
                    sequence_features, sentence_features = self._compute_features(
                        [message], attribute=attribute
                    )

                    self._set_features(
                        [message], sequence_features, sentence_features, attribute
                    )
        return messages

    def _set_features(
        self,
        examples: List[Message],
        sequence_features: np.ndarray,
        sentence_features: np.ndarray,
        attribute: Text,
    ) -> None:
        for index, example in enumerate(examples):
            self.add_features_to_message(
                sequence=sequence_features[index],
                sentence=sentence_features[index],
                message=example,
                attribute=attribute,
            )

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.

        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        reuse the tokens that are already assigned to the message. If individual tokens
        are split up into multiple tokens, add this information to the
        respected tokens.
        """
        tokens_in = message.get(TOKENS_NAMES[attribute])

        tokens_out = []

        for token in tokens_in:
            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token.text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        message.set(TOKENS_NAMES[attribute], tokens_out)
        return tokens_out

    @staticmethod
    def _clean_tokens(tokens: List[bytes]) -> List[Text]:
        """Encode tokens and remove special char added by ConveRT."""
        decoded_tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in decoded_tokens if string]
