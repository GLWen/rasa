# =============================================================================
# Rasa NLU Count Vectors Featurizer 计数向量特征化器模块
# 本模块实现了基于 sklearn CountVectorizer 的稀疏特征提取器，
# 用于将文本转换为词频向量特征，支持词级别和字符级别的特征提取
# =============================================================================

# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入标准库模块
import logging  # 日志记录模块，用于记录程序运行状态和错误信息
import re  # 正则表达式模块，用于文本模式匹配和替换
from typing import Any, Dict, List, Optional, Text, Tuple, Set, Type, Union  # 类型注解模块，提供类型提示功能

# 导入科学计算库
import numpy as np  # 数值计算库，提供多维数组和数学运算功能
import scipy.sparse  # 稀疏矩阵库，用于高效存储和处理稀疏数据
from sklearn.feature_extraction.text import CountVectorizer  # sklearn 计数向量化器，将文本转换为词频矩阵

# 导入 Rasa 核心模块
import rasa.shared.utils.io  # 共享工具模块，提供文件读写和序列化功能
from rasa.engine.graph import GraphComponent, ExecutionContext  # 图组件和执行上下文，用于组件管理和执行控制
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.nlu.constants import (  # NLU 常量导入
    TOKENS_NAMES,  # 标记名称常量，定义各种标记的键名
    MESSAGE_ATTRIBUTES,  # 消息属性常量，定义消息的各种属性
    DENSE_FEATURIZABLE_ATTRIBUTES,  # 可密集特征化的属性，支持密集特征提取的属性列表
)
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer  # 稀疏特征化器基类，提供稀疏特征提取的通用接口
from rasa.nlu.tokenizers.tokenizer import Tokenizer  # 标记化器，用于将文本分割为标记
from rasa.nlu.utils.spacy_utils import SpacyModel  # Spacy 模型，提供自然语言处理功能
from rasa.shared.constants import DOCS_URL_COMPONENTS  # 文档URL常量，指向组件文档的链接
from rasa.shared.exceptions import RasaException, FileIOException  # Rasa 异常类，用于错误处理
from rasa.shared.nlu.constants import TEXT, INTENT, INTENT_RESPONSE_KEY, ACTION_NAME  # NLU 常量，定义消息属性的键名
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本

# 缓冲区槽前缀常量，用于标识缓冲区中的词汇表项
BUFFER_SLOTS_PREFIX = "buf_"

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True  # 注册为消息特征化器组件类型，标记为可训练组件
)
class CountVectorsFeaturizer(SparseFeaturizer, GraphComponent):
    """基于 sklearn 的 `CountVectorizer` 创建标记计数特征序列的稀疏特征化器。

    该类继承自 SparseFeaturizer 和 GraphComponent，实现了基于词频的稀疏特征提取。
    所有仅由数字组成的标记（例如 123 和 99，但不包括 ab12d）将由单个特征表示。
    
    支持两种分析模式：
    1. 词级别分析（analyzer='word'）：基于词汇进行特征提取
    2. 字符级别分析（analyzer='char_wb'）：基于字符n-gram进行特征提取
    
    将 `analyzer` 设置为 'char_wb' 以使用子词语义哈希的思想，
    参考论文：https://arxiv.org/abs/1810.07150
    """

    OOV_words: List[Text]  # 词汇表外词汇列表，存储训练时未见过但在预测时可能出现的词汇

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
        
        该方法定义了 CountVectorsFeaturizer 的所有可配置参数及其默认值。
        配置参数主要来自 sklearn 的 CountVectorizer，并添加了 Rasa 特有的参数。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        return {
            **SparseFeaturizer.get_default_config(),  # 继承稀疏特征化器基类的默认配置
            # 词汇表配置
            "use_shared_vocab": False,  # 是否使用共享词汇表，False表示每个属性使用独立词汇表
            
            # sklearn CountVectorizer 核心参数
            "analyzer": "word",  # 分析器类型：'word'（词级别）或 'char_wb'（字符级别，词边界内）
            "strip_accents": None,  # 去除重音符号的方式：'ascii'、'unicode' 或 None（不去除）
            "stop_words": None,  # 停用词列表：'english'、自定义列表或 None（不使用停用词）
            
            # 文档频率过滤参数
            "min_df": 1,  # 最小文档频率：词汇必须至少在指定数量的文档中出现才被包含
            "max_df": 1.0,  # 最大文档频率：词汇在超过指定比例的文档中出现时被忽略
            
            # N-gram 范围配置
            "min_ngram": 1,  # 最小 n-gram 长度，默认为1（单个词）
            "max_ngram": 1,  # 最大 n-gram 长度，默认为1（单个词）
            
            # 词汇表大小限制
            "max_features": None,  # 最大特征数量，None表示不限制
            
            # 文本预处理参数
            "lowercase": True,  # 是否将所有字符转换为小写
            
            # 词汇表外（OOV）词汇处理
            "OOV_token": None,  # OOV 标记，用于替换未见过词汇的占位符
            "OOV_words": [],  # 预定义的 OOV 词汇列表
            
            # 词元化配置
            "use_lemma": True,  # 是否使用词的词元形式进行计数（需要词元化器支持）
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """获取此组件运行所需的额外 Python 依赖项。
        
        该方法定义了组件运行所需的外部包，确保在组件初始化前已安装必要的依赖。
        
        Returns:
            依赖包名称列表，包含 sklearn 包
        """
        return ["sklearn"]  # 需要 sklearn 包，用于 CountVectorizer 功能

    def _load_count_vect_params(self) -> None:
        """从配置中加载 CountVectorizer 相关参数到实例变量。
        
        该方法将配置字典中的参数提取到实例变量中，便于后续使用。
        这些参数将用于初始化 sklearn 的 CountVectorizer 实例。
        """
        # 词汇表配置参数
        self.use_shared_vocab = self._config["use_shared_vocab"]  # 是否使用共享词汇表，影响不同属性是否共享同一个词汇表

        # 分析器配置参数
        self.analyzer = self._config["analyzer"]  # 分析器类型，决定是基于词还是字符进行特征提取

        # 文本预处理参数
        self.strip_accents = self._config["strip_accents"]  # 去除重音符号的方式，用于文本标准化
        self.stop_words = self._config["stop_words"]  # 停用词列表，用于过滤常见但无意义的词汇
        self.lowercase = self._config["lowercase"]  # 是否转换为小写，用于文本标准化

        # 文档频率过滤参数
        self.min_df = self._config["min_df"]  # 最小文档频率，词汇必须至少在此数量的文档中出现
        self.max_df = self._config["max_df"]  # 最大文档频率，词汇超过此频率时被忽略

        # N-gram 范围参数
        self.min_ngram = self._config["min_ngram"]  # 最小 n-gram 长度，定义特征提取的最小单位
        self.max_ngram = self._config["max_ngram"]  # 最大 n-gram 长度，定义特征提取的最大单位

        # 词汇表大小限制参数
        self.max_features = self._config["max_features"]  # 最大特征数量，限制词汇表的大小

        # 词元化参数
        self.use_lemma = self._config["use_lemma"]  # 是否使用词元形式，影响词汇的标准化程度

    def _load_vocabulary_params(self) -> Tuple[Text, List[Text]]:
        """加载词汇表外（OOV）词汇相关参数。
        
        该方法处理 OOV 标记和 OOV 词汇的配置，确保配置的一致性和正确性。
        如果启用了小写转换，会将 OOV 相关参数也转换为小写以保持一致性。
        
        Returns:
            包含 OOV_token 和 OOV_words 的元组，用于后续的词汇表外词汇处理
        """
        OOV_token = self._config["OOV_token"]  # 获取 OOV 标记，用于替换未见过词汇的占位符

        OOV_words = self._config["OOV_words"]  # 获取预定义的 OOV 词汇列表
        if OOV_words and not OOV_token:  # 如果提供了 OOV 词汇但没有提供 OOV 标记
            logger.error(
                "The list OOV_words={} was given, but "
                "OOV_token was not. OOV words are ignored."
                "".format(OOV_words)
            )
            self.OOV_words = []  # 清空 OOV 词汇列表，因为缺少 OOV 标记无法使用

        if self.lowercase and OOV_token:  # 如果启用了小写转换且存在 OOV 标记
            # 将 OOV 标记转换为小写以保持一致性
            OOV_token = OOV_token.lower()
            if OOV_words:  # 如果存在 OOV 词汇列表
                OOV_words = [w.lower() for w in OOV_words]  # 将所有 OOV 词汇转换为小写

        return OOV_token, OOV_words

    def _get_attribute_vocabulary(self, attribute: Text) -> Optional[Dict[Text, int]]:
        """从指定属性的计数向量化器获取训练好的词汇表。
        
        该方法尝试从已训练的向量化器中提取词汇表，用于后续的 OOV 词汇处理。
        如果向量化器未训练或不存在，则返回 None。
        
        Args:
            attribute: 消息属性名称（如 TEXT、INTENT 等）
            
        Returns:
            词汇表字典（词汇到索引的映射）或 None（如果未训练）
        """
        try:
            return self.vectorizers[attribute].vocabulary_  # 返回训练好的词汇表字典
        except (AttributeError, TypeError, KeyError):  # 捕获可能的异常（向量化器不存在、未训练等）
            return None

    def _check_analyzer(self) -> None:
        """检查分析器配置并发出相应的警告信息。
        
        当分析器设置为字符级别时，某些参数（如 OOV 标记、停用词）可能被忽略。
        该方法会检查配置的一致性并发出警告，帮助用户了解哪些参数可能无效。
        """
        if self.analyzer != "word":  # 如果分析器不是词级别（即字符级别）
            if self.OOV_token is not None:  # 如果设置了 OOV 标记
                logger.warning(
                    "Analyzer is set to character, "
                    "provided OOV word token will be ignored."
                )
            if self.stop_words is not None:  # 如果设置了停用词
                logger.warning(
                    "Analyzer is set to character, "
                    "provided stop words will be ignored."
                )
            if self.max_ngram == 1:  # 如果最大 n-gram 为 1
                logger.warning(
                    "Analyzer is set to character, "
                    "but max n-gram is set to 1. "
                    "It means that the vocabulary will "
                    "contain single letters only."
                )

    @staticmethod
    def _attributes_for(analyzer: Text) -> List[Text]:
        """根据分析器类型确定应该进行特征化的属性列表。
        
        该方法根据分析器的类型（词级别或字符级别）返回相应的属性列表。
        词级别分析器可以处理所有消息属性，而字符级别分析器只能处理密集特征化的属性。
        
        Args:
            analyzer: 分析器类型（'word' 或 'char_wb'）
            
        Returns:
            应该进行特征化的属性列表
        """
        # 意图属性只能由词级别的计数向量化器进行特征化
        return (
            MESSAGE_ATTRIBUTES if analyzer == "word" else DENSE_FEATURIZABLE_ATTRIBUTES
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None,
        oov_token: Optional[Text] = None,
        oov_words: Optional[List[Text]] = None,
    ) -> None:
        """使用 sklearn 框架构造新的计数向量化器。
        
        该构造函数初始化 CountVectorsFeaturizer 实例，加载配置参数，
        设置词汇表外词汇处理，并准备向量化器实例。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文，包含节点名称和执行模式信息
            vectorizers: 可选的预训练向量化器字典
            oov_token: 可选的 OOV 标记
            oov_words: 可选的 OOV 词汇列表
        """
        super().__init__(execution_context.node_name, config)  # 调用父类构造函数

        # 存储模型存储和资源管理对象
        self._model_storage = model_storage
        self._resource = resource

        # 加载 sklearn CountVectorizer 相关参数
        self._load_count_vect_params()

        # 处理词汇表外（OOV）词汇
        if oov_token and oov_words:  # 如果提供了 OOV 参数
            self.OOV_token = oov_token
            self.OOV_words = oov_words
        else:  # 否则从配置中加载
            self.OOV_token, self.OOV_words = self._load_vocabulary_params()

        # 检查分析器配置并发出警告（某些参数可能被忽略）
        self._check_analyzer()

        # 设置应该进行特征化的属性列表
        self._attributes = self._attributes_for(self.analyzer)

        # 声明 CountVectorizer 实例字典
        self.vectorizers = vectorizers or {}

        # 设置微调模式标志
        self.finetune_mode = execution_context.is_finetuning

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CountVectorsFeaturizer:
        """创建新的未训练组件实例。
        
        这是一个类方法，用于创建新的 CountVectorsFeaturizer 实例。
        通常在训练开始时调用，创建一个全新的、未训练的组件。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            
        Returns:
            新的未训练的 CountVectorsFeaturizer 实例
        """
        return cls(config, model_storage, resource, execution_context)

    def _get_message_tokens_by_attribute(
        self, message: "Message", attribute: Text
    ) -> List[Text]:
        """从消息的指定属性中获取文本标记。
        
        该方法从消息对象中提取指定属性的标记，并根据配置决定是否使用词元形式。
        
        Args:
            message: 消息对象
            attribute: 属性名称（如 TEXT、INTENT 等）
            
        Returns:
            标记文本列表，如果属性不存在则返回空列表
        """
        if message.get(TOKENS_NAMES[attribute]):  # 如果属性存在标记
            return [
                t.lemma if self.use_lemma else t.text  # 根据配置选择词元或原始文本
                for t in message.get(TOKENS_NAMES[attribute])
            ]
        else:
            return []  # 属性不存在时返回空列表

    def _process_tokens(self, tokens: List[Text], attribute: Text = TEXT) -> List[Text]:
        """对标记应用处理和清理步骤。
        
        该方法对输入的标记进行预处理，包括数字替换、小写转换等。
        对于意图、动作名称等属性，不进行任何处理，保持原始形式。
        
        Args:
            tokens: 输入标记列表
            attribute: 属性名称，默认为 TEXT
            
        Returns:
            处理后的标记列表
        """
        if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
            # 对意图属性不进行任何处理，将它们视为完整的标签
            return tokens

        # 将所有数字替换为 NUMBER 标记，用于数字标准化
        tokens = [re.sub(r"\b[0-9]+\b", "__NUMBER__", text) for text in tokens]

        # 如果需要，转换为小写
        if self.lowercase:
            tokens = [text.lower() for text in tokens]

        return tokens

    def _replace_with_oov_token(
        self, tokens: List[Text], attribute: Text
    ) -> List[Text]:
        """将词汇表外（OOV）词汇替换为 OOV 标记。
        
        该方法处理训练时未见过但在预测时可能出现的词汇，将它们替换为预定义的 OOV 标记。
        这有助于提高模型对未知词汇的鲁棒性。
        
        Args:
            tokens: 输入标记列表
            attribute: 属性名称
            
        Returns:
            替换 OOV 词汇后的标记列表
        """
        if self.OOV_token and self.analyzer == "word":  # 如果设置了 OOV 标记且使用词级别分析器
            attribute_vocab = self._get_attribute_vocabulary(attribute)
            if attribute_vocab is not None and self.OOV_token in attribute_vocab:
                # 向量化器已训练，进行预测时的处理
                attribute_vocabulary_tokens = set(attribute_vocab.keys())
                tokens = [
                    t if t in attribute_vocabulary_tokens else self.OOV_token
                    for t in tokens
                ]
            elif self.OOV_words:
                # 向量化器未训练，进行训练时的处理
                tokens = [self.OOV_token if t in self.OOV_words else t for t in tokens]

        return tokens

    def _get_processed_message_tokens_by_attribute(
        self, message: Message, attribute: Text = TEXT
    ) -> List[Text]:
        """获取消息指定属性的处理后文本标记。
        
        该方法从消息中提取指定属性的标记，并应用完整的预处理流程，
        包括标记提取、文本处理、OOV 词汇替换等步骤。
        
        Args:
            message: 消息对象
            attribute: 属性名称，默认为 TEXT
            
        Returns:
            完全处理后的标记列表
        """
        if message.get(attribute) is None:
            # 如果属性不存在，返回空列表，因为 sklearn countvectorizer 不喜欢 None 对象
            return []

        tokens = self._get_message_tokens_by_attribute(message, attribute)  # 获取原始标记
        tokens = self._process_tokens(tokens, attribute)  # 应用文本处理
        tokens = self._replace_with_oov_token(tokens, attribute)  # 替换 OOV 词汇

        return tokens

    # noinspection PyPep8Naming
    def _check_OOV_present(self, all_tokens: List[List[Text]], attribute: Text) -> None:
        """Check if an OOV word is present."""
        if not self.OOV_token or self.OOV_words or not all_tokens:
            return

        for tokens in all_tokens:
            for text in tokens:
                if self.OOV_token in text or (
                    self.lowercase and self.OOV_token in text.lower()
                ):
                    return

        if any(text for tokens in all_tokens for text in tokens):
            training_data_type = "NLU" if attribute == TEXT else "ResponseSelector"

            # if there is some text in tokens, warn if there is no oov token
            rasa.shared.utils.io.raise_warning(
                f"The out of vocabulary token '{self.OOV_token}' was configured, but "
                f"could not be found in any one of the {training_data_type} "
                f"training examples. All unseen words will be "
                f"ignored during prediction.",
                docs=DOCS_URL_COMPONENTS + "#countvectorsfeaturizer",
            )

    def _get_all_attributes_processed_tokens(
        self, training_data: TrainingData
    ) -> Dict[Text, List[List[Text]]]:
        """Get processed text for all attributes of examples in training data."""
        processed_attribute_tokens = {}
        for attribute in self._attributes:
            all_tokens = [
                self._get_processed_message_tokens_by_attribute(example, attribute)
                for example in training_data.training_examples
            ]
            if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                # check for oov tokens only in text based attributes
                self._check_OOV_present(all_tokens, attribute)
            processed_attribute_tokens[attribute] = all_tokens

        return processed_attribute_tokens

    @staticmethod
    def _convert_attribute_tokens_to_texts(
        attribute_tokens: Dict[Text, List[List[Text]]]
    ) -> Dict[Text, List[Text]]:
        attribute_texts = {}

        for attribute in attribute_tokens.keys():
            list_of_tokens = attribute_tokens[attribute]
            attribute_texts[attribute] = [" ".join(tokens) for tokens in list_of_tokens]

        return attribute_texts

    def _update_vectorizer_vocabulary(
        self, attribute: Text, new_vocabulary: Set[Text]
    ) -> None:
        """Updates the existing vocabulary of the vectorizer with new unseen words.

        Args:
            attribute: Message attribute for which vocabulary should be updated.
            new_vocabulary: Set of words to expand the vocabulary with if they are
                unseen.
        """
        existing_vocabulary: Dict[Text, int] = self.vectorizers[attribute].vocabulary
        self._merge_new_vocabulary_tokens(existing_vocabulary, new_vocabulary)
        self._set_vocabulary(attribute, existing_vocabulary)

    def _merge_new_vocabulary_tokens(
        self, existing_vocabulary: Dict[Text, int], vocabulary: Set[Text]
    ) -> None:
        """Merges new vocabulary tokens with the existing vocabulary.

        New vocabulary items should always be added to the end of the existing
        vocabulary and the order of the existing vocabulary should not be disturbed.

        Args:
            existing_vocabulary: existing vocabulary
            vocabulary: set of new tokens

        Raises:
            RasaException: if `use_shared_vocab` is set to True and there are new
                           vocabulary items added during incremental training.
        """
        for token in vocabulary:
            if token not in existing_vocabulary:
                if self.use_shared_vocab:
                    raise RasaException(
                        "Using a shared vocabulary in `CountVectorsFeaturizer` is not "
                        "supported during incremental training since it requires "
                        "dynamically adjusting layers that correspond to label "
                        f"attributes such as {INTENT_RESPONSE_KEY}, {INTENT}, etc. "
                        "This is currently not possible. In order to avoid this "
                        "exception we suggest to set `use_shared_vocab=False` or train"
                        " from scratch."
                    )
                existing_vocabulary[token] = len(existing_vocabulary)

    def _set_vocabulary(
        self, attribute: Text, original_vocabulary: Dict[Text, int]
    ) -> None:
        """Sets the vocabulary of the vectorizer of attribute.

        Args:
            attribute: Message attribute for which vocabulary should be set
            original_vocabulary: Vocabulary for the attribute to be set.
        """
        self.vectorizers[attribute].vocabulary_ = original_vocabulary
        self.vectorizers[attribute]._validate_vocabulary()

    @staticmethod
    def _construct_vocabulary_from_texts(
        vectorizer: CountVectorizer, texts: List[Text]
    ) -> Set:
        """Applies vectorizer's preprocessor on texts to get the vocabulary from texts.

        Args:
            vectorizer: Sklearn's count vectorizer which has been pre-configured.
            texts: Examples from which the vocabulary should be constructed

        Returns:
            Unique vocabulary words extracted.
        """
        analyzer = vectorizer.build_analyzer()
        vocabulary_words = set()
        for example in texts:
            example_vocabulary: List[Text] = analyzer(example)
            vocabulary_words.update(example_vocabulary)
        return vocabulary_words

    @staticmethod
    def _attribute_texts_is_non_empty(attribute_texts: List[Text]) -> bool:
        return any(attribute_texts)

    def _train_with_shared_vocab(self, attribute_texts: Dict[Text, List[Text]]) -> None:
        """Constructs the vectorizers and train them with a shared vocab."""
        combined_cleaned_texts = []
        for attribute in self._attributes:
            combined_cleaned_texts += attribute_texts[attribute]

        # To train a shared vocabulary, we use TEXT as the
        # attribute for which a combined vocabulary is built.
        if not self.finetune_mode:
            self.vectorizers = self._create_shared_vocab_vectorizers(
                {
                    "strip_accents": self.strip_accents,
                    "lowercase": self.lowercase,
                    "stop_words": self.stop_words,
                    "min_ngram": self.min_ngram,
                    "max_ngram": self.max_ngram,
                    "max_df": self.max_df,
                    "min_df": self.min_df,
                    "max_features": self.max_features,
                    "analyzer": self.analyzer,
                }
            )
            self._fit_vectorizer_from_scratch(TEXT, combined_cleaned_texts)
        else:
            self._fit_loaded_vectorizer(TEXT, combined_cleaned_texts)
        self._log_vocabulary_stats(TEXT)

    def _train_with_independent_vocab(
        self, attribute_texts: Dict[Text, List[Text]]
    ) -> None:
        """Constructs the vectorizers and train them with an independent vocab."""
        if not self.finetune_mode:
            self.vectorizers = self._create_independent_vocab_vectorizers(
                {
                    "strip_accents": self.strip_accents,
                    "lowercase": self.lowercase,
                    "stop_words": self.stop_words,
                    "min_ngram": self.min_ngram,
                    "max_ngram": self.max_ngram,
                    "max_df": self.max_df,
                    "min_df": self.min_df,
                    "max_features": self.max_features,
                    "analyzer": self.analyzer,
                }
            )
        for attribute in self._attributes:
            if self._attribute_texts_is_non_empty(attribute_texts[attribute]):
                if not self.finetune_mode:
                    self._fit_vectorizer_from_scratch(
                        attribute, attribute_texts[attribute]
                    )
                else:
                    self._fit_loaded_vectorizer(attribute, attribute_texts[attribute])

                self._log_vocabulary_stats(attribute)
            else:
                logger.debug(
                    f"No text provided for {attribute} attribute in any messages of "
                    f"training data. Skipping training a CountVectorizer for it."
                )

    def _log_vocabulary_stats(self, attribute: Text) -> None:
        """Logs number of vocabulary items that were created for a specified attribute.

        Args:
            attribute: Message attribute for which vocabulary stats are logged.
        """
        if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            vocabulary_size = len(self.vectorizers[attribute].vocabulary_)
            logger.info(
                f"{vocabulary_size} vocabulary items "
                f"were created for {attribute} attribute."
            )

    def _fit_loaded_vectorizer(
        self, attribute: Text, attribute_texts: List[Text]
    ) -> None:
        """Fits training texts to a previously trained count vectorizer.

        We do not use the `.fit()` method because the new unseen
        words should occupy the buffer slots of the vocabulary.

        Args:
            attribute: Message attribute for which the vectorizer is to be trained.
            attribute_texts: Training texts for the attribute
        """
        # Get vocabulary words by the preprocessor
        new_vocabulary = self._construct_vocabulary_from_texts(
            self.vectorizers[attribute], attribute_texts
        )
        # update the vocabulary of vectorizer with new vocabulary
        self._update_vectorizer_vocabulary(attribute, new_vocabulary)

    def _fit_vectorizer_from_scratch(
        self, attribute: Text, attribute_texts: List[Text]
    ) -> None:
        """Fits training texts to an untrained count vectorizer.

        Args:
            attribute: Message attribute for which the vectorizer is to be trained.
            attribute_texts: Training texts for the attribute
        """
        try:
            self.vectorizers[attribute].fit(attribute_texts)
        except ValueError:
            logger.warning(
                f"Unable to train CountVectorizer for message "
                f"attribute {attribute} since the call to sklearn's "
                f"`.fit()` method failed. Leaving an untrained "
                f"CountVectorizer for it."
            )

    def _create_features(
        self, attribute: Text, all_tokens: List[List[Text]]
    ) -> Tuple[
        List[Optional[scipy.sparse.spmatrix]], List[Optional[scipy.sparse.spmatrix]]
    ]:
        if not self.vectorizers.get(attribute):
            return [None], [None]

        sequence_features: List[Optional[scipy.sparse.spmatrix]] = []
        sentence_features: List[Optional[scipy.sparse.spmatrix]] = []

        for i, tokens in enumerate(all_tokens):
            if not tokens:
                # nothing to featurize
                sequence_features.append(None)
                sentence_features.append(None)
                continue

            # vectorizer.transform returns a sparse matrix of size
            # [n_samples, n_features]
            # set input to list of tokens if sequence should be returned
            # otherwise join all tokens to a single string and pass that as a list
            if not tokens:
                # attribute is not set (e.g. response not present)
                sequence_features.append(None)
                sentence_features.append(None)
                continue

            seq_vec = self.vectorizers[attribute].transform(tokens)
            seq_vec.sort_indices()

            sequence_features.append(seq_vec.tocoo())

            if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                tokens_text = [" ".join(tokens)]
                sentence_vec = self.vectorizers[attribute].transform(tokens_text)
                sentence_vec.sort_indices()

                sentence_features.append(sentence_vec.tocoo())
            else:
                sentence_features.append(None)

        return sequence_features, sentence_features

    def _get_featurized_attribute(
        self, attribute: Text, all_tokens: List[List[Text]]
    ) -> Tuple[
        List[Optional[scipy.sparse.spmatrix]], List[Optional[scipy.sparse.spmatrix]]
    ]:
        """Returns features of a particular attribute for complete data."""
        if self._get_attribute_vocabulary(attribute) is not None:
            # count vectorizer was trained
            return self._create_features(attribute, all_tokens)
        else:
            return [], []

    def train(
        self, training_data: TrainingData, model: Optional[SpacyModel] = None
    ) -> Resource:
        """Trains the featurizer.

        Take parameters from config and
        construct a new count vectorizer using the sklearn framework.
        """
        if model is not None:
            # create spacy lemma_ for OOV_words
            self.OOV_words = [
                t.lemma_ if self.use_lemma else t.text
                for w in self.OOV_words
                for t in model.model(w)
            ]

        # process sentences and collect data for all attributes
        processed_attribute_tokens = self._get_all_attributes_processed_tokens(
            training_data
        )

        # train for all attributes
        attribute_texts = self._convert_attribute_tokens_to_texts(
            processed_attribute_tokens
        )
        if self.use_shared_vocab:
            self._train_with_shared_vocab(attribute_texts)
        else:
            self._train_with_independent_vocab(attribute_texts)

        self.persist()

        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: the training data

        Returns:
          same training data after processing
        """
        self.process(training_data.training_examples)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming message and compute and set features."""
        if self.vectorizers is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return messages

        for message in messages:
            for attribute in self._attributes:

                message_tokens = self._get_processed_message_tokens_by_attribute(
                    message, attribute
                )

                # features shape (1, seq, dim)
                sequence_features, sentence_features = self._create_features(
                    attribute, [message_tokens]
                )
                self.add_features_to_message(
                    sequence_features[0], sentence_features[0], attribute, message
                )

        return messages

    def _collect_vectorizer_vocabularies(self) -> Dict[Text, Optional[Dict[Text, int]]]:
        """Gets vocabulary for all attributes."""
        attribute_vocabularies = {}
        for attribute in self._attributes:
            attribute_vocabularies[attribute] = self._get_attribute_vocabulary(
                attribute
            )
        return attribute_vocabularies

    @staticmethod
    def _is_any_model_trained(
        attribute_vocabularies: Dict[Text, Optional[Dict[Text, int]]]
    ) -> bool:
        """Check if any model got trained."""
        return any(value is not None for value in attribute_vocabularies.values())

    @staticmethod
    def convert_vocab(
        vocab: Dict[str, Union[int, Optional[Dict[str, int]]]], to_int: bool
    ) -> Dict[str, Union[None, int, np.int64, Dict[str, Union[int, np.int64]]]]:
        """Converts numpy integers in the vocabulary to Python integers."""

        def convert_value(value: int) -> Union[int, np.int64]:
            """Helper function to convert a single value based on to_int flag."""
            return int(value) if to_int else np.int64(value)

        result_dict: Dict[
            str, Union[None, int, np.int64, Dict[str, Union[int, np.int64]]]
        ] = {}
        for key, sub_dict in vocab.items():
            if isinstance(sub_dict, int):
                result_dict[key] = convert_value(sub_dict)
            elif not sub_dict:
                result_dict[key] = None
            else:
                result_dict[key] = {
                    sub_key: convert_value(value) for sub_key, value in sub_dict.items()
                }

        return result_dict

    def persist(self) -> None:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """
        if not self.vectorizers:
            return

        with self._model_storage.write_to(self._resource) as model_dir:
            # vectorizer instance was not None, some models could have been trained
            attribute_vocabularies = self._collect_vectorizer_vocabularies()
            if self._is_any_model_trained(attribute_vocabularies):
                # Definitely need to persist some vocabularies
                featurizer_file = model_dir / "vocabularies.json"

                # Only persist vocabulary from one attribute if `use_shared_vocab`.
                # Can be loaded and distributed to all attributes.
                loaded_vocab = (
                    attribute_vocabularies[TEXT]
                    if self.use_shared_vocab
                    else attribute_vocabularies
                )
                vocab = self.convert_vocab(loaded_vocab, to_int=True)

                rasa.shared.utils.io.dump_obj_as_json_to_file(featurizer_file, vocab)

                # Dump OOV words separately as they might have been modified during
                # training
                rasa.shared.utils.io.dump_obj_as_json_to_file(
                    model_dir / "oov_words.json", self.OOV_words
                )

    @classmethod
    def _create_shared_vocab_vectorizers(
        cls, parameters: Dict[Text, Any], vocabulary: Optional[Any] = None
    ) -> Dict[Text, CountVectorizer]:
        """Create vectorizers for all attributes with shared vocabulary."""
        shared_vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b" if parameters["analyzer"] == "word" else None,
            strip_accents=parameters["strip_accents"],
            lowercase=parameters["lowercase"],
            stop_words=parameters["stop_words"],
            ngram_range=(parameters["min_ngram"], parameters["max_ngram"]),
            max_df=parameters["max_df"],
            min_df=parameters["min_df"],
            max_features=parameters["max_features"],
            analyzer=parameters["analyzer"],
            vocabulary=vocabulary,
        )

        attribute_vectorizers = {}

        for attribute in cls._attributes_for(parameters["analyzer"]):
            attribute_vectorizers[attribute] = shared_vectorizer

        return attribute_vectorizers

    @classmethod
    def _create_independent_vocab_vectorizers(
        cls, parameters: Dict[Text, Any], vocabulary: Optional[Any] = None
    ) -> Dict[Text, CountVectorizer]:
        """Create vectorizers for all attributes with independent vocabulary."""
        attribute_vectorizers = {}

        for attribute in cls._attributes_for(parameters["analyzer"]):
            attribute_vocabulary = vocabulary[attribute] if vocabulary else None

            attribute_vectorizer = CountVectorizer(
                token_pattern=r"(?u)\b\w+\b"
                if parameters["analyzer"] == "word"
                else None,
                strip_accents=parameters["strip_accents"],
                lowercase=parameters["lowercase"],
                stop_words=parameters["stop_words"],
                ngram_range=(parameters["min_ngram"], parameters["max_ngram"]),
                max_df=parameters["max_df"],
                min_df=parameters["min_df"]
                if attribute == rasa.shared.nlu.constants.TEXT
                else 1,
                max_features=parameters["max_features"],
                analyzer=parameters["analyzer"],
                vocabulary=attribute_vocabulary,
            )
            attribute_vectorizers[attribute] = attribute_vectorizer

        return attribute_vectorizers

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CountVectorsFeaturizer:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                featurizer_file = model_dir / "vocabularies.json"
                vocabulary = rasa.shared.utils.io.read_json_file(featurizer_file)
                vocabulary = cls.convert_vocab(vocabulary, to_int=False)

                share_vocabulary = config["use_shared_vocab"]

                if share_vocabulary:
                    vectorizers = cls._create_shared_vocab_vectorizers(
                        config, vocabulary=vocabulary
                    )
                else:
                    vectorizers = cls._create_independent_vocab_vectorizers(
                        config, vocabulary=vocabulary
                    )

                oov_words = rasa.shared.utils.io.read_json_file(
                    model_dir / "oov_words.json"
                )

                ftr = cls(
                    config,
                    model_storage,
                    resource,
                    execution_context,
                    vectorizers=vectorizers,
                    oov_token=config["OOV_token"],
                    oov_words=oov_words,
                )

                # make sure the vocabulary has been loaded correctly
                for attribute in vectorizers:
                    ftr.vectorizers[attribute]._validate_vocabulary()

                return ftr

        except (ValueError, FileNotFoundError, FileIOException):
            logger.debug(
                f"Failed to load `{cls.__class__.__name__}` from model storage. "
                f"Resource '{resource.name}' doesn't exist."
            )
            return cls(
                config=config,
                model_storage=model_storage,
                resource=resource,
                execution_context=execution_context,
            )

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
