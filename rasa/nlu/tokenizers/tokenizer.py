# 导入抽象基类模块
import abc
# 导入日志记录模块
import logging
# 导入正则表达式模块
import re

# 导入类型注解相关的类型
from typing import Text, List, Dict, Any, Optional

# 导入图引擎相关模块
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入训练数据相关模块
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
# 导入NLU常量
from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from rasa.shared.nlu.constants import (
    INTENT,  # 意图常量
    INTENT_RESPONSE_KEY,  # 意图响应键常量
    RESPONSE_IDENTIFIER_DELIMITER,  # 响应标识符分隔符
    ACTION_NAME,  # 动作名称常量
)
# 导入共享工具模块
import rasa.shared.utils.io

# 创建日志记录器
logger = logging.getLogger(__name__)


# 定义Token类
class Token:
    """由 `Tokenizers` 使用的类，用于将单个消息分割成多个 `Token` 对象。

    Token 类表示文本中的一个词汇单元，包含文本内容、位置信息和附加数据。
    它是分词器处理的基本单位，用于后续的特征提取和意图识别。
    """

    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        """创建一个 `Token` 对象。

        Args:
            text: 词汇单元的文本内容。
            start: 词汇单元在整个消息中的起始索引位置。
            end: 词汇单元在整个消息中的结束索引位置。如果未提供，则自动计算。
            data: 词汇单元的附加数据字典，用于存储额外的属性信息。
            lemma: 词汇单元的词形还原版本，如果未提供则使用原文本。
        """
        # 存储词汇单元的文本内容
        self.text = text
        # 存储起始位置
        self.start = start
        # 存储结束位置，如果未提供则根据文本长度自动计算
        self.end = end if end else start + len(text)

        # 存储附加数据，如果未提供则使用空字典
        self.data = data if data else {}
        # 存储词形还原版本，如果未提供则使用原文本
        self.lemma = lemma or text

    def set(self, prop: Text, info: Any) -> None:
        """设置属性值。

        在Token的附加数据字典中设置指定的属性值。

        Args:
            prop: 属性名称。
            info: 属性值。
        """
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        """获取属性值。

        从Token的附加数据字典中获取指定的属性值。

        Args:
            prop: 属性名称。
            default: 如果属性不存在时返回的默认值。

        Returns:
            属性值或默认值。
        """
        return self.data.get(prop, default)

    def __eq__(self, other: Any) -> bool:
        """判断两个Token对象是否相等。

        通过比较起始位置、结束位置、文本内容和词形还原版本来判断相等性。

        Args:
            other: 要比较的另一个对象。

        Returns:
            如果两个Token相等返回True，否则返回False。
        """
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) == (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __lt__(self, other: Any) -> bool:
        """判断当前Token是否小于另一个Token。

        用于Token对象的排序，按照起始位置、结束位置、文本内容和词形还原版本进行比较。

        Args:
            other: 要比较的另一个对象。

        Returns:
            如果当前Token小于另一个Token返回True，否则返回False。
        """
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) < (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __repr__(self) -> Text:
        """返回Token对象的字符串表示。

        用于调试和日志记录，显示Token的关键信息。

        Returns:
            Token对象的字符串表示。
        """
        return f"<Token object value='{self.text}' start={self.start} end={self.end} \
        at {hex(id(self))}>"

    def fingerprint(self) -> Text:
        """返回Token的稳定哈希值。

        用于缓存和去重，基于Token的所有关键属性生成稳定的哈希值。

        Returns:
            Token的指纹哈希值。
        """
        return rasa.shared.utils.io.deep_container_fingerprint(
            [self.text, self.start, self.end, self.lemma, self.data]
        )


# 定义分词器抽象基类
class Tokenizer(GraphComponent, abc.ABC):
    """分词器的基类。

    所有分词器都必须继承此类并实现 `tokenize` 方法。分词器负责将文本分割成
    词汇单元（Token），这是NLU管道中的关键步骤，为后续的特征提取和意图识别
    提供基础数据。
    """

    def __init__(self, config: Dict[Text, Any]) -> None:
        """构造一个新的分词器。

        初始化分词器配置，包括意图分词标志、分隔符、分词模式等。

        Args:
            config: 分词器配置字典。
        """
        # 存储配置字典
        self._config = config
        # 标志位：是否对意图进行分词
        self.intent_tokenization_flag = config["intent_tokenization_flag"]
        # 意图分词使用的分隔符
        self.intent_split_symbol = config["intent_split_symbol"]
        # 用于进一步分割词汇的分词模式
        token_pattern = config.get("token_pattern")
        self.token_pattern_regex = None
        if token_pattern:
            # 编译正则表达式模式
            self.token_pattern_regex = re.compile(token_pattern)
        # 用于贪婪分割意图为前缀和后缀的分隔符，None表示不分割
        self.prefix_separator_symbol = config.get("prefix_separator_symbol")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """创建一个新的组件（参见父类的完整文档字符串）。

        实现GraphComponent接口的create方法，用于创建分词器实例。

        Args:
            config: 组件配置。
            model_storage: 模型存储接口。
            resource: 资源定位器。
            execution_context: 执行上下文。

        Returns:
            创建的分词器组件实例。
        """
        return cls(config)

    @abc.abstractmethod
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """对传入消息的指定属性文本进行分词。

        这是分词器的核心抽象方法，所有具体分词器都必须实现此方法。
        负责将文本分割成词汇单元列表。

        Args:
            message: 包含要分词文本的消息对象。
            attribute: 要分词的属性名称（如 'text', 'intent' 等）。

        Returns:
            分词后的Token对象列表。
        """
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """对所有训练数据进行分词。

        遍历训练数据中的所有示例，对每个消息的各个属性进行分词处理。
        这是训练阶段的主要处理方法。

        Args:
            training_data: 包含训练示例的训练数据对象。

        Returns:
            分词处理后的训练数据对象。
        """
        # 遍历所有训练示例
        for example in training_data.training_examples:
            # 遍历消息的所有属性
            for attribute in MESSAGE_ATTRIBUTES:
                # 检查属性是否存在且不为空
                if (
                    example.get(attribute) is not None
                    and not example.get(attribute) == ""
                ):
                    # 对特殊属性（意图、动作名、意图响应键）使用特殊分词方法
                    if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
                        tokens = self._split_name(example, attribute)
                    else:
                        # 对其他属性使用标准分词方法
                        tokens = self.tokenize(example, attribute)
                    # 将分词结果设置到示例中
                    example.set(TOKENS_NAMES[attribute], tokens)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """对传入的消息进行分词。

        这是推理阶段的主要处理方法，对消息列表中的每个消息进行分词处理。

        Args:
            messages: 要分词的消息列表。

        Returns:
            分词处理后的消息列表。
        """
        # 遍历所有消息
        for message in messages:
            # 遍历消息的所有属性
            for attribute in MESSAGE_ATTRIBUTES:
                # 检查属性是否为字符串类型
                if isinstance(message.get(attribute), str):
                    # 对特殊属性使用特殊分词方法
                    if attribute in [
                        INTENT,
                        ACTION_NAME,
                        RESPONSE_IDENTIFIER_DELIMITER,
                    ]:
                        tokens = self._split_name(message, attribute)
                    else:
                        # 对其他属性使用标准分词方法
                        tokens = self.tokenize(message, attribute)

                    # 将分词结果设置到消息中
                    message.set(TOKENS_NAMES[attribute], tokens)
        return messages

    def _tokenize_on_split_symbol(self, text: Text) -> List[Text]:
        """根据分隔符对文本进行分词。

        根据配置的意图分词标志和分隔符，将文本分割成词汇列表。

        Args:
            text: 要分词的文本。

        Returns:
            分词后的词汇列表。
        """
        # 根据意图分词标志决定是否使用分隔符分词
        words = (
            text.split(self.intent_split_symbol)
            if self.intent_tokenization_flag
            else [text]
        )

        return words

    def _split_name(self, message: Message, attribute: Text = INTENT) -> List[Token]:
        """对名称类属性进行特殊分词处理。

        处理意图、动作名等特殊属性，支持前缀分隔符和响应标识符分隔符。

        Args:
            message: 包含要分词属性的消息对象。
            attribute: 要分词的属性名称，默认为意图。

        Returns:
            分词后的Token对象列表。
        """
        # 获取原始文本
        orig_text = message.get(attribute)

        # 检查是否需要按前缀分隔符分割
        if (
            self.prefix_separator_symbol is not None
            and self.prefix_separator_symbol in orig_text
        ):
            # 按前缀分隔符分割，最多分割一次
            prefix, text = orig_text.split(self.prefix_separator_symbol, maxsplit=1)
        else:
            # 不分割，前缀为None
            prefix, text = None, orig_text

        # 对于INTENT_RESPONSE_KEY属性，首先按响应标识符分隔符分割
        if attribute == INTENT_RESPONSE_KEY:
            intent, response_key = text.split(RESPONSE_IDENTIFIER_DELIMITER)
            # 分别对意图和响应键进行分词
            words = self._tokenize_on_split_symbol(
                intent
            ) + self._tokenize_on_split_symbol(response_key)

        else:
            # 对其他属性直接分词
            words = self._tokenize_on_split_symbol(text)

        # 如果存在前缀，将其分词结果添加到前面
        if prefix is not None:
            words = self._tokenize_on_split_symbol(prefix) + words

        # 将词汇列表转换为Token对象列表
        return self._convert_words_to_tokens(words, orig_text)

    def _apply_token_pattern(self, tokens: List[Token]) -> List[Token]:
        """对给定的Token应用分词模式。

        使用配置的正则表达式模式进一步分割Token，实现更细粒度的分词。

        Args:
            tokens: 要分割的Token列表。

        Returns:
            应用分词模式后的Token列表。
        """
        # 如果没有配置分词模式，直接返回原Token列表
        if not self.token_pattern_regex:
            return tokens

        final_tokens = []
        # 遍历每个Token
        for token in tokens:
            # 使用正则表达式查找匹配的子串
            new_tokens = self.token_pattern_regex.findall(token.text)
            # 过滤掉空字符串
            new_tokens = [t for t in new_tokens if t]

            # 如果没有找到匹配的子串，保留原Token
            if not new_tokens:
                final_tokens.append(token)

            # 计算每个新Token在原文本中的位置
            running_offset = 0
            for new_token in new_tokens:
                # 找到新Token在原Token文本中的位置
                word_offset = token.text.index(new_token, running_offset)
                word_len = len(new_token)
                running_offset = word_offset + word_len
                # 创建新的Token，保持原Token的数据和词形还原信息
                final_tokens.append(
                    Token(
                        new_token,
                        token.start + word_offset,
                        data=token.data,
                        lemma=token.lemma,
                    )
                )

        return final_tokens

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        """将词汇列表转换为Token对象列表。

        根据词汇在原文本中的位置信息创建Token对象。

        Args:
            words: 词汇列表。
            text: 原始文本。

        Returns:
            Token对象列表。
        """
        # 跟踪当前处理位置
        running_offset = 0
        tokens = []

        # 遍历每个词汇
        for word in words:
            # 在文本中找到词汇的位置
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            # 更新处理位置
            running_offset = word_offset + word_len
            # 创建Token对象
            tokens.append(Token(word, word_offset))

        return tokens
