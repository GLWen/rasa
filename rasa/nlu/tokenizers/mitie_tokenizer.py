# 启用类型注解的前向引用功能，允许在类型注解中使用字符串形式的类型
from __future__ import annotations
# 导入类型注解相关的类型
from typing import List, Text, Dict, Any

# 导入图引擎相关模块
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入分词器基类和Token类
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# 导入消息类
from rasa.shared.nlu.training_data.message import Message

# 导入默认编码常量
from rasa.shared.utils.io import DEFAULT_ENCODING


# 注册为默认配方中的消息分词器组件，不可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
# 定义MITIE分词器类
class MitieTokenizer(Tokenizer):
    """使用 `mitie` 库对消息进行分词。

    MITIE (MIT Information Extraction) 是一个信息提取库，提供了强大的
    自然语言处理功能，包括分词、命名实体识别等。该分词器利用MITIE库的
    分词能力，支持多种语言的分词处理。
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回默认配置（参见父类的完整文档字符串）。

        定义了MITIE分词器的默认参数设置。

        返回:
            包含默认配置的字典。
        """
        return {
            # 标志位：是否对意图进行分词
            "intent_tokenization_flag": False,
            # 意图分词使用的分隔符
            "intent_split_symbol": "_",
            # 用于检测词汇的正则表达式模式
            "token_pattern": None,
            # 前缀分割使用的分隔符
            "prefix_separator_symbol": None,
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """此组件运行所需的额外Python依赖包。

        返回:
            必需的Python包名称列表。
        """
        return ["mitie"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MitieTokenizer:
        """创建一个新的组件（参见父类的完整文档字符串）。

        实现GraphComponent接口的create方法，用于创建MITIE分词器实例。

        Args:
            config: 组件配置。
            model_storage: 模型存储接口。
            resource: 资源定位器。
            execution_context: 执行上下文。

        Returns:
            创建的MITIE分词器实例。
        """
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """对传入消息的指定属性文本进行分词。

        使用MITIE库对文本进行分词，这是MITIE分词器的核心方法。
        支持多种语言的分词处理，并保持精确的位置信息。

        Args:
            message: 包含要分词文本的消息对象。
            attribute: 要分词的属性名称。

        Returns:
            分词后的Token对象列表。
        """
        # 导入mitie模块
        import mitie

        # 获取要分词的文本
        text = message.get(attribute)

        # 将文本编码为字节序列，使用默认编码
        encoded_sentence = text.encode(DEFAULT_ENCODING)
        # 使用MITIE进行分词，返回(词汇, 字节偏移)的元组列表
        tokenized = mitie.tokenize_with_offsets(encoded_sentence)
        # 将MITIE分词结果转换为Token对象列表
        tokens = [
            self._token_from_offset(token, offset, encoded_sentence)
            for token, offset in tokenized
        ]

        # 应用分词模式（如果配置了的话）
        return self._apply_token_pattern(tokens)

    def _token_from_offset(
        self, text: bytes, offset: int, encoded_sentence: bytes
    ) -> Token:
        """从字节偏移量创建Token对象。

        将MITIE返回的字节序列和偏移量转换为Rasa的Token对象格式。

        Args:
            text: 词汇的字节序列。
            offset: 词汇在编码句子中的字节偏移量。
            encoded_sentence: 完整的编码句子字节序列。

        Returns:
            创建的Token对象。
        """
        return Token(
            # 将字节序列解码为字符串
            text.decode(DEFAULT_ENCODING),
            # 将字节偏移量转换为字符偏移量
            self._byte_to_char_offset(encoded_sentence, offset),
        )

    @staticmethod
    def _byte_to_char_offset(text: bytes, byte_offset: int) -> int:
        """将字节偏移量转换为字符偏移量。

        由于不同编码下字节数和字符数可能不同，需要将MITIE返回的
        字节偏移量转换为字符偏移量，以便正确创建Token对象。

        Args:
            text: 完整的编码文本字节序列。
            byte_offset: 字节偏移量。

        Returns:
            对应的字符偏移量。
        """
        return len(text[:byte_offset].decode(DEFAULT_ENCODING))
