# 启用类型注解的前向引用功能，允许在类型注解中使用字符串形式的类型
from __future__ import annotations
# 导入类型注解相关的类型
from typing import Any, Dict, List, Optional, Text

# 导入regex模块，用于处理Unicode正则表达式
import regex

# 导入共享工具模块
import rasa.shared.utils.io
# 导入工具模块
import rasa.utils.io

# 导入图引擎相关模块
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入分词器基类和Token类
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# 导入常量
from rasa.shared.constants import DOCS_URL_COMPONENTS
# 导入消息类
from rasa.shared.nlu.training_data.message import Message


# 注册为默认配方中的消息分词器组件，不可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
# 定义空白字符分词器类
class WhitespaceTokenizer(Tokenizer):
    """空白字符分词器，用于创建实体提取的特征。

    基于空白字符和正则表达式进行分词，适用于大多数拉丁语系语言。
    通过复杂的正则表达式模式处理各种边界情况，包括URL、邮箱、数字等。
    """

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """返回不支持的语言列表。

        空白字符分词器不适用于不使用空格分隔词汇的语言，如中文、日文、泰文等。

        Returns:
            不支持的语言代码列表。
        """
        return ["zh", "ja", "th"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置。

        定义了空白字符分词器的默认参数设置。

        Returns:
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

    def __init__(self, config: Dict[Text, Any]) -> None:
        """初始化分词器。

        设置分词器的配置和表情符号处理模式。

        Args:
            config: 分词器配置字典。
        """
        # 调用父类初始化方法
        super().__init__(config)
        # 获取表情符号正则表达式模式
        self.emoji_pattern = rasa.utils.io.get_emoji_regex()

        # 检查是否包含已废弃的配置选项
        if "case_sensitive" in self._config:
            # 发出警告，提示配置选项已迁移
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> WhitespaceTokenizer:
        """创建一个新的组件（参见父类的完整文档字符串）。

        实现GraphComponent接口的create方法，用于创建空白字符分词器实例。

        Args:
            config: 组件配置。
            model_storage: 模型存储接口。
            resource: 资源定位器。
            execution_context: 执行上下文。

        Returns:
            创建的空白字符分词器实例。
        """
        return cls(config)

    def remove_emoji(self, text: Text) -> Text:
        """如果整个文本（即词汇）匹配表情符号正则表达式，则移除表情符号。

        检查文本是否完全由表情符号组成，如果是则返回空字符串。

        Args:
            text: 要检查的文本。

        Returns:
            移除表情符号后的文本，如果原文本是表情符号则返回空字符串。
        """
        # 使用fullmatch检查文本是否完全匹配表情符号模式
        match = self.emoji_pattern.fullmatch(text)

        # 如果匹配到表情符号，返回空字符串
        if match is not None:
            return ""

        # 否则返回原文本
        return text

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """对消息的指定属性进行分词。

        使用复杂的正则表达式模式处理各种边界情况，包括URL、邮箱、数字等。
        这是空白字符分词器的核心分词方法。

        Args:
            message: 包含要分词文本的消息对象。
            attribute: 要分词的属性名称。

        Returns:
            分词后的Token对象列表。
        """
        # 获取要分词的文本
        text = message.get(attribute)

        # 需要使用regex而不是re，因为Unicode正则表达式匹配问题
        # 参考：https://stackoverflow.com/questions/12746458/python-unicode-regular-expression-matching-failing-with-some-unicode-characters

        # 移除非词汇字符，使用复杂的正则表达式模式处理各种情况
        words = regex.sub(
            # 模式1：后面是空格或字符串结尾的非词汇字符
            r"[^\w#@&]+(?=\s|$)|"
            # 模式2：前面是空格或字符串开头，后面不是数字的非词汇字符
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # 模式3：不在数字之间，且不是特殊字符（如.@&-等）的非词汇字符
            # 例如：10'000.00 或 blabla@gmail.com
            # 以及非URL字符
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",  # 用空格替换匹配的字符
            text,
        ).split()  # 按空格分割成词汇列表

        # 移除每个词汇中的表情符号
        words = [self.remove_emoji(w) for w in words]
        # 过滤掉空字符串
        words = [w for w in words if w]

        # 如果移除了所有内容（如表情符号 `:)`），将整个文本作为1个词汇
        if not words:
            words = [text]

        # 将词汇列表转换为Token对象列表
        tokens = self._convert_words_to_tokens(words, text)

        # 应用分词模式（如果配置了的话）
        return self._apply_token_pattern(tokens)
