# 导入类型检查模块
import typing
# 导入类型注解相关的类型
from typing import Dict, Text, List, Any, Optional, Type

# 导入默认配方注册装饰器
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 导入SpaCy NLP工具类
from rasa.nlu.utils.spacy_utils import SpacyNLP
# 导入分词器基类和Token类
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# 导入SpaCy文档常量
from rasa.nlu.constants import SPACY_DOCS
# 导入消息类
from rasa.shared.nlu.training_data.message import Message

# 仅在类型检查时导入，避免运行时循环导入
if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

# 定义词性标注键名常量
POS_TAG_KEY = "pos"


# 注册为默认配方中的消息分词器组件，不可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
# 定义SpaCy分词器类
class SpacyTokenizer(Tokenizer):
    """使用SpaCy库进行分词的分词器。

    SpaCy是一个先进的自然语言处理库，提供了强大的分词、词性标注、
    命名实体识别等功能。该分词器利用SpaCy的Doc对象进行分词，
    支持多种语言，并提供丰富的语言学信息。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """在此组件之前应该包含在管道中的组件。

        返回:
            必需的组件类型列表，SpaCy分词器需要SpacyNLP组件先运行。
        """
        return [SpacyNLP]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """组件的默认配置（参见父类的完整文档字符串）。

        定义了SpaCy分词器的默认参数设置。

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
        return ["spacy"]

    def _get_doc(self, message: Message, attribute: Text) -> Optional["Doc"]:
        """获取消息中指定属性的SpaCy文档对象。

        从消息中提取SpaCy处理后的文档对象，该对象包含分词、词性标注等语言学信息。

        Args:
            message: 包含SpaCy文档的消息对象。
            attribute: 要获取文档的属性名称。

        Returns:
            SpaCy文档对象，如果不存在则返回None。
        """
        return message.get(SPACY_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """对传入消息的指定属性文本进行分词。

        使用SpaCy库对文本进行分词，这是SpaCy分词器的核心方法。
        利用SpaCy的Doc对象获取分词结果，并提取词性标注等语言学信息。

        Args:
            message: 包含要分词文本的消息对象。
            attribute: 要分词的属性名称。

        Returns:
            分词后的Token对象列表，包含词性标注信息。
        """
        # 获取SpaCy文档对象
        doc = self._get_doc(message, attribute)
        # 如果文档不存在，返回空列表
        if not doc:
            return []

        # 从SpaCy文档中提取Token对象，包含词性标注信息
        tokens = [
            Token(
                t.text, t.idx, lemma=t.lemma_, data={POS_TAG_KEY: self._tag_of_token(t)}
            )
            for t in doc
            if t.text and t.text.strip()
        ]

        # 应用分词模式（如果配置了的话）
        return self._apply_token_pattern(tokens)

    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        """获取Token的词性标注。

        根据SpaCy版本和Token属性获取词性标注信息。
        兼容SpaCy 2.x和3.x版本的API差异。

        Args:
            token: SpaCy的Token对象。

        Returns:
            词性标注字符串。
        """
        # 导入spacy模块
        import spacy

        # 检查SpaCy版本和Token属性，选择合适的API
        if spacy.about.__version__ > "2" and token._.has("tag"):
            # SpaCy 3.x版本，使用扩展属性
            return token._.get("tag")
        else:
            # SpaCy 2.x版本或没有扩展属性，使用标准属性
            return token.tag_
