# =============================================================================
# Rasa NLU CRF Entity Extractor 条件随机场实体提取器模块
# 本模块实现了基于条件随机场（CRF）的命名实体识别功能，
# 用于从文本中提取实体信息
# =============================================================================

# 导入未来版本注解支持
from __future__ import annotations

# 导入标准库模块
import logging  # 日志记录
import typing  # 类型注解
from collections import OrderedDict  # 有序字典
from enum import Enum  # 枚举类
from typing import Any, Dict, List, Optional, Text, Tuple, Callable, Type  # 类型注解

# 导入科学计算库
import numpy as np  # 数值计算

# 导入 Rasa 核心模块
import rasa.nlu.utils.bilou_utils as bilou_utils  # BILOU 标签工具
import rasa.shared.utils.io  # 共享工具模块
import rasa.utils.train_utils  # 训练工具
from rasa.engine.graph import GraphComponent, ExecutionContext  # 图组件和执行上下文
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方
from rasa.engine.storage.resource import Resource  # 资源管理
from rasa.engine.storage.storage import ModelStorage  # 模型存储
from rasa.nlu.constants import TOKENS_NAMES  # 标记名称常量
from rasa.nlu.extractors.extractor import EntityExtractorMixin  # 实体提取器混入
from rasa.nlu.test import determine_token_labels  # 确定标记标签
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY  # 词性标记键
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer  # 标记和标记化器
from rasa.shared.constants import DOCS_URL_COMPONENTS  # 文档URL常量
from rasa.shared.nlu.constants import (
    TEXT,  # 文本常量
    ENTITIES,  # 实体常量
    ENTITY_ATTRIBUTE_TYPE,  # 实体属性类型
    ENTITY_ATTRIBUTE_GROUP,  # 实体属性组
    ENTITY_ATTRIBUTE_ROLE,  # 实体属性角色
    NO_ENTITY_TAG,  # 非实体标签
    SPLIT_ENTITIES_BY_COMMA,  # 按逗号分割实体
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,  # 按逗号分割实体默认值
)
from rasa.shared.nlu.training_data.message import Message  # 消息类
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据
from rasa.utils.tensorflow.constants import BILOU_FLAG, FEATURIZERS  # TensorFlow 常量

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 类型检查时导入 CRF
if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF

# 配置特征常量
CONFIG_FEATURES = "features"  # 特征配置键


class CRFToken:
    """CRF 标记类，用于存储标记的特征信息。
    
    该类封装了用于 CRF 训练和预测的标记特征，
    包括文本、词性、模式、密集特征和实体标签。
    """
    
    def __init__(
        self,
        text: Text,
        pos_tag: Text,
        pattern: Dict[Text, Any],
        dense_features: np.ndarray,
        entity_tag: Text,
        entity_role_tag: Text,
        entity_group_tag: Text,
    ):
        """初始化 CRF 标记。
        
        Args:
            text: 标记文本
            pos_tag: 词性标记
            pattern: 模式特征
            dense_features: 密集特征数组
            entity_tag: 实体标签
            entity_role_tag: 实体角色标签
            entity_group_tag: 实体组标签
        """
        self.text = text  # 标记文本
        self.pos_tag = pos_tag  # 词性标记
        self.pattern = pattern  # 模式特征
        self.dense_features = dense_features  # 密集特征数组
        self.entity_tag = entity_tag  # 实体标签
        self.entity_role_tag = entity_role_tag  # 实体角色标签
        self.entity_group_tag = entity_group_tag  # 实体组标签

    def to_dict(self) -> Dict[str, Any]:
        """将 CRF 标记转换为字典格式。
        
        Returns:
            包含标记信息的字典
        """
        return {
            "text": self.text,  # 标记文本
            "pos_tag": self.pos_tag,  # 词性标记
            "pattern": self.pattern,  # 模式特征
            "dense_features": [str(x) for x in list(self.dense_features)],  # 密集特征（转换为字符串）
            "entity_tag": self.entity_tag,  # 实体标签
            "entity_role_tag": self.entity_role_tag,  # 实体角色标签
            "entity_group_tag": self.entity_group_tag,  # 实体组标签
        }

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "CRFToken":
        """从字典创建 CRF 标记实例。
        
        Args:
            data: 包含标记信息的字典
            
        Returns:
            CRF 标记实例
        """
        return cls(
            data["text"],  # 标记文本
            data["pos_tag"],  # 词性标记
            data["pattern"],  # 模式特征
            np.array([float(x) for x in data["dense_features"]]),  # 密集特征（转换为浮点数组）
            data["entity_tag"],  # 实体标签
            data["entity_role_tag"],  # 实体角色标签
            data["entity_group_tag"],  # 实体组标签
        )


class CRFEntityExtractorOptions(str, Enum):
    """CRF 实体提取器可以使用的特征选项。
    
    该枚举定义了 CRF 实体提取器支持的所有特征类型，
    包括文本特征、词性特征、模式特征等。
    """

    PATTERN = "pattern"  # 模式特征
    LOW = "low"  # 小写特征
    TITLE = "title"  # 标题特征
    PREFIX5 = "prefix5"  # 5字符前缀
    PREFIX2 = "prefix2"  # 2字符前缀
    SUFFIX5 = "suffix5"  # 5字符后缀
    SUFFIX3 = "suffix3"  # 3字符后缀
    SUFFIX2 = "suffix2"  # 2字符后缀
    SUFFIX1 = "suffix1"  # 1字符后缀
    BIAS = "bias"  # 偏置特征
    POS = "pos"  # 词性特征
    POS2 = "pos2"  # 2字符词性特征
    UPPER = "upper"  # 大写特征
    DIGIT = "digit"  # 数字特征
    TEXT_DENSE_FEATURES = "text_dense_features"  # 文本密集特征
    ENTITY = "entity"  # 实体特征


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True  # 实体提取器组件类型，可训练
)
class CRFEntityExtractor(GraphComponent, EntityExtractorMixin):
    """实现条件随机场（CRF）进行命名实体识别。
    
    该类使用条件随机场算法从文本中提取实体信息，
    支持多种特征类型和 BILOU 标记方案。
    """

    CONFIG_FEATURES = "features"  # 特征配置键

    # 特征函数字典，将特征选项映射到提取函数
    function_dict: Dict[Text, Callable[[CRFToken], Any]] = {
        CRFEntityExtractorOptions.LOW: lambda crf_token: crf_token.text.lower(),  # 小写文本
        CRFEntityExtractorOptions.TITLE: lambda crf_token: crf_token.text.istitle(),  # 是否标题格式
        CRFEntityExtractorOptions.PREFIX5: lambda crf_token: crf_token.text[:5],  # 5字符前缀
        CRFEntityExtractorOptions.PREFIX2: lambda crf_token: crf_token.text[:2],  # 2字符前缀
        CRFEntityExtractorOptions.SUFFIX5: lambda crf_token: crf_token.text[-5:],  # 5字符后缀
        CRFEntityExtractorOptions.SUFFIX3: lambda crf_token: crf_token.text[-3:],  # 3字符后缀
        CRFEntityExtractorOptions.SUFFIX2: lambda crf_token: crf_token.text[-2:],  # 2字符后缀
        CRFEntityExtractorOptions.SUFFIX1: lambda crf_token: crf_token.text[-1:],  # 1字符后缀
        CRFEntityExtractorOptions.BIAS: lambda _: "bias",  # 偏置特征
        CRFEntityExtractorOptions.POS: lambda crf_token: crf_token.pos_tag,  # 词性标记
        CRFEntityExtractorOptions.POS2: lambda crf_token: crf_token.pos_tag[:2]  # 2字符词性标记
        if crf_token.pos_tag is not None
        else None,
        CRFEntityExtractorOptions.UPPER: lambda crf_token: crf_token.text.isupper(),  # 是否全大写
        CRFEntityExtractorOptions.DIGIT: lambda crf_token: crf_token.text.isdigit(),  # 是否全数字
        CRFEntityExtractorOptions.PATTERN: lambda crf_token: crf_token.pattern,  # 模式特征
        CRFEntityExtractorOptions.TEXT_DENSE_FEATURES: (  # 文本密集特征
            lambda crf_token: CRFEntityExtractor._convert_dense_features_for_crfsuite(  # noqa: E501
                crf_token
            )
        ),
        CRFEntityExtractorOptions.ENTITY: lambda crf_token: crf_token.entity_tag,  # 实体标签
    }

    @classmethod
    def required_components(cls) -> List[Type]:
        """在此组件之前应包含在管道中的组件。
        
        Returns:
            必需的组件类型列表
        """
        return [Tokenizer]  # 需要标记化器组件

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """组件的默认配置（完整文档字符串请参见父类）。
        
        Returns:
            默认配置字典
        """
        return {
            # BILOU_flag 确定是否使用 BILOU 标记
            # 更严格但每个实体需要更多示例
            # 经验法则：仅当每个实体超过 100 个示例时使用
            BILOU_FLAG: True,  # BILOU 标志
            # 按逗号分割实体，这对于成分列表等有意义，
            # 但对于地址的各个部分没有意义
            SPLIT_ENTITIES_BY_COMMA: True,  # 按逗号分割实体
            # crf_features 是 [before, token, after] 数组，其中 before、token、
            # after 包含每个标记要使用的特征键，
            # 例如，before 数组中的 'title' 将具有特征
            # "前面的标记是标题格式吗？"
            # POS 特征需要 SpacyTokenizer
            # 模式特征需要 RegexFeaturizer
            CONFIG_FEATURES: [  # 特征配置
                [  # 前一个标记的特征
                    CRFEntityExtractorOptions.LOW,  # 小写
                    CRFEntityExtractorOptions.TITLE,  # 标题格式
                    CRFEntityExtractorOptions.UPPER,  # 大写
                ],
                [  # 当前标记的特征
                    CRFEntityExtractorOptions.LOW,  # 小写
                    CRFEntityExtractorOptions.BIAS,  # 偏置
                    CRFEntityExtractorOptions.PREFIX5,  # 5字符前缀
                    CRFEntityExtractorOptions.PREFIX2,  # 2字符前缀
                    CRFEntityExtractorOptions.SUFFIX5,  # 5字符后缀
                    CRFEntityExtractorOptions.SUFFIX3,  # 3字符后缀
                    CRFEntityExtractorOptions.SUFFIX2,  # 2字符后缀
                    CRFEntityExtractorOptions.UPPER,  # 大写
                    CRFEntityExtractorOptions.TITLE,  # 标题格式
                    CRFEntityExtractorOptions.DIGIT,  # 数字
                    CRFEntityExtractorOptions.PATTERN,  # 模式
                ],
                [  # 后一个标记的特征
                    CRFEntityExtractorOptions.LOW,  # 小写
                    CRFEntityExtractorOptions.TITLE,  # 标题格式
                    CRFEntityExtractorOptions.UPPER,  # 大写
                ],
            ],
            # 优化算法的最大迭代次数
            "max_iterations": 50,  # 最大迭代次数
            # L1 正则化的权重
            "L1_c": 0.1,  # L1 正则化系数
            # L2 正则化的权重
            "L2_c": 0.1,  # L2 正则化系数
            # 要使用的密集特征化器名称
            # 如果列表为空，则使用所有可用的密集特征
            "featurizers": [],  # 特征化器列表
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        entity_taggers: Optional[Dict[Text, "CRF"]] = None,
    ) -> None:
        """创建实体提取器实例。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储
            resource: 资源
            entity_taggers: 实体标记器字典
        """
        self.component_config = config  # 组件配置
        self._model_storage = model_storage  # 模型存储
        self._resource = resource  # 资源

        self.entity_taggers = entity_taggers  # 实体标记器

        # CRF 顺序：类型、角色、组
        self.crf_order = [
            ENTITY_ATTRIBUTE_TYPE,  # 实体属性类型
            ENTITY_ATTRIBUTE_ROLE,  # 实体属性角色
            ENTITY_ATTRIBUTE_GROUP,  # 实体属性组
        ]

        # 验证配置
        self._validate_configuration()

        # 初始化实体分割配置
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            config[SPLIT_ENTITIES_BY_COMMA], SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )

    def _validate_configuration(self) -> None:
        """验证配置参数。
        
        Raises:
            ValueError: 如果特征列表数量不是奇数
        """
        if len(self.component_config.get(CONFIG_FEATURES, [])) % 2 != 1:
            raise ValueError(
                "Need an odd number of crf feature lists to have a center word."
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CRFEntityExtractor:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn_crfsuite", "sklearn"]

    def train(self, training_data: TrainingData) -> Resource:
        """在数据集上训练提取器。
        
        Args:
            training_data: 训练数据
            
        Returns:
            资源对象
        """
        # 检查是否至少有一个带实体注释的示例
        if not training_data.entity_examples:
            logger.debug(
                "No training examples with entities present. Skip training"
                "of 'CRFEntityExtractor'."
            )
            return self._resource

        # 检查实体注释的正确性
        self.check_correct_entity_annotations(training_data)

        # 如果启用 BILOU 标志，应用 BILOU 标记方案
        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        # 只保留我们实际有训练数据的标签的 CRF
        self._update_crf_order(training_data)

        # 过滤掉预训练的实体示例
        entity_examples = self.filter_trainable_entities(training_data.nlu_examples)
        # 过滤出有特征的示例
        entity_examples = [
            message
            for message in entity_examples
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]
        # 转换为 CRF 标记格式
        dataset = [self._convert_to_crf_tokens(example) for example in entity_examples]

        # 训练模型
        self.entity_taggers = self.train_model(
            dataset, self.component_config, self.crf_order
        )

        # 持久化模型
        self.persist(dataset)

        return self._resource

    def _update_crf_order(self, training_data: TrainingData) -> None:
        """Train only CRFs we actually have training data for."""
        _crf_order = []

        for tag_name in self.crf_order:
            if tag_name == ENTITY_ATTRIBUTE_TYPE and training_data.entities:
                _crf_order.append(ENTITY_ATTRIBUTE_TYPE)
            elif tag_name == ENTITY_ATTRIBUTE_ROLE and training_data.entity_roles:
                _crf_order.append(ENTITY_ATTRIBUTE_ROLE)
            elif tag_name == ENTITY_ATTRIBUTE_GROUP and training_data.entity_groups:
                _crf_order.append(ENTITY_ATTRIBUTE_GROUP)

        self.crf_order = _crf_order

    def process(self, messages: List[Message]) -> List[Message]:
        """用实体增强消息。
        
        Args:
            messages: 消息列表
            
        Returns:
            增强后的消息列表
        """
        for message in messages:
            # 提取实体
            entities = self.extract_entities(message)
            # 添加提取器名称
            entities = self.add_extractor_name(entities)
            # 设置实体到消息中
            message.set(
                ENTITIES, message.get(ENTITIES, []) + entities, add_to_output=True
            )

        return messages

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """使用训练好的模型从给定消息中提取实体。
        
        Args:
            message: 要提取实体的消息
            
        Returns:
            提取的实体列表
        """
        # 检查是否有实体标记器和特征
        if self.entity_taggers is None or not message.features_present(
            attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
        ):
            return []

        # 获取标记和转换为 CRF 格式
        tokens = message.get(TOKENS_NAMES[TEXT])
        crf_tokens = self._convert_to_crf_tokens(message)

        # 预测实体标签
        predictions: Dict[Text, List[Dict[Text, float]]] = {}
        for tag_name, entity_tagger in self.entity_taggers.items():
            # 对于第二级 CRF，使用预测的实体标签作为特征
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            if include_tag_features:
                self._add_tag_to_crf_token(crf_tokens, predictions)

            # 将 CRF 标记转换为特征
            features = self._crf_tokens_to_features(
                crf_tokens, self.component_config, include_tag_features
            )
            # 预测标签概率
            predictions[tag_name] = entity_tagger.predict_marginals_single(features)

        # 将预测转换为标签列表和置信度列表
        tags, confidences = self._tag_confidences(tokens, predictions)

        # 将预测转换为实体
        return self.convert_predictions_into_entities(
            message.get(TEXT), tokens, tags, self.split_entities_config, confidences
        )

    def _add_tag_to_crf_token(
        self,
        crf_tokens: List[CRFToken],
        predictions: Dict[Text, List[Dict[Text, float]]],
    ) -> None:
        """Add predicted entity tags to CRF tokens."""
        if ENTITY_ATTRIBUTE_TYPE in predictions:
            _tags, _ = self._most_likely_tag(predictions[ENTITY_ATTRIBUTE_TYPE])
            for tag, token in zip(_tags, crf_tokens):
                token.entity_tag = tag

    def _most_likely_tag(
        self, predictions: List[Dict[Text, float]]
    ) -> Tuple[List[Text], List[float]]:
        """Get the entity tags with the highest confidence.

        Args:
            predictions: list of mappings from entity tag to confidence value

        Returns:
            List of entity tags and list of confidence values.
        """
        _tags = []
        _confidences = []

        for token_predictions in predictions:
            tag = max(token_predictions, key=lambda key: token_predictions[key])
            _tags.append(tag)

            if self.component_config[BILOU_FLAG]:
                # if we are using BILOU flags, we will sum up the prob
                # of the B, I, L and U tags for an entity
                _confidences.append(
                    sum(
                        _confidence
                        for _tag, _confidence in token_predictions.items()
                        if bilou_utils.tag_without_prefix(tag)
                        == bilou_utils.tag_without_prefix(_tag)
                    )
                )
            else:
                _confidences.append(token_predictions[tag])

        return _tags, _confidences

    def _tag_confidences(
        self, tokens: List[Token], predictions: Dict[Text, List[Dict[Text, float]]]
    ) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
        """Get most likely tag predictions with confidence values for tokens."""
        tags = {}
        confidences = {}

        for tag_name, predicted_tags in predictions.items():
            if len(tokens) != len(predicted_tags):
                raise Exception(
                    "Inconsistency in amount of tokens between crfsuite and message"
                )

            _tags, _confidences = self._most_likely_tag(predicted_tags)

            if self.component_config[BILOU_FLAG]:
                _tags, _confidences = bilou_utils.ensure_consistent_bilou_tagging(
                    _tags, _confidences
                )

            confidences[tag_name] = _confidences
            tags[tag_name] = _tags

        return tags, confidences

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CRFEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                dataset = rasa.shared.utils.io.read_json_file(
                    model_dir / "crf_dataset.json"
                )
                crf_order = rasa.shared.utils.io.read_json_file(
                    model_dir / "crf_order.json"
                )

                dataset = [
                    [CRFToken.create_from_dict(token_data) for token_data in sub_list]
                    for sub_list in dataset
                ]

                entity_taggers = cls.train_model(dataset, config, crf_order)

                entity_extractor = cls(config, model_storage, resource, entity_taggers)
                entity_extractor.crf_order = crf_order
                return entity_extractor
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource)

    def persist(self, dataset: List[List[CRFToken]]) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            data_to_store = [
                [token.to_dict() for token in sub_list] for sub_list in dataset
            ]

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_dir / "crf_dataset.json", data_to_store
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_dir / "crf_order.json", self.crf_order
            )

    @classmethod
    def _crf_tokens_to_features(
        cls,
        crf_tokens: List[CRFToken],
        config: Dict[str, Any],
        include_tag_features: bool = False,
    ) -> List[Dict[Text, Any]]:
        """Convert the list of tokens into discrete features."""
        configured_features = config[CONFIG_FEATURES]
        sentence_features = []

        for token_idx in range(len(crf_tokens)):
            # the features for the current token include features of the token
            # before and after the current features (if defined in the config)
            # token before (-1), current token (0), token after (+1)
            window_size = len(configured_features)
            half_window_size = window_size // 2
            window_range = range(-half_window_size, half_window_size + 1)

            token_features = cls._create_features_for_token(
                crf_tokens,
                token_idx,
                half_window_size,
                window_range,
                include_tag_features,
                config,
            )

            sentence_features.append(token_features)

        return sentence_features

    @classmethod
    def _create_features_for_token(
        cls,
        crf_tokens: List[CRFToken],
        token_idx: int,
        half_window_size: int,
        window_range: range,
        include_tag_features: bool,
        config: Dict[str, Any],
    ) -> Dict[Text, Any]:
        """Convert a token into discrete features including words before and after."""
        configured_features = config[CONFIG_FEATURES]
        prefixes = [str(i) for i in window_range]

        token_features = {}

        # iterate over the tokens in the window range (-1, 0, +1) to collect the
        # features for the token at token_idx
        for pointer_position in window_range:
            current_token_idx = token_idx + pointer_position

            if current_token_idx >= len(crf_tokens):
                # token is at the end of the sentence
                token_features["EOS"] = True
            elif current_token_idx < 0:
                # token is at the beginning of the sentence
                token_features["BOS"] = True
            else:
                token = crf_tokens[current_token_idx]

                # get the features to extract for the token we are currently looking at
                current_feature_idx = pointer_position + half_window_size
                features = configured_features[current_feature_idx]

                prefix = prefixes[current_feature_idx]

                # we add the 'entity' feature to include the entity type as features
                # for the role and group CRFs
                # (do not modify features, otherwise we will end up adding 'entity'
                # over and over again, making training very slow)
                additional_features = []
                if include_tag_features:
                    additional_features.append(CRFEntityExtractorOptions.ENTITY)

                for feature in features + additional_features:
                    if feature == CRFEntityExtractorOptions.PATTERN:
                        # add all regexes extracted from the 'RegexFeaturizer' as a
                        # feature: 'pattern_name' is the name of the pattern the user
                        # set in the training data, 'matched' is either 'True' or
                        # 'False' depending on whether the token actually matches the
                        # pattern or not
                        regex_patterns = cls.function_dict[feature](token)
                        for pattern_name, matched in regex_patterns.items():
                            token_features[
                                f"{prefix}:{feature}:{pattern_name}"
                            ] = matched
                    else:
                        value = cls.function_dict[feature](token)
                        token_features[f"{prefix}:{feature}"] = value

        return token_features

    @staticmethod
    def _crf_tokens_to_tags(crf_tokens: List[CRFToken], tag_name: Text) -> List[Text]:
        """Return the list of tags for the given tag name."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            return [crf_token.entity_role_tag for crf_token in crf_tokens]
        if tag_name == ENTITY_ATTRIBUTE_GROUP:
            return [crf_token.entity_group_tag for crf_token in crf_tokens]

        return [crf_token.entity_tag for crf_token in crf_tokens]

    @staticmethod
    def _pattern_of_token(message: Message, idx: int) -> Dict[Text, bool]:
        """Get the patterns of the token at the given index extracted by the
        'RegexFeaturizer'.

        The 'RegexFeaturizer' adds all patterns listed in the training data to the
        token. The pattern name is mapped to either 'True' (pattern applies to token) or
        'False' (pattern does not apply to token).

        Args:
            message: The message.
            idx: The token index.

        Returns:
            The pattern dict.
        """
        if message.get(TOKENS_NAMES[TEXT]) is not None:
            return message.get(TOKENS_NAMES[TEXT])[idx].get(
                CRFEntityExtractorOptions.PATTERN, {}
            )
        return {}

    def _get_dense_features(self, message: Message) -> Optional[np.ndarray]:
        """Convert dense features to python-crfsuite feature format."""
        features, _ = message.get_dense_features(
            TEXT, self.component_config["featurizers"]
        )

        if features is None:
            return None

        tokens = message.get(TOKENS_NAMES[TEXT])
        if len(tokens) != len(features.features):
            rasa.shared.utils.io.raise_warning(
                f"Number of dense features ({len(features.features)}) for attribute "
                f"'{TEXT}' does not match number of tokens ({len(tokens)}).",
                docs=DOCS_URL_COMPONENTS + "#crfentityextractor",
            )
            return None

        return features.features

    @staticmethod
    def _convert_dense_features_for_crfsuite(
        crf_token: CRFToken,
    ) -> Dict[Text, Dict[Text, float]]:
        """Converts dense features of CRFTokens to dicts for the crfsuite."""
        feature_dict = {
            str(index): token_features
            for index, token_features in enumerate(crf_token.dense_features)
        }
        converted = {"text_dense_features": feature_dict}
        return converted

    def _convert_to_crf_tokens(self, message: Message) -> List[CRFToken]:
        """Take a message and convert it to crfsuite format."""
        crf_format = []
        tokens = message.get(TOKENS_NAMES[TEXT])

        text_dense_features = self._get_dense_features(message)
        tags = self._get_tags(message)

        for i, token in enumerate(tokens):
            pattern = self._pattern_of_token(message, i)
            entity = self.get_tag_for(tags, ENTITY_ATTRIBUTE_TYPE, i)
            group = self.get_tag_for(tags, ENTITY_ATTRIBUTE_GROUP, i)
            role = self.get_tag_for(tags, ENTITY_ATTRIBUTE_ROLE, i)
            pos_tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(
                    text=token.text,
                    pos_tag=pos_tag,
                    entity_tag=entity,
                    entity_group_tag=group,
                    entity_role_tag=role,
                    pattern=pattern,
                    dense_features=dense_features,
                )
            )

        return crf_format

    def _get_tags(self, message: Message) -> Dict[Text, List[Text]]:
        """Get assigned entity tags of message."""
        tokens = message.get(TOKENS_NAMES[TEXT])
        tags = {}

        for tag_name in self.crf_order:
            if self.component_config[BILOU_FLAG]:
                bilou_key = bilou_utils.get_bilou_key_for_tag(tag_name)
                if message.get(bilou_key):
                    _tags = message.get(bilou_key)
                else:
                    _tags = [NO_ENTITY_TAG for _ in tokens]
            else:
                _tags = [
                    determine_token_labels(
                        token, message.get(ENTITIES), attribute_key=tag_name
                    )
                    for token in tokens
                ]
            tags[tag_name] = _tags

        return tags

    @classmethod
    def train_model(
        cls,
        df_train: List[List[CRFToken]],
        config: Dict[str, Any],
        crf_order: List[str],
    ) -> OrderedDict[str, CRF]:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        entity_taggers = OrderedDict()

        for tag_name in crf_order:
            logger.debug(f"Training CRF for '{tag_name}'.")

            # add entity tag features for second level CRFs
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            X_train = (
                cls._crf_tokens_to_features(sentence, config, include_tag_features)
                for sentence in df_train
            )
            y_train = (
                cls._crf_tokens_to_tags(sentence, tag_name) for sentence in df_train
            )

            entity_tagger = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                # coefficient for L1 penalty
                c1=config["L1_c"],
                # coefficient for L2 penalty
                c2=config["L2_c"],
                # stop earlier
                max_iterations=config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True,
            )
            entity_tagger.fit(X_train, y_train)

            entity_taggers[tag_name] = entity_tagger

            logger.debug("Training finished.")

        return entity_taggers
