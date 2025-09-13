# =============================================================================
# Rasa NLU Response Selector 响应选择器模块
# 本模块实现了基于监督嵌入的响应选择器，用于从候选响应中选择最合适的响应
# =============================================================================

# 导入未来版本注解支持
from __future__ import annotations

# 导入标准库模块
import copy  # 深拷贝
import logging  # 日志记录
from rasa.nlu.featurizers.featurizer import Featurizer  # 特征化器基类

# 导入科学计算库
import numpy as np  # 数值计算
import tensorflow as tf  # 深度学习框架

# 导入类型注解
from typing import Any, Dict, Optional, Text, Tuple, Union, List, Type

# 导入 Rasa 核心模块
from rasa.engine.graph import ExecutionContext  # 执行上下文
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方
from rasa.engine.storage.resource import Resource  # 资源管理
from rasa.engine.storage.storage import ModelStorage  # 模型存储
from rasa.shared.constants import DIAGNOSTIC_DATA  # 诊断数据常量
from rasa.shared.nlu.training_data import util  # 训练数据工具
import rasa.shared.utils.io  # 共享工具模块
from rasa.shared.exceptions import InvalidConfigException  # 配置异常
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据
from rasa.shared.nlu.training_data.message import Message  # 消息类
from rasa.nlu.classifiers.diet_classifier import (  # DIET 分类器相关导入
    DIET,  # DIET 模型基类
    LABEL_KEY,  # 标签键
    LABEL_SUB_KEY,  # 标签子键
    SENTENCE,  # 句子特征
    SEQUENCE,  # 序列特征
    DIETClassifier,  # DIET 分类器
)
from rasa.nlu.extractors.extractor import EntityTagSpec  # 实体标签规范
from rasa.utils.tensorflow import rasa_layers  # Rasa TensorFlow 层
from rasa.utils.tensorflow.constants import (  # TensorFlow 常量导入
    LABEL,  # 标签常量
    HIDDEN_LAYERS_SIZES,  # 隐藏层大小
    SHARE_HIDDEN_LAYERS,  # 共享隐藏层
    TRANSFORMER_SIZE,  # 转换器大小
    NUM_TRANSFORMER_LAYERS,  # 转换器层数
    NUM_HEADS,  # 注意力头数
    BATCH_SIZES,  # 批次大小
    BATCH_STRATEGY,  # 批次策略
    EPOCHS,  # 训练轮数
    RANDOM_SEED,  # 随机种子
    LEARNING_RATE,  # 学习率
    RANKING_LENGTH,  # 排名长度
    RENORMALIZE_CONFIDENCES,  # 重新归一化置信度
    LOSS_TYPE,  # 损失类型
    SIMILARITY_TYPE,  # 相似度类型
    NUM_NEG,  # 负样本数量
    SPARSE_INPUT_DROPOUT,  # 稀疏输入丢弃
    DENSE_INPUT_DROPOUT,  # 密集输入丢弃
    MASKED_LM,  # 掩码语言模型
    ENTITY_RECOGNITION,  # 实体识别
    INTENT_CLASSIFICATION,  # 意图分类
    EVAL_NUM_EXAMPLES,  # 评估示例数量
    EVAL_NUM_EPOCHS,  # 评估轮数
    UNIDIRECTIONAL_ENCODER,  # 单向编码器
    DROP_RATE,  # 丢弃率
    DROP_RATE_ATTENTION,  # 注意力丢弃率
    CONNECTION_DENSITY,  # 连接密度
    NEGATIVE_MARGIN_SCALE,  # 负边距缩放
    REGULARIZATION_CONSTANT,  # 正则化常数
    SCALE_LOSS,  # 损失缩放
    USE_MAX_NEG_SIM,  # 使用最大负相似度
    MAX_NEG_SIM,  # 最大负相似度
    MAX_POS_SIM,  # 最大正相似度
    EMBEDDING_DIMENSION,  # 嵌入维度
    BILOU_FLAG,  # BILOU 标志
    KEY_RELATIVE_ATTENTION,  # 键相对注意力
    VALUE_RELATIVE_ATTENTION,  # 值相对注意力
    MAX_RELATIVE_POSITION,  # 最大相对位置
    RETRIEVAL_INTENT,  # 检索意图
    USE_TEXT_AS_LABEL,  # 使用文本作为标签
    CROSS_ENTROPY,  # 交叉熵
    AUTO,  # 自动
    BALANCED,  # 平衡
    TENSORBOARD_LOG_DIR,  # TensorBoard 日志目录
    TENSORBOARD_LOG_LEVEL,  # TensorBoard 日志级别
    CONCAT_DIMENSION,  # 连接维度
    FEATURIZERS,  # 特征化器
    CHECKPOINT_MODEL,  # 检查点模型
    DENSE_DIMENSION,  # 密集维度
    CONSTRAIN_SIMILARITIES,  # 约束相似度
    MODEL_CONFIDENCE,  # 模型置信度
    SOFTMAX,  # Softmax
)
from rasa.nlu.constants import (  # NLU 常量导入
    RESPONSE_SELECTOR_PROPERTY_NAME,  # 响应选择器属性名
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,  # 响应选择器检索意图
    RESPONSE_SELECTOR_RESPONSES_KEY,  # 响应选择器响应键
    RESPONSE_SELECTOR_PREDICTION_KEY,  # 响应选择器预测键
    RESPONSE_SELECTOR_RANKING_KEY,  # 响应选择器排名键
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,  # 响应选择器话语动作键
    RESPONSE_SELECTOR_DEFAULT_INTENT,  # 响应选择器默认意图
    DEFAULT_TRANSFORMER_SIZE,  # 默认转换器大小
)
from rasa.shared.nlu.constants import (  # 共享 NLU 常量导入
    TEXT,  # 文本常量
    INTENT,  # 意图常量
    RESPONSE,  # 响应常量
    INTENT_RESPONSE_KEY,  # 意图响应键
    INTENT_NAME_KEY,  # 意图名称键
    PREDICTED_CONFIDENCE_KEY,  # 预测置信度键
)

from rasa.utils.tensorflow.model_data import RasaModelData  # Rasa 模型数据
from rasa.utils.tensorflow.models import RasaModel  # Rasa 模型

# 初始化日志记录器
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True  # 意图分类器组件类型，可训练
)
class ResponseSelector(DIETClassifier):
    """使用监督嵌入的响应选择器。

    响应选择器将用户输入和候选响应嵌入到同一空间中。
    监督嵌入通过最大化它们之间的相似性来训练。
    它还提供未"获胜"响应的排名。

    监督响应选择器需要在管道中前置特征化器。
    该特征化器创建用于嵌入的特征。
    建议使用 ``CountVectorsFeaturizer``，
    可以选择性地在 ``SpacyNLP`` 和 ``SpacyTokenizer`` 之前使用。

    基于 starspace 思想：https://arxiv.org/abs/1709.03856。
    但是，在此实现中，`mu` 参数的处理方式不同，
    并且添加了额外的隐藏层和丢弃。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """在此组件之前应包含在管道中的组件。
        
        Returns:
            必需的组件类型列表
        """
        return [Featurizer]  # 需要特征化器组件

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """组件的默认配置（完整文档字符串请参见父类）。
        
        Returns:
            默认配置字典
        """
        return {
            **DIETClassifier.get_default_config(),  # 继承 DIET 分类器配置
            # ## 使用的神经网络架构
            # 用户消息和标签的嵌入层之前的隐藏层大小
            # 隐藏层数量等于对应列表的长度
            HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: [256, 128]},  # 隐藏层大小
            # 是否在输入词和响应之间共享隐藏层权重
            SHARE_HIDDEN_LAYERS: False,  # 共享隐藏层
            # 转换器中的单元数
            TRANSFORMER_SIZE: None,  # 转换器大小
            # 转换器层数
            NUM_TRANSFORMER_LAYERS: 0,  # 转换器层数
            # 转换器中注意力头数
            NUM_HEADS: 4,  # 注意力头数
            # 如果为 'True'，在注意力中使用键相对嵌入
            KEY_RELATIVE_ATTENTION: False,  # 键相对注意力
            # 如果为 'True'，在注意力中使用值相对嵌入
            VALUE_RELATIVE_ATTENTION: False,  # 值相对注意力
            # 相对嵌入的最大位置。仅在键或值相对注意力开启时生效
            MAX_RELATIVE_POSITION: 5,  # 最大相对位置
            # 使用单向或双向编码器
            UNIDIRECTIONAL_ENCODER: False,  # 单向编码器
            # ## 训练参数
            # 初始和最终批次大小：
            # 批次大小将在每个轮次线性增加
            BATCH_SIZES: [64, 256],  # 批次大小
            # 创建批次时使用的策略
            # 可以是 'sequence' 或 'balanced'
            BATCH_STRATEGY: BALANCED,  # 批次策略
            # 训练的轮次数
            EPOCHS: 300,  # 训练轮数
            # 设置随机种子为任何 'int' 以获得可重现的结果
            RANDOM_SEED: None,  # 随机种子
            # 优化器的初始学习率
            LEARNING_RATE: 0.001,  # 学习率
            # ## 嵌入参数
            # 嵌入向量的维度大小
            EMBEDDING_DIMENSION: 20,  # 嵌入维度
            # 如果没有密集特征时使用的默认密集维度
            DENSE_DIMENSION: {TEXT: 512, LABEL: 512},  # 密集维度
            # 用于连接序列和句子特征的默认维度
            CONCAT_DIMENSION: {TEXT: 512, LABEL: 512},  # 连接维度
            # 错误标签的数量。算法将在训练期间最小化
            # 它们与用户输入的相似性
            NUM_NEG: 20,  # 负样本数量
            # 使用的相似度度量类型，'auto'、'cosine' 或 'inner'
            SIMILARITY_TYPE: AUTO,  # 相似度类型
            # 损失函数的类型，'cross_entropy' 或 'margin'
            LOSS_TYPE: CROSS_ENTROPY,  # 损失类型
            # 应预测置信度的顶级动作数量
            # 如果应报告所有意图的置信度，则设置为 0
            RANKING_LENGTH: 10,  # 排名长度
            # 确定所选顶级动作的置信度是否应重新归一化，
            # 使它们总和为 1。默认情况下，我们不重新归一化
            # 并按原样返回顶级动作的置信度
            # 注意：重新归一化仅在通过 `softmax` 生成置信度时才有意义
            RENORMALIZE_CONFIDENCES: False,  # 重新归一化置信度
            # 指示算法应尝试使正确标签的嵌入向量相似
            # 对于 'cosine' 相似度类型，应为 0.0 < ... < 1.0
            MAX_POS_SIM: 0.8,  # 最大正相似度
            # 错误标签的最大负相似度
            # 对于 'cosine' 相似度类型，应为 -1.0 < ... < 1.0
            MAX_NEG_SIM: -0.4,  # 最大负相似度
            # 如果为 'True'，算法仅最小化错误意图标签上的最大相似度，
            # 仅在 'loss_type' 设置为 'margin' 时使用
            USE_MAX_NEG_SIM: True,  # 使用最大负相似度
            # 与正确预测的置信度成反比地缩放损失
            SCALE_LOSS: True,  # 损失缩放
            # ## 正则化参数
            # 正则化的规模
            REGULARIZATION_CONSTANT: 0.002,  # 正则化常数
            # 内部层中可训练权重的比例
            CONNECTION_DENSITY: 1.0,  # 连接密度
            # 最小化不同标签嵌入之间最大相似度的重要性的规模
            NEGATIVE_MARGIN_SCALE: 0.8,  # 负边距缩放
            # 编码器的丢弃率
            DROP_RATE: 0.2,  # 丢弃率
            # 注意力的丢弃率
            DROP_RATE_ATTENTION: 0,  # 注意力丢弃率
            # 如果为 'True'，对稀疏输入张量应用丢弃
            SPARSE_INPUT_DROPOUT: False,  # 稀疏输入丢弃
            # 如果为 'True'，对密集输入张量应用丢弃
            DENSE_INPUT_DROPOUT: False,  # 密集输入丢弃
            # ## 评估参数
            # 计算验证准确性的频率
            # 小值可能损害性能，例如模型准确性
            EVAL_NUM_EPOCHS: 20,  # 评估轮数
            # 用于保留验证集的示例数量
            # 大值可能损害性能，例如模型准确性
            EVAL_NUM_EXAMPLES: 0,  # 评估示例数量
            # ## 选择器配置
            # 如果为 'True'，输入消息的随机标记将被掩码，
            # 模型应预测这些标记
            MASKED_LM: False,  # 掩码语言模型
            # 此响应选择器要训练的意图名称
            RETRIEVAL_INTENT: None,  # 检索意图
            # 布尔标志，检查响应的实际文本是否应
            # 用作训练模型的基本真实标签
            USE_TEXT_AS_LABEL: False,  # 使用文本作为标签
            # 如果要使用 tensorboard 可视化训练和验证指标，
            # 请将此选项设置为有效的输出目录
            TENSORBOARD_LOG_DIR: None,  # TensorBoard 日志目录
            # 定义何时记录 tensorboard 的训练指标
            # 在每个轮次后或每个训练步骤后
            # 有效值：'epoch' 和 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",  # TensorBoard 日志级别
            # 指定用作序列和句子特征的特征
            # 默认使用管道中的所有特征
            FEATURIZERS: [],  # 特征化器
            # 执行模型检查点
            CHECKPOINT_MODEL: False,  # 检查点模型
            # 如果为 'True'，对所有相似度项应用 sigmoid 并将其
            # 添加到损失函数中，以确保相似度值近似有界
            # 仅在交叉熵损失内部使用
            CONSTRAIN_SIMILARITIES: False,  # 约束相似度
            # 推理期间返回的模型置信度。目前，唯一
            # 可能的值是 `softmax`
            MODEL_CONFIDENCE: SOFTMAX,  # 模型置信度
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        all_retrieval_intents: Optional[List[Text]] = None,
        responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """使用默认值声明实例变量。

        Args:
            config: 组件的配置
            model_storage: 图组件可用于持久化和加载自己的存储
            resource: 此组件的资源定位器，可用于从 `model_storage` 持久化和加载自身
            execution_context: 关于当前图运行的信息
            index_label_id_mapping: 用于编码的标签和索引之间的映射
            entity_tag_specs: 所有实体标签的格式规范
            model: 模型架构
            all_retrieval_intents: 数据中定义的所有检索意图
            responses: 数据中定义的所有响应
            finetune_mode: 如果为 `True`，使用预训练权重加载模型，
                否则使用随机权重初始化
            sparse_feature_sizes: 模型训练的稀疏特征的大小
        """
        component_config = config  # 组件配置

        # 以下属性不能为 ResponseSelector 调整
        component_config[INTENT_CLASSIFICATION] = True  # 意图分类
        component_config[ENTITY_RECOGNITION] = False  # 实体识别
        component_config[BILOU_FLAG] = None  # BILOU 标志

        # 初始化默认值
        self.responses = responses or {}  # 响应字典
        self.all_retrieval_intents = all_retrieval_intents or []  # 所有检索意图
        self.retrieval_intent = None  # 检索意图
        self.use_text_as_label = False  # 使用文本作为标签

        # 调用父类初始化
        super().__init__(
            component_config,
            model_storage,
            resource,
            execution_context,
            index_label_id_mapping,
            entity_tag_specs,
            model,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @property
    def label_key(self) -> Text:
        """Returns label key."""
        return LABEL_KEY

    @property
    def label_sub_key(self) -> Text:
        """Returns label sub_key."""
        return LABEL_SUB_KEY

    @staticmethod
    def model_class(  # type: ignore[override]
        use_text_as_label: bool,
    ) -> Type[RasaModel]:
        """Returns model class."""
        if use_text_as_label:
            return DIET2DIET
        else:
            return DIET2BOW

    def _load_selector_params(self) -> None:
        self.retrieval_intent = self.component_config[RETRIEVAL_INTENT]
        self.use_text_as_label = self.component_config[USE_TEXT_AS_LABEL]

    def _warn_about_transformer_and_hidden_layers_enabled(
        self, selector_name: Text
    ) -> None:
        """Warns user if they enabled the transformer but didn't disable hidden layers.

        ResponseSelector defaults specify considerable hidden layer sizes, but
        this is for cases where no transformer is used. If a transformer exists,
        then, from our experience, the best results are achieved with no hidden layers
        used between the feature-combining layers and the transformer.
        """
        default_config = self.get_default_config()
        hidden_layers_is_at_default_value = (
            self.component_config[HIDDEN_LAYERS_SIZES]
            == default_config[HIDDEN_LAYERS_SIZES]
        )
        config_for_disabling_hidden_layers: Dict[Text, List[Any]] = {
            k: [] for k, _ in default_config[HIDDEN_LAYERS_SIZES].items()
        }
        # warn if the hidden layers aren't disabled
        if (
            self.component_config[HIDDEN_LAYERS_SIZES]
            != config_for_disabling_hidden_layers
        ):
            # make the warning text more contextual by explaining what the user did
            # to the hidden layers' config (i.e. what it is they should change)
            if hidden_layers_is_at_default_value:
                what_user_did = "left the hidden layer sizes at their default value:"
            else:
                what_user_did = "set the hidden layer sizes to be non-empty by setting"

            rasa.shared.utils.io.raise_warning(
                f"You have enabled a transformer inside {selector_name} by"
                f" setting a positive value for `{NUM_TRANSFORMER_LAYERS}`, but you "
                f"{what_user_did} `{HIDDEN_LAYERS_SIZES}="
                f"{self.component_config[HIDDEN_LAYERS_SIZES]}`. We recommend to "
                f"disable the hidden layers when using a transformer, by specifying "
                f"`{HIDDEN_LAYERS_SIZES}={config_for_disabling_hidden_layers}`.",
                category=UserWarning,
            )

    def _warn_and_correct_transformer_size(self, selector_name: Text) -> None:
        """Corrects transformer size so that training doesn't break; informs the user.

        If a transformer is used, the default `transformer_size` breaks things.
        We need to set a reasonable default value so that the model works fine.
        """
        if (
            self.component_config[TRANSFORMER_SIZE] is None
            or self.component_config[TRANSFORMER_SIZE] < 1
        ):
            rasa.shared.utils.io.raise_warning(
                f"`{TRANSFORMER_SIZE}` is set to "
                f"`{self.component_config[TRANSFORMER_SIZE]}` for "
                f"{selector_name}, but a positive size is required when using "
                f"`{NUM_TRANSFORMER_LAYERS} > 0`. {selector_name} will proceed, using "
                f"`{TRANSFORMER_SIZE}={DEFAULT_TRANSFORMER_SIZE}`. "
                f"Alternatively, specify a different value in the component's config.",
                category=UserWarning,
            )
            self.component_config[TRANSFORMER_SIZE] = DEFAULT_TRANSFORMER_SIZE

    def _check_config_params_when_transformer_enabled(self) -> None:
        """Checks & corrects config parameters when the transformer is enabled.

        This is needed because the defaults for individual config parameters are
        interdependent and some defaults should change when the transformer is enabled.
        """
        if self.component_config[NUM_TRANSFORMER_LAYERS] > 0:
            selector_name = "ResponseSelector" + (
                f"({self.retrieval_intent})" if self.retrieval_intent else ""
            )
            self._warn_about_transformer_and_hidden_layers_enabled(selector_name)
            self._warn_and_correct_transformer_size(selector_name)

    def _check_config_parameters(self) -> None:
        """Checks that component configuration makes sense; corrects it where needed."""
        super()._check_config_parameters()
        self._load_selector_params()
        # Once general DIET-related parameters have been checked, check also the ones
        # specific to ResponseSelector.
        self._check_config_params_when_transformer_enabled()

    def _set_message_property(
        self, message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:
        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[
            RESPONSE_SELECTOR_RETRIEVAL_INTENTS
        ] = self.all_retrieval_intents
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.

        Args:
            training_data: training data to preprocessed.
        """
        # Collect all retrieval intents present in the data before filtering
        self.all_retrieval_intents = list(training_data.retrieval_intents)

        if self.retrieval_intent:
            training_data = training_data.filter_training_examples(
                lambda ex: self.retrieval_intent == ex.get(INTENT)
            )
        else:
            # retrieval intent was left to its default value
            logger.info(
                "Retrieval intent parameter was left to its default value. This "
                "response selector will be trained on training examples combining "
                "all retrieval intents."
            )

        label_attribute = RESPONSE if self.use_text_as_label else INTENT_RESPONSE_KEY

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=label_attribute
        )

        self.responses = training_data.responses

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=label_attribute
        )

        model_data = self._create_model_data(
            training_data.intent_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    def _resolve_intent_response_key(
        self, label: Dict[Text, Optional[Text]]
    ) -> Optional[Text]:
        """Given a label, return the response key based on the label id.

        Args:
            label: predicted label by the selector

        Returns:
            The match for the label that was found in the known responses.
            It is always guaranteed to have a match, otherwise that case should have
            been caught earlier and a warning should have been raised.
        """
        for key, responses in self.responses.items():

            # First check if the predicted label was the key itself
            search_key = util.template_key_to_intent_response_key(key)
            if search_key == label.get("name"):
                return search_key

            # Otherwise loop over the responses to check if the text has a direct match
            for response in responses:
                if response.get(TEXT, "") == label.get("name"):
                    return search_key
        return None

    def process(self, messages: List[Message]) -> List[Message]:
        """Selects most like response for message.

        Args:
            messages: List containing latest user message.

        Returns:
            List containing the message augmented with the most likely response,
            the associated intent_response_key and its similarity to the input.
        """
        for message in messages:
            out = self._predict(message)
            top_label, label_ranking = self._predict_label(out)

            # Get the exact intent_response_key and the associated
            # responses for the top predicted label
            label_intent_response_key = (
                self._resolve_intent_response_key(top_label)
                or top_label[INTENT_NAME_KEY]
            )
            label_responses = self.responses.get(
                util.intent_response_key_to_template_key(label_intent_response_key)
            )

            if label_intent_response_key and not label_responses:
                # responses seem to be unavailable,
                # likely an issue with the training data
                # we'll use a fallback instead
                rasa.shared.utils.io.raise_warning(
                    f"Unable to fetch responses for {label_intent_response_key} "
                    f"This means that there is likely an issue with the training data."
                    f"Please make sure you have added responses for this intent."
                )
                label_responses = [{TEXT: label_intent_response_key}]

            for label in label_ranking:
                label[INTENT_RESPONSE_KEY] = (
                    self._resolve_intent_response_key(label) or label[INTENT_NAME_KEY]
                )
                # Remove the "name" key since it is either the same as
                # "intent_response_key" or it is the response text which
                # is not needed in the ranking.
                label.pop(INTENT_NAME_KEY)

            selector_key = (
                self.retrieval_intent
                if self.retrieval_intent
                else RESPONSE_SELECTOR_DEFAULT_INTENT
            )

            logger.debug(
                f"Adding following selector key to message property: {selector_key}"
            )

            utter_action_key = util.intent_response_key_to_template_key(
                label_intent_response_key
            )
            prediction_dict = {
                RESPONSE_SELECTOR_PREDICTION_KEY: {
                    RESPONSE_SELECTOR_RESPONSES_KEY: label_responses,
                    PREDICTED_CONFIDENCE_KEY: top_label[PREDICTED_CONFIDENCE_KEY],
                    INTENT_RESPONSE_KEY: label_intent_response_key,
                    RESPONSE_SELECTOR_UTTER_ACTION_KEY: utter_action_key,
                },
                RESPONSE_SELECTOR_RANKING_KEY: label_ranking,
            }

            self._set_message_property(message, prediction_dict, selector_key)

            if (
                self._execution_context.should_add_diagnostic_data
                and out
                and DIAGNOSTIC_DATA in out
            ):
                message.add_diagnostic_data(
                    self._execution_context.node_name, out.get(DIAGNOSTIC_DATA)
                )

        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        if self.model is None:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.responses.json", self.responses
            )

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.retrieval_intents.json",
                self.all_retrieval_intents,
            )

        super().persist()

    @classmethod
    def _load_model_class(
        cls,
        tf_model_file: Text,
        model_data_example: RasaModelData,
        label_data: RasaModelData,
        entity_tag_specs: List[EntityTagSpec],
        config: Dict[Text, Any],
        finetune_mode: bool = False,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )
        return cls.model_class(config[USE_TEXT_AS_LABEL]).load(
            tf_model_file,
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=copy.deepcopy(config),
            finetune_mode=finetune_mode,
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        return self.model_class(self.use_text_as_label)(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> ResponseSelector:
        """Loads the trained model from the provided directory."""
        model = super().load(
            config, model_storage, resource, execution_context, **kwargs
        )

        try:
            with model_storage.read_from(resource) as model_path:
                file_name = cls.__name__
                responses = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.responses.json"
                )
                all_retrieval_intents = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.retrieval_intents.json"
                )
                model.responses = responses
                model.all_retrieval_intents = all_retrieval_intents
                return model
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)


class DIET2BOW(DIET):
    """DIET2BOW transformer implementation."""

    def _create_metrics(self) -> None:
        # self.metrics preserve order
        # output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")
        # output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")

        self.metrics_to_log.append("r_acc")
        if debug_log_level:
            self.metrics_to_log.append("r_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {"t": "total", "m": "mask", "r": "response"}
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.response_loss.update_state(loss)
        self.response_acc.update_state(acc)


class DIET2DIET(DIET):
    """Diet 2 Diet transformer implementation."""

    def _check_data(self) -> None:
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise InvalidConfigException(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if (
            self.config[SHARE_HIDDEN_LAYERS]
            and self.data_signature[TEXT][SENTENCE]
            != self.data_signature[LABEL][SENTENCE]
        ):
            raise ValueError(
                "If hidden layer weights are shared, data signatures "
                "for text_features and label_features must coincide."
            )

    def _create_metrics(self) -> None:
        # self.metrics preserve order
        # output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")
        # output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")

        self.metrics_to_log.append("r_acc")
        if debug_log_level:
            self.metrics_to_log.append("r_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {"t": "total", "m": "mask", "r": "response"}
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        self.text_name = TEXT
        self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

        # For user text and response text, prepare layers that combine different feature
        # types, embed everything using a transformer and optionally also do masked
        # language modeling. Omit input dropout for label features.
        label_config = self.config.copy()
        label_config.update({SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False})
        for attribute, config in [
            (self.text_name, self.config),
            (self.label_name, label_config),
        ]:
            self._tf_layers[
                f"sequence_layer.{attribute}"
            ] = rasa_layers.RasaSequenceLayer(
                attribute, self.data_signature[attribute], config
            )

        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        self._prepare_label_classification_layers(predictor_attribute=self.text_name)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        # Combine all feature types into one and embed using a transformer.
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                self.tf_label_data[LABEL][SEQUENCE],
                self.tf_label_data[LABEL][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        # Last token is taken from the last position with real features, determined
        # - by the number of real tokens, i.e. by the sequence length of sequence-level
        #   features, and
        # - by the presence or absence of sentence-level features (reflected in the
        #   effective sequence length of these features being 1 or 0.
        # We need to combine the two lengths to correctly get the last position.
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            self.tf_label_data, LABEL
        )
        sentence_label = self._last_token(
            label_transformed, sequence_feature_lengths + sentence_feature_lengths
        )

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](sentence_label)

        return all_label_ids, all_labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        # Process all features for text.
        sequence_feature_lengths_text = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        (
            text_transformed,
            text_in,
            _,
            text_seq_ids,
            mlm_mask_booleanean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths_text,
            ),
            training=self._training,
        )

        # Process all features for labels.
        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                tf_batch_data[LABEL][SEQUENCE],
                tf_batch_data[LABEL][SENTENCE],
                sequence_feature_lengths_label,
            ),
            training=self._training,
        )

        losses = []

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed,
                text_in,
                text_seq_ids,
                mlm_mask_booleanean_text,
                self.text_name,
            )

            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        # Get sentence feature vector for label classification. The vector is extracted
        # from the last position with real features. To determine this position, we
        # combine the sequence lengths of sequence- and sentence-level features.
        sentence_feature_lengths_text = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )
        sentence_vector_text = self._last_token(
            text_transformed,
            sequence_feature_lengths_text + sentence_feature_lengths_text,
        )

        # Extract sentence vector for the label attribute in the same way.
        sentence_feature_lengths_label = self._get_sentence_feature_lengths(
            tf_batch_data, LABEL
        )
        sentence_vector_label = self._last_token(
            label_transformed,
            sequence_feature_lengths_label + sentence_feature_lengths_label,
        )
        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]

        loss, acc = self._calculate_label_loss(
            sentence_vector_text, sentence_vector_label, label_ids
        )
        self.response_loss.update_state(loss)
        self.response_acc.update_state(acc)
        losses.append(loss)

        return tf.math.add_n(losses)

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels()

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(text_transformed, sequence_feature_lengths)
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )
        predictions["i_scores"] = scores

        return predictions
