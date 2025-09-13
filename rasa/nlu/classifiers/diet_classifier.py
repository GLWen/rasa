# =============================================================================
# Rasa NLU DIET Classifier 双意图和实体转换器分类器模块
# 本模块实现了 DIET (Dual Intent and Entity Transformer) 分类器，
# 用于同时进行意图分类和实体提取的多任务学习模型
# =============================================================================

# 导入未来版本注解支持
from __future__ import annotations

# 导入标准库模块
import copy  # 深拷贝功能
import logging  # 日志记录
from collections import defaultdict  # 默认字典
from pathlib import Path  # 路径处理
from typing import Any, Dict, List, Optional, Text, Tuple, Union, TypeVar, Type  # 类型注解

# 导入科学计算库
import numpy as np  # 数值计算
import scipy.sparse  # 稀疏矩阵
import tensorflow as tf  # 深度学习框架

# 导入 Rasa 核心模块
from rasa.exceptions import ModelNotFound  # 模型未找到异常
from rasa.nlu.featurizers.featurizer import Featurizer  # 特征化器基类
from rasa.engine.graph import ExecutionContext, GraphComponent  # 图执行上下文和组件
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方
from rasa.engine.storage.resource import Resource  # 资源管理
from rasa.engine.storage.storage import ModelStorage  # 模型存储
from rasa.nlu.extractors.extractor import EntityExtractorMixin  # 实体提取器混入
from rasa.nlu.classifiers.classifier import IntentClassifier  # 意图分类器基类
import rasa.shared.utils.io  # 共享工具模块
import rasa.nlu.utils.bilou_utils as bilou_utils  # BILOU 标签工具
from rasa.shared.constants import DIAGNOSTIC_DATA  # 诊断数据常量
from rasa.nlu.extractors.extractor import EntityTagSpec  # 实体标签规范
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH  # 标签排名长度
from rasa.utils import train_utils  # 训练工具
from rasa.utils.tensorflow import rasa_layers  # Rasa TensorFlow 层
from rasa.utils.tensorflow.feature_array import (
    FeatureArray,  # 特征数组
    serialize_nested_feature_arrays,  # 序列化嵌套特征数组
    deserialize_nested_feature_arrays,  # 反序列化嵌套特征数组
)
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel  # Rasa 模型基类
from rasa.utils.tensorflow.model_data import (
    RasaModelData,  # Rasa 模型数据
    FeatureSignature,  # 特征签名
)
from rasa.nlu.constants import TOKENS_NAMES, DEFAULT_TRANSFORMER_SIZE  # NLU 常量

# 导入 NLU 常量
from rasa.shared.nlu.constants import (
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,  # 按逗号分割实体的默认值
    TEXT,  # 文本常量
    INTENT,  # 意图常量
    INTENT_RESPONSE_KEY,  # 意图响应键
    ENTITIES,  # 实体常量
    ENTITY_ATTRIBUTE_TYPE,  # 实体属性类型
    ENTITY_ATTRIBUTE_GROUP,  # 实体属性组
    ENTITY_ATTRIBUTE_ROLE,  # 实体属性角色
    NO_ENTITY_TAG,  # 非实体标签
    SPLIT_ENTITIES_BY_COMMA,  # 按逗号分割实体
)

# 导入异常和训练数据
from rasa.shared.exceptions import InvalidConfigException  # 无效配置异常
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据
from rasa.shared.nlu.training_data.message import Message  # 消息类

# 导入 TensorFlow 常量
from rasa.utils.tensorflow.constants import (
    DROP_SMALL_LAST_BATCH,  # 丢弃小批次
    LABEL,  # 标签
    IDS,  # ID
    HIDDEN_LAYERS_SIZES,  # 隐藏层大小
    RENORMALIZE_CONFIDENCES,  # 重新归一化置信度
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
    LOSS_TYPE,  # 损失类型
    SIMILARITY_TYPE,  # 相似度类型
    NUM_NEG,  # 负样本数
    SPARSE_INPUT_DROPOUT,  # 稀疏输入丢弃
    DENSE_INPUT_DROPOUT,  # 密集输入丢弃
    MASKED_LM,  # 掩码语言模型
    ENTITY_RECOGNITION,  # 实体识别
    TENSORBOARD_LOG_DIR,  # TensorBoard 日志目录
    INTENT_CLASSIFICATION,  # 意图分类
    EVAL_NUM_EXAMPLES,  # 评估样本数
    EVAL_NUM_EPOCHS,  # 评估轮数
    UNIDIRECTIONAL_ENCODER,  # 单向编码器
    DROP_RATE,  # 丢弃率
    DROP_RATE_ATTENTION,  # 注意力丢弃率
    CONNECTION_DENSITY,  # 连接密度
    NEGATIVE_MARGIN_SCALE,  # 负边距缩放
    REGULARIZATION_CONSTANT,  # 正则化常数
    SCALE_LOSS,  # 缩放损失
    USE_MAX_NEG_SIM,  # 使用最大负相似度
    MAX_NEG_SIM,  # 最大负相似度
    MAX_POS_SIM,  # 最大正相似度
    EMBEDDING_DIMENSION,  # 嵌入维度
    BILOU_FLAG,  # BILOU 标志
    KEY_RELATIVE_ATTENTION,  # 键相对注意力
    VALUE_RELATIVE_ATTENTION,  # 值相对注意力
    MAX_RELATIVE_POSITION,  # 最大相对位置
    AUTO,  # 自动
    BALANCED,  # 平衡
    CROSS_ENTROPY,  # 交叉熵
    TENSORBOARD_LOG_LEVEL,  # TensorBoard 日志级别
    CONCAT_DIMENSION,  # 连接维度
    FEATURIZERS,  # 特征化器
    CHECKPOINT_MODEL,  # 检查点模型
    SEQUENCE,  # 序列
    SENTENCE,  # 句子
    SEQUENCE_LENGTH,  # 序列长度
    DENSE_DIMENSION,  # 密集维度
    MASK,  # 掩码
    CONSTRAIN_SIMILARITIES,  # 约束相似度
    MODEL_CONFIDENCE,  # 模型置信度
    SOFTMAX,  # Softmax
    RUN_EAGERLY,  # 急切运行
)

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 特征类型常量
SPARSE = "sparse"  # 稀疏特征
DENSE = "dense"    # 密集特征

# 标签相关常量
LABEL_KEY = LABEL      # 标签键
LABEL_SUB_KEY = IDS    # 标签子键

# 可能的实体标签类型
POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]

# DIET 分类器类型变量
DIETClassifierT = TypeVar("DIETClassifierT", bound="DIETClassifier")


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,  # 意图分类器组件类型
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,   # 实体提取器组件类型
    ],
    is_trainable=True,  # 可训练组件
)
class DIETClassifier(GraphComponent, IntentClassifier, EntityExtractorMixin):
    """用于意图分类和实体提取的多任务模型。

    DIET 是双意图和实体转换器（Dual Intent and Entity Transformer）。
    该架构基于一个转换器，该转换器在两个任务之间共享。
    通过条件随机场（CRF）标记层在转换器输出序列上预测实体标签序列，
    该序列对应于输入标记序列。``__CLS__`` 标记的转换器输出和意图标签
    被嵌入到单个语义向量空间中。我们使用点积损失来最大化与目标标签的
    相似性并最小化与负样本的相似性。
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
        # 更改默认参数时请确保更新文档
        return {
            # ## 使用的神经网络架构
            # 用户消息和标签的嵌入层之前的隐藏层大小
            # 隐藏层数等于对应列表的长度
            HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},  # 隐藏层大小
            # 是否在用户消息和标签之间共享隐藏层权重
            SHARE_HIDDEN_LAYERS: False,  # 共享隐藏层
            # 转换器中的单元数
            TRANSFORMER_SIZE: DEFAULT_TRANSFORMER_SIZE,  # 转换器大小
            # 转换器层数
            NUM_TRANSFORMER_LAYERS: 2,  # 转换器层数
            # 转换器中的注意力头数
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
            # 训练的轮数
            EPOCHS: 300,  # 训练轮数
            # 设置随机种子为任何 'int' 以获得可重现的结果
            RANDOM_SEED: None,  # 随机种子
            # 优化器的初始学习率
            LEARNING_RATE: 0.001,  # 学习率
            # ## 嵌入参数
            # 嵌入向量的维度大小
            EMBEDDING_DIMENSION: 20,  # 嵌入维度
            # 用于稀疏特征的密集维度
            DENSE_DIMENSION: {TEXT: 128, LABEL: 20},  # 密集维度
            # 用于连接序列和句子特征的默认维度
            CONCAT_DIMENSION: {TEXT: 128, LABEL: 20},  # 连接维度
            # 错误标签的数量。算法将在训练期间最小化它们与用户输入的相似性
            NUM_NEG: 20,  # 负样本数
            # 使用的相似性度量类型，可以是 'auto'、'cosine' 或 'inner'
            SIMILARITY_TYPE: AUTO,  # 相似度类型
            # 损失函数的类型，可以是 'cross_entropy' 或 'margin'
            LOSS_TYPE: CROSS_ENTROPY,  # 损失类型
            # 应报告置信度的顶级意图数量
            # 如果应报告所有意图的置信度，则设置为 0
            RANKING_LENGTH: LABEL_RANKING_LENGTH,  # 排名长度
            # 指示算法应尝试使正确标签的嵌入向量有多相似
            # 对于 'cosine' 相似度类型，应为 0.0 < ... < 1.0
            MAX_POS_SIM: 0.8,  # 最大正相似度
            # 错误标签的最大负相似度
            # 对于 'cosine' 相似度类型，应为 -1.0 < ... < 1.0
            MAX_NEG_SIM: -0.4,  # 最大负相似度
            # 如果为 'True'，算法仅最小化错误意图标签上的最大相似度，
            # 仅在 'loss_type' 设置为 'margin' 时使用
            USE_MAX_NEG_SIM: True,  # 使用最大负相似度
            # 如果为 'True'，按正确预测的置信度反比例缩放损失
            SCALE_LOSS: False,  # 缩放损失
            # ## 正则化参数
            # 正则化的规模
            REGULARIZATION_CONSTANT: 0.002,  # 正则化常数
            # 最小化不同标签嵌入之间最大相似度的重要性规模，
            # 仅在 'loss_type' 设置为 'margin' 时使用
            NEGATIVE_MARGIN_SCALE: 0.8,  # 负边距缩放
            # 编码器的丢弃率
            DROP_RATE: 0.2,  # 丢弃率
            # 注意力的丢弃率
            DROP_RATE_ATTENTION: 0,  # 注意力丢弃率
            # 内部层中可训练权重的比例
            CONNECTION_DENSITY: 0.2,  # 连接密度
            # 如果为 'True'，对稀疏输入张量应用丢弃
            SPARSE_INPUT_DROPOUT: True,  # 稀疏输入丢弃
            # 如果为 'True'，对密集输入张量应用丢弃
            DENSE_INPUT_DROPOUT: True,  # 密集输入丢弃
            # ## 评估参数
            # 计算验证准确性的频率
            # 小值可能会损害性能
            EVAL_NUM_EPOCHS: 20,  # 评估轮数
            # 用于保留验证集的示例数量
            # 大值可能会损害性能，例如模型准确性
            # 设置为 0 表示无验证
            EVAL_NUM_EXAMPLES: 0,  # 评估样本数
            # ## 模型配置
            # 如果为 'True'，训练意图分类并预测意图
            INTENT_CLASSIFICATION: True,  # 意图分类
            # 如果为 'True'，训练命名实体识别并预测实体
            ENTITY_RECOGNITION: True,  # 实体识别
            # 如果为 'True'，输入消息的随机标记将被掩码，模型应预测这些标记
            MASKED_LM: False,  # 掩码语言模型
            # 'BILOU_flag' 确定是否使用 BILOU 标记
            # 如果设置为 'True'，标记更严格，但每个实体需要更多示例
            # 经验法则：每个实体应该有超过 100 个示例
            BILOU_FLAG: True,  # BILOU 标志
            # 如果要使用 tensorboard 可视化训练和验证指标，
            # 请将此选项设置为有效的输出目录
            TENSORBOARD_LOG_DIR: None,  # TensorBoard 日志目录
            # 定义何时记录 tensorboard 的训练指标
            # 可以在每个轮次后或每个训练步骤后
            # 有效值：'epoch' 和 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",  # TensorBoard 日志级别
            # 执行模型检查点
            CHECKPOINT_MODEL: False,  # 检查点模型
            # 指定用作序列和句子特征的特征
            # 默认使用管道中的所有特征
            FEATURIZERS: [],  # 特征化器
            # 按逗号分割实体，这对于成分列表等有意义，
            # 但对于地址的各个部分没有意义
            SPLIT_ENTITIES_BY_COMMA: True,  # 按逗号分割实体
            # 如果为 'True'，对所有相似性项应用 sigmoid 并将其添加到损失函数中，
            # 以确保相似性值近似有界。仅在交叉熵损失内部使用
            CONSTRAIN_SIMILARITIES: False,  # 约束相似度
            # 推理期间返回的模型置信度。目前唯一可能的值是 `softmax`
            MODEL_CONFIDENCE: SOFTMAX,  # 模型置信度
            # 确定所选顶级意图的置信度是否应重新归一化，使其总和为 1
            # 默认情况下，我们不重新归一化，按原样返回顶级意图的置信度
            # 注意：重新归一化仅在通过 `softmax` 生成置信度时才有意义
            RENORMALIZE_CONFIDENCES: False,  # 重新归一化置信度
            # 确定是否构建模型图
            # 当模型仅训练或推断几个步骤时，这是有利的，
            # 因为图的编译往往比运行它花费更多时间
            # 建议不要调整优化参数
            RUN_EAGERLY: False,  # 急切运行
            # 确定如果最后一个批次包含少于一半批次大小的示例是否应丢弃
            DROP_SMALL_LAST_BATCH: False,  # 丢弃小批次
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
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """使用默认值声明实例变量。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储
            resource: 资源
            execution_context: 执行上下文
            index_label_id_mapping: 索引到标签ID的映射
            entity_tag_specs: 实体标签规范列表
            model: Rasa模型实例
            sparse_feature_sizes: 稀疏特征大小
        """
        # 检查是否配置了训练轮数
        if EPOCHS not in config:
            rasa.shared.utils.io.raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        # 设置基本属性
        self.component_config = config  # 组件配置
        self._model_storage = model_storage  # 模型存储
        self._resource = resource  # 资源
        self._execution_context = execution_context  # 执行上下文

        # 检查配置参数
        self._check_config_parameters()

        # 将数字转换为标签
        self.index_label_id_mapping = index_label_id_mapping or {}  # 索引到标签ID的映射

        self._entity_tag_specs = entity_tag_specs  # 实体标签规范

        self.model = model  # 模型实例

        # 设置检查点目录
        self.tmp_checkpoint_dir = None
        if self.component_config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

        # 初始化数据相关属性
        self._label_data: Optional[RasaModelData] = None  # 标签数据
        self._data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None  # 数据示例

        # 初始化实体分割配置
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            self.component_config[SPLIT_ENTITIES_BY_COMMA],
            SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
        )

        # 设置微调模式和稀疏特征大小
        self.finetune_mode = self._execution_context.is_finetuning  # 微调模式
        self._sparse_feature_sizes = sparse_feature_sizes  # 稀疏特征大小

    # init helpers
    def _check_masked_lm(self) -> None:
        if (
            self.component_config[MASKED_LM]
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

    def _check_share_hidden_layers_sizes(self) -> None:
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            first_hidden_layer_sizes = next(
                iter(self.component_config[HIDDEN_LAYERS_SIZES].values())
            )
            # check that all hidden layer sizes are the same
            identical_hidden_layer_sizes = all(
                current_hidden_layer_sizes == first_hidden_layer_sizes
                for current_hidden_layer_sizes in self.component_config[
                    HIDDEN_LAYERS_SIZES
                ].values()
            )
            if not identical_hidden_layer_sizes:
                raise ValueError(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

    def _check_config_parameters(self) -> None:
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        self._check_masked_lm()
        self._check_share_hidden_layers_sizes()

        self.component_config = train_utils.update_confidence_type(
            self.component_config
        )

        train_utils.validate_configuration_settings(self.component_config)

        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )
        self.component_config = train_utils.update_evaluation_parameters(
            self.component_config
        )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DIETClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    @property
    def label_key(self) -> Optional[Text]:
        """Return key if intent classification is activated."""
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @property
    def label_sub_key(self) -> Optional[Text]:
        """Return sub key if intent classification is activated."""
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        return DIET

    # training data helpers:
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary."""
        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _create_entity_tag_specs(
        self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""
        _tag_specs = []

        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,
                        tags_to_ids=tag_id_index_mapping,
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),
                        num_tags=len(tag_id_index_mapping),
                    )
                )

        return _tag_specs

    @staticmethod
    def _tag_id_index_mapping_for(
        tag_name: Text, training_data: TrainingData
    ) -> Optional[Dict[Text, int]]:
        """Create mapping from tag name to id."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            distinct_tags = training_data.entity_roles
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            distinct_tags = training_data.entity_groups
        else:
            distinct_tags = training_data.entities

        distinct_tags = distinct_tags - {NO_ENTITY_TAG} - {None}

        if not distinct_tags:
            return None

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tags), 1)
        }
        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    def _check_labels_features_exist(
        self, labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""
        return all(
            label_example.features_present(
                attribute, self.component_config[FEATURIZERS]
            )
            for label_example in labels_example
        )

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Dict[Text, Union[scipy.sparse.spmatrix, np.ndarray]]:

        (
            sparse_sequence_features,
            sparse_sentence_features,
        ) = message.get_sparse_features(attribute, self.component_config[FEATURIZERS])
        dense_sequence_features, dense_sentence_features = message.get_dense_features(
            attribute, self.component_config[FEATURIZERS]
        )

        if dense_sequence_features is not None and sparse_sequence_features is not None:
            if (
                dense_sequence_features.features.shape[0]
                != sparse_sequence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sequence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )
        if dense_sentence_features is not None and sparse_sentence_features is not None:
            if (
                dense_sentence_features.features.shape[0]
                != sparse_sentence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sentence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute not in [INTENT, INTENT_RESPONSE_KEY]
        ):
            sparse_sequence_features = None
            dense_sequence_features = None

        out = {}

        if sparse_sentence_features is not None:
            out[f"{SPARSE}_{SENTENCE}"] = sparse_sentence_features.features
        if sparse_sequence_features is not None:
            out[f"{SPARSE}_{SEQUENCE}"] = sparse_sequence_features.features
        if dense_sentence_features is not None:
            out[f"{DENSE}_{SENTENCE}"] = dense_sentence_features.features
        if dense_sequence_features is not None:
            out[f"{DENSE}_{SEQUENCE}"] = dense_sequence_features.features

        return out

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            if (0 < num_text_sentence_features != num_label_sentence_features > 0) or (
                0 < num_text_sequence_features != num_label_sequence_features > 0
            ):
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> Tuple[List[FeatureArray], List[FeatureArray]]:
        """Collects precomputed encodings."""
        features = defaultdict(list)

        for e in label_examples:
            label_features = self._extract_features(e, attribute)
            for feature_key, feature_value in label_features.items():
                features[feature_key].append(feature_value)
        sequence_features = []
        sentence_features = []
        for feature_name, feature_value in features.items():
            if SEQUENCE in feature_name:
                sequence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
            else:
                sentence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
        return sequence_features, sentence_features

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[FeatureArray]:
        """Computes one-hot representation for the labels."""
        logger.debug("No label features found. Computing default label features.")

        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [
            FeatureArray(
                np.array([np.expand_dims(a, 0) for a in eye_matrix]),
                number_of_dimensions=3,
            )
        ]

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """
        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            sentence_features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)
        if label_data.does_feature_not_exist(
            LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [
                FeatureArray(
                    np.expand_dims(label_ids, -1),
                    number_of_dimensions=2,
                )
            ],
        )

        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        if self._label_data is None:
            return []

        feature_arrays = self._label_data.get(LABEL, SENTENCE)
        all_label_features = feature_arrays[0]
        return [
            FeatureArray(
                np.array([all_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_label_features.number_of_dimensions,
            )
        ]

    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
        training: bool = True,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object."""
        from rasa.utils.tensorflow import model_data_utils

        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # we don't have any intent labels during prediction, just add them during
            # training
            attributes_to_consider.append(label_attribute)
        if (
            training
            and self.component_config[ENTITY_RECOGNITION]
            and self._entity_tag_specs
        ):
            # Add entities as labels only during training and only if there was
            # training data added for entities with DIET configured to predict entities.
            attributes_to_consider.append(ENTITIES)

        if training and label_attribute is not None:
            # only use those training examples that have the label_attribute set
            # during training
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        training_data = [
            message
            for message in training_data
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]

        if not training_data:
            # no training data are present to train
            return RasaModelData()

        (
            features_for_examples,
            sparse_feature_sizes,
        ) = model_data_utils.featurize_training_examples(
            training_data,
            attributes_to_consider,
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
        )
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )

        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )
        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        # Current implementation doesn't yet account for updating sparse
        # feature sizes of label attributes. That's why we remove them.
        sparse_feature_sizes = self._remove_label_sparse_feature_sizes(
            sparse_feature_sizes=sparse_feature_sizes, label_attribute=label_attribute
        )
        model_data.add_sparse_feature_sizes(sparse_feature_sizes)

        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data

    @staticmethod
    def _remove_label_sparse_feature_sizes(
        sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        label_attribute: Optional[Text] = None,
    ) -> Dict[Text, Dict[Text, List[int]]]:

        if label_attribute in sparse_feature_sizes:
            del sparse_feature_sizes[label_attribute]
        return sparse_feature_sizes

    def _add_label_features(
        self,
        model_data: RasaModelData,
        training_data: List[Message],
        label_attribute: Text,
        label_id_dict: Dict[Text, int],
        training: bool = True,
    ) -> None:
        label_ids = []
        if training and self.component_config[INTENT_CLASSIFICATION]:
            for example in training_data:
                if example.get(label_attribute):
                    label_ids.append(label_id_dict[example.get(label_attribute)])
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [
                    FeatureArray(
                        np.expand_dims(label_ids, -1),
                        number_of_dimensions=2,
                    )
                ],
            )

        if (
            label_attribute
            and model_data.does_feature_not_exist(label_attribute, SENTENCE)
            and model_data.does_feature_not_exist(label_attribute, SEQUENCE)
        ):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL, SENTENCE, self._use_default_label_features(np.array(label_ids))
            )

        # as label_attribute can have different values, e.g. INTENT or RESPONSE,
        # copy over the features to the LABEL key to make
        # it easier to access the label features inside the model itself
        model_data.update_key(label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(label_attribute, MASK, LABEL, MASK)

        model_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        if (
            self.component_config[BILOU_FLAG]
            and self.component_config[ENTITY_RECOGNITION]
        ):
            bilou_utils.apply_bilou_schema(training_data)

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )

        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )
        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_KEY, LABEL_SUB_KEY))) >= 2

    def train(self, training_data: TrainingData) -> Resource:
        """在数据集上训练嵌入意图分类器。
        
        Args:
            training_data: 训练数据
            
        Returns:
            资源对象
        """
        # 预处理训练数据
        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return self._resource

        # 检查微调模式
        if not self.model and self.finetune_mode:
            raise rasa.shared.exceptions.InvalidParameterException(
                f"{self.__class__.__name__} was instantiated "
                f"with `model=None` and `finetune_mode=True`. "
                f"This is not a valid combination as the component "
                f"needs an already instantiated and trained model "
                f"to continue training in finetune mode."
            )

        # 检查意图分类
        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return self._resource
                
        # 检查实体识别
        if self.component_config.get(ENTITY_RECOGNITION):
            self.check_correct_entity_annotations(training_data)

        # 保存一个示例用于持久化和加载
        self._data_example = model_data.first_data_example()

        if not self.finetune_mode:
            # 没有预训练模型可加载。创建模型的新实例
            self.model = self._instantiate_model_class(model_data)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    self.component_config[LEARNING_RATE]  # 学习率
                ),
                run_eagerly=self.component_config[RUN_EAGERLY],  # 急切运行
            )
        else:
            if self.model is None:
                raise ModelNotFound("Model could not be found. ")

            # 调整模型以进行增量训练
            self.model.adjust_for_incremental_training(
                data_example=self._data_example,
                new_sparse_feature_sizes=model_data.get_sparse_feature_sizes(),
                old_sparse_feature_sizes=self._sparse_feature_sizes,
            )
        self._sparse_feature_sizes = model_data.get_sparse_feature_sizes()

        # 创建数据生成器
        data_generator, validation_data_generator = train_utils.create_data_generators(
            model_data,
            self.component_config[BATCH_SIZES],  # 批次大小
            self.component_config[EPOCHS],  # 训练轮数
            self.component_config[BATCH_STRATEGY],  # 批次策略
            self.component_config[EVAL_NUM_EXAMPLES],  # 评估样本数
            self.component_config[RANDOM_SEED],  # 随机种子
            drop_small_last_batch=self.component_config[DROP_SMALL_LAST_BATCH],  # 丢弃小批次
        )
        
        # 创建回调函数
        callbacks = train_utils.create_common_callbacks(
            self.component_config[EPOCHS],  # 训练轮数
            self.component_config[TENSORBOARD_LOG_DIR],  # TensorBoard日志目录
            self.component_config[TENSORBOARD_LOG_LEVEL],  # TensorBoard日志级别
            self.tmp_checkpoint_dir,  # 临时检查点目录
        )

        # 训练模型
        self.model.fit(
            data_generator,  # 数据生成器
            epochs=self.component_config[EPOCHS],  # 训练轮数
            validation_data=validation_data_generator,  # 验证数据
            validation_freq=self.component_config[EVAL_NUM_EPOCHS],  # 验证频率
            callbacks=callbacks,  # 回调函数
            verbose=False,  # 不显示详细信息
            shuffle=False,  # 我们在数据生成器内部使用自定义洗牌
        )

        # 持久化模型
        self.persist()

        return self._resource

    # process helpers
    def _predict(
        self, message: Message
    ) -> Optional[Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]]:
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message], training=False)
        if model_data.is_empty():
            return None
        return self.model.run_inference(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""
        label: Dict[Text, Any] = {"name": None, "confidence": 0.0}
        label_ranking: List[Dict[Text, Any]] = []

        if predict_out is None:
            return label, label_ranking

        message_sim = predict_out["i_scores"]
        message_sim = message_sim.flatten()  # sim is a matrix

        # if X contains all zeros do not predict some label
        if message_sim.size == 0:
            return label, label_ranking

        # rank the confidences
        ranking_length = self.component_config[RANKING_LENGTH]
        renormalize = (
            self.component_config[RENORMALIZE_CONFIDENCES]
            and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        )
        ranked_label_indices, message_sim = train_utils.rank_and_mask(
            message_sim, ranking_length=ranking_length, renormalize=renormalize
        )

        # construct the label and ranking
        casted_message_sim: List[float] = message_sim.tolist()  # np.float to float
        top_label_idx = ranked_label_indices[0]
        label = {
            "name": self.index_label_id_mapping[top_label_idx],
            "confidence": casted_message_sim[top_label_idx],
        }

        ranking = [(idx, casted_message_sim[idx]) for idx in ranked_label_indices]
        label_ranking = [
            {"name": self.index_label_id_mapping[label_idx], "confidence": score}
            for label_idx, score in ranking
        ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        predicted_tags, confidence_values = train_utils.entity_label_to_tags(
            predict_out, self._entity_tag_specs, self.component_config[BILOU_FLAG]
        )

        entities = self.convert_predictions_into_entities(
            message.get(TEXT),
            message.get(TOKENS_NAMES[TEXT], []),
            predicted_tags,
            self.split_entities_config,
            confidence_values,
        )

        entities = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + entities

        return entities

    def process(self, messages: List[Message]) -> List[Message]:
        """用意图、实体和诊断数据增强消息。
        
        Args:
            messages: 消息列表
            
        Returns:
            增强后的消息列表
        """
        for message in messages:
            # 预测消息
            out = self._predict(message)

            # 如果启用意图分类
            if self.component_config[INTENT_CLASSIFICATION]:
                label, label_ranking = self._predict_label(out)

                # 设置意图和意图排名
                message.set(INTENT, label, add_to_output=True)
                message.set("intent_ranking", label_ranking, add_to_output=True)

            # 如果启用实体识别
            if self.component_config[ENTITY_RECOGNITION]:
                entities = self._predict_entities(out, message)

                # 设置实体
                message.set(ENTITIES, entities, add_to_output=True)

            # 如果应该添加诊断数据
            if out and self._execution_context.should_add_diagnostic_data:
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
            tf_model_file = model_path / f"{file_name}.tf_model"

            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            if self.component_config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # Save an empty file to flag that this model has been
                # produced using checkpointing
                checkpoint_marker = model_path / f"{file_name}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            self.model.save(str(tf_model_file))

            # save data example
            serialize_nested_feature_arrays(
                self._data_example,
                model_path / f"{file_name}.data_example.st",
                model_path / f"{file_name}.data_example_metadata.json",
            )
            # save label data
            serialize_nested_feature_arrays(
                dict(self._label_data.data) if self._label_data is not None else {},
                model_path / f"{file_name}.label_data.st",
                model_path / f"{file_name}.label_data_metadata.json",
            )

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.sparse_feature_sizes.json",
                self._sparse_feature_sizes,
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.index_label_id_mapping.json",
                self.index_label_id_mapping,
            )

            entity_tag_specs = (
                [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
                if self._entity_tag_specs
                else []
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.entity_tag_specs.json", entity_tag_specs
            )

    @classmethod
    def load(
        cls: Type[DIETClassifierT],
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> DIETClassifierT:
        """Loads a policy from the storage (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                return cls._load(
                    model_path, config, model_storage, resource, execution_context
                )
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)

    @classmethod
    def _load(
        cls: Type[DIETClassifierT],
        model_path: Path,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DIETClassifierT:
        """Loads the trained model from the provided directory."""
        (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            data_example,
            sparse_feature_sizes,
        ) = cls._load_from_files(model_path)

        config = train_utils.update_confidence_type(config)
        config = train_utils.update_similarity_type(config)

        model = cls._load_model(
            entity_tag_specs,
            label_data,
            config,
            data_example,
            model_path,
            finetune_mode=execution_context.is_finetuning,
        )

        return cls(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
            index_label_id_mapping=index_label_id_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @classmethod
    def _load_from_files(
        cls, model_path: Path
    ) -> Tuple[
        Dict[int, Text],
        List[EntityTagSpec],
        RasaModelData,
        Dict[Text, Dict[Text, List[FeatureArray]]],
        Dict[Text, Dict[Text, List[int]]],
    ]:
        file_name = cls.__name__

        # load data example
        data_example = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.data_example.st"),
            str(model_path / f"{file_name}.data_example_metadata.json"),
        )
        # load label data
        loaded_label_data = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.label_data.st"),
            str(model_path / f"{file_name}.label_data_metadata.json"),
        )
        label_data = RasaModelData(data=loaded_label_data)

        sparse_feature_sizes = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.sparse_feature_sizes.json"
        )
        index_label_id_mapping = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.index_label_id_mapping.json"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.entity_tag_specs.json"
        )
        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        return (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            data_example,
            sparse_feature_sizes,
        )

    @classmethod
    def _load_model(
        cls,
        entity_tag_specs: List[EntityTagSpec],
        label_data: RasaModelData,
        config: Dict[Text, Any],
        data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
        model_path: Path,
        finetune_mode: bool = False,
    ) -> "RasaModel":
        file_name = cls.__name__
        tf_model_file = model_path / f"{file_name}.tf_model"

        label_key = LABEL_KEY if config[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if config[INTENT_CLASSIFICATION] else None

        model_data_example = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )

        model = cls._load_model_class(
            tf_model_file,
            model_data_example,
            label_data,
            entity_tag_specs,
            config,
            finetune_mode=finetune_mode,
        )

        return model

    @classmethod
    def _load_model_class(
        cls,
        tf_model_file: Text,
        model_data_example: RasaModelData,
        label_data: RasaModelData,
        entity_tag_specs: List[EntityTagSpec],
        config: Dict[Text, Any],
        finetune_mode: bool,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        return cls.model_class().load(
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
        return self.model_class()(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )


class DIET(TransformerRasaModel):
    """DIET 模型类，继承自 TransformerRasaModel。
    
    DIET (Dual Intent and Entity Transformer) 是一个多任务学习模型，
    用于同时进行意图分类和实体提取。
    """
    
    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
        config: Dict[Text, Any],
    ) -> None:
        """初始化 DIET 模型。
        
        Args:
            data_signature: 数据签名
            label_data: 标签数据
            entity_tag_specs: 实体标签规范
            config: 配置字典
        """
        # 在调用父类之前创建实体标签规范，否则构建模型会失败
        super().__init__("DIET", config, data_signature, label_data)
        self._entity_tag_specs = self._ordered_tag_specs(entity_tag_specs)  # 有序的标签规范

        # 预测数据签名，只包含文本特征
        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if TEXT in feature_name
        }

        # TensorFlow 训练相关
        self._create_metrics()  # 创建指标
        self._update_metrics_to_log()  # 更新要记录的指标

        # 用于高效预测
        self.all_labels_embed: Optional[tf.Tensor] = None  # 所有标签的嵌入

        # 准备层
        self._prepare_layers()

    @staticmethod
    def _ordered_tag_specs(
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> List[EntityTagSpec]:
        """Ensure that order of entity tag specs matches CRF layer order."""
        if entity_tag_specs is None:
            return []

        crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        ordered_tag_spec = []

        for tag_name in crf_order:
            for tag_spec in entity_tag_specs:
                if tag_name == tag_spec.tag_name:
                    ordered_tag_spec.append(tag_spec)

        return ordered_tag_spec

    def _check_data(self) -> None:
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL not in self.data_signature:
                raise InvalidConfigException(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

            if self.config[SHARE_HIDDEN_LAYERS]:
                different_sentence_signatures = False
                different_sequence_signatures = False
                if (
                    SENTENCE in self.data_signature[TEXT]
                    and SENTENCE in self.data_signature[LABEL]
                ):
                    different_sentence_signatures = (
                        self.data_signature[TEXT][SENTENCE]
                        != self.data_signature[LABEL][SENTENCE]
                    )
                if (
                    SEQUENCE in self.data_signature[TEXT]
                    and SEQUENCE in self.data_signature[LABEL]
                ):
                    different_sequence_signatures = (
                        self.data_signature[TEXT][SEQUENCE]
                        != self.data_signature[LABEL][SEQUENCE]
                    )

                if different_sentence_signatures or different_sequence_signatures:
                    raise ValueError(
                        "If hidden layer weights are shared, data signatures "
                        "for text_features and label_features must coincide."
                    )

        if self.config[ENTITY_RECOGNITION] and (
            ENTITIES not in self.data_signature
            or ENTITY_ATTRIBUTE_TYPE not in self.data_signature[ENTITIES]
        ):
            logger.debug(
                f"You specified '{self.__class__.__name__}' to train entities, but "
                f"no entities are present in the training data. Skipping training of "
                f"entities."
            )
            self.config[ENTITY_RECOGNITION] = False

    def _create_metrics(self) -> None:
        # self.metrics will have the same order as they are created
        # so create loss metrics first to output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_group_loss = tf.keras.metrics.Mean(name="g_loss")
        self.entity_role_loss = tf.keras.metrics.Mean(name="r_loss")
        # create accuracy metrics second to output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.entity_group_f1 = tf.keras.metrics.Mean(name="g_f1")
        self.entity_role_f1 = tf.keras.metrics.Mean(name="r_f1")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log.append("i_acc")
            if debug_log_level:
                self.metrics_to_log.append("i_loss")
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log.append(f"{name[0]}_f1")
                    if debug_log_level:
                        self.metrics_to_log.append(f"{name[0]}_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {
            "t": "total",
            "i": "intent",
            "e": "entity",
            "m": "mask",
            "r": "role",
            "g": "group",
        }
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        # For user text, prepare layers that combine different feature types, embed
        # everything using a transformer and optionally also do masked language
        # modeling.
        self.text_name = TEXT
        self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ] = rasa_layers.RasaSequenceLayer(
            self.text_name, self.data_signature[self.text_name], self.config
        )
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        # Intent labels are treated similarly to user text but without the transformer,
        # without masked language modelling, and with no dropout applied to the
        # individual features, only to the overall label embedding after all label
        # features have been combined.
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

            # disable input dropout applied to sparse and dense label features
            label_config = self.config.copy()
            label_config.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )

            self._tf_layers[
                f"feature_combining_layer.{self.label_name}"
            ] = rasa_layers.RasaFeatureCombiningLayer(
                self.label_name, self.label_signature[self.label_name], label_config
            )

            self._prepare_ffnn_layer(
                self.label_name,
                self.config[HIDDEN_LAYERS_SIZES][self.label_name],
                self.config[DROP_RATE],
            )

            self._prepare_label_classification_layers(predictor_attribute=TEXT)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_mask_lm_loss(self, name: Text) -> None:
        # for embedding predicted tokens at masked positions
        self._prepare_embed_layers(f"{name}_lm_mask")

        # for embedding the true tokens that got masked
        self._prepare_embed_layers(f"{name}_golden_token")

        # mask loss is additional loss
        # set scaling to False, so that it doesn't overpower other losses
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _create_bow(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sequence_feature_lengths: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:

        x, _ = self._tf_layers[f"feature_combining_layer.{name}"](
            (sequence_features, sentence_features, sequence_feature_lengths),
            training=self._training,
        )

        # convert to bag-of-words by summing along the sequence dimension
        x = tf.reduce_sum(x, axis=1)

        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        x = self._create_bow(
            self.tf_label_data[LABEL][SEQUENCE],
            self.tf_label_data[LABEL][SENTENCE],
            sequence_feature_lengths,
            self.label_name,
        )
        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _mask_loss(
        self,
        outputs: tf.Tensor,
        inputs: tf.Tensor,
        seq_ids: tf.Tensor,
        mlm_mask_boolean: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        mlm_mask_boolean = tf.cond(
            tf.reduce_any(mlm_mask_boolean),
            lambda: mlm_mask_boolean,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(mlm_mask_boolean)),
        )

        mlm_mask_boolean = tf.squeeze(mlm_mask_boolean, -1)

        # Pick elements that were masked, throwing away the batch & sequence dimension
        # and effectively switching from shape (batch_size, sequence_length, units) to
        # (num_masked_elements, units).
        outputs = tf.boolean_mask(outputs, mlm_mask_boolean)
        inputs = tf.boolean_mask(inputs, mlm_mask_boolean)
        ids = tf.boolean_mask(seq_ids, mlm_mask_boolean)

        tokens_predicted_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        tokens_true_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        # To limit the otherwise computationally expensive loss calculation, we
        # constrain the label space in MLM (i.e. token space) to only those tokens that
        # were masked in this batch. Hence the reduced list of token embeddings
        # (tokens_true_embed) and the reduced list of labels (ids) are passed as
        # all_labels_embed and all_labels, respectively. In the future, we could be less
        # restrictive and construct a slightly bigger label space which could include
        # tokens not masked in the current batch too.
        return self._tf_layers[f"loss.{name}_mask"](
            inputs_embed=tokens_predicted_embed,
            labels_embed=tokens_true_embed,
            labels=ids,
            all_labels_embed=tokens_true_embed,
            all_labels=ids,
        )

    def _calculate_label_loss(
        self, text_features: tf.Tensor, label_features: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        return self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )

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

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )

        (
            text_transformed,
            text_in,
            mask_combined_sequence_sentence,
            text_seq_ids,
            mlm_mask_boolean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        losses = []

        # Lengths of sequences in case of sentence-level features are always 1, but they
        # can effectively be 0 if sentence-level features aren't present.
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        combined_sequence_sentence_feature_lengths = (
            sequence_feature_lengths + sentence_feature_lengths
        )

        if self.config[MASKED_LM] and self._training:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, mlm_mask_boolean_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                combined_sequence_sentence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )

        return tf.math.add_n(losses)

    def _batch_loss_intent(
        self,
        combined_sequence_sentence_feature_lengths_text: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        # get sentence features vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )
        loss, acc = self._calculate_label_loss(sentence_vector, label, label_ids)

        self._update_label_metrics(loss, acc)

        return loss

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)

    def _batch_loss_entities(
        self,
        mask_combined_sequence_sentence: tf.Tensor,
        sequence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> List[tf.Tensor]:
        losses = []

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            tag_ids = tf_batch_data[ENTITIES][tag_spec.tag_name][0]
            # add a zero (no entity) for the sentence features to match the shape of
            # inputs
            tag_ids = tf.pad(tag_ids, [[0, 0], [0, 1], [0, 0]])

            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                tag_spec.tag_name,
                entity_tags,
            )

            if tag_spec.tag_name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(tag_ids[:, :, 0], tf.int32), depth=tag_spec.num_tags
                )

            self._update_entity_metrics(loss, f1, tag_spec.tag_name)

            losses.append(loss)

        return losses

    def _update_entity_metrics(
        self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text
    ) -> None:
        if tag_name == ENTITY_ATTRIBUTE_TYPE:
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            self.entity_group_loss.update_state(loss)
            self.entity_group_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_ROLE:
            self.entity_role_loss.update_state(loss)
            self.entity_role_f1.update_state(f1)

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, tf.Tensor]:
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
        sentence_feature_lengths = self._get_sentence_feature_lengths(
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

        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(
                    sequence_feature_lengths + sentence_feature_lengths,
                    text_transformed,
                )
            )

        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_feature_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
        self, sequence_feature_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:
        predictions: Dict[Text, tf.Tensor] = {}

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            # skip crf layer if it was not trained
            if tag_spec.num_tags == 0:
                continue

            name = tag_spec.tag_name
            _input = text_transformed

            if entity_tags is not None:
                _tags = self._tf_layers[f"embed.{name}.tags"](entity_tags)
                _input = tf.concat([_input, _tags], axis=-1)

            _logits = self._tf_layers[f"embed.{name}.logits"](_input)
            pred_ids, confidences = self._tf_layers[f"crf.{name}"](
                _logits, sequence_feature_lengths
            )

            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

            if name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
        self,
        combined_sequence_sentence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:

        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        return {"i_scores": scores}
