# 未来注解支持
from __future__ import annotations

# 标准库导入
import logging  # 日志记录
from pathlib import Path  # 路径处理
from collections import defaultdict  # 默认字典
import contextlib  # 上下文管理
from typing import Any, List, Optional, Text, Dict, Tuple, Union, Type  # 类型提示

# 第三方库导入
import numpy as np  # 数值计算
import tensorflow as tf  # 深度学习框架

# Rasa 引擎相关导入
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方
from rasa.engine.graph import ExecutionContext  # 执行上下文
from rasa.engine.storage.resource import Resource  # 资源
from rasa.engine.storage.storage import ModelStorage  # 模型存储

# Rasa 异常导入
from rasa.exceptions import ModelNotFound  # 模型未找到异常

# NLU 相关导入
from rasa.nlu.constants import TOKENS_NAMES  # 令牌名称常量
from rasa.nlu.extractors.extractor import EntityTagSpec, EntityExtractorMixin  # 实体提取器

# 核心动作导入
import rasa.core.actions.action  # 核心动作模块

# 特征化器导入
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization  # 消息容器
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer  # 跟踪器特征化器
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer  # 最大历史特征化器

# 共享异常导入
from rasa.shared.exceptions import RasaException  # Rasa基础异常

# NLU 常量导入
from rasa.shared.nlu.constants import (
    ACTION_TEXT,  # 动作文本
    ACTION_NAME,  # 动作名称
    INTENT,  # 意图
    TEXT,  # 文本
    ENTITIES,  # 实体
    FEATURE_TYPE_SENTENCE,  # 句子特征类型
    ENTITY_ATTRIBUTE_TYPE,  # 实体属性类型
    ENTITY_TAGS,  # 实体标签
    EXTRACTOR,  # 提取器
    SPLIT_ENTITIES_BY_COMMA,  # 按逗号分割实体
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,  # 按逗号分割实体默认值
)

# 策略相关导入
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData  # 策略基类
from rasa.core.constants import (
    DIALOGUE,  # 对话
    POLICY_MAX_HISTORY,  # 策略最大历史
    DEFAULT_MAX_HISTORY,  # 默认最大历史
    DEFAULT_POLICY_PRIORITY,  # 默认策略优先级
    POLICY_PRIORITY,  # 策略优先级
)

# 共享常量导入
from rasa.shared.constants import DIAGNOSTIC_DATA  # 诊断数据

# 核心常量导入
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS, ACTION_LISTEN_NAME  # 活跃循环、槽位、监听动作

# 跟踪器相关导入
from rasa.shared.core.trackers import DialogueStateTracker  # 对话状态跟踪器
from rasa.shared.core.generator import TrackerWithCachedStates  # 带缓存状态的跟踪器

# 事件相关导入
from rasa.shared.core.events import EntitiesAdded, Event  # 实体添加事件

# 域相关导入
from rasa.shared.core.domain import Domain  # 对话域

# 训练数据相关导入
from rasa.shared.nlu.training_data.message import Message  # 消息类
from rasa.shared.nlu.training_data.features import (
    Features,  # 特征类
    save_features,  # 保存特征
    load_features,  # 加载特征
)

# 工具模块导入
import rasa.shared.utils.io  # 共享IO工具
import rasa.utils.io  # IO工具
from rasa.utils import train_utils  # 训练工具

# TensorFlow 特征数组导入
from rasa.utils.tensorflow.feature_array import (
    FeatureArray,  # 特征数组
    serialize_nested_feature_arrays,  # 序列化嵌套特征数组
    deserialize_nested_feature_arrays,  # 反序列化嵌套特征数组
)

# TensorFlow 模型导入
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel  # Rasa模型和Transformer模型
from rasa.utils.tensorflow import rasa_layers  # Rasa层

# 模型数据导入
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature, Data  # 模型数据
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format  # 数据格式转换

# TensorFlow 常量导入
from rasa.utils.tensorflow.constants import (
    LABEL,  # 标签
    IDS,  # ID
    TRANSFORMER_SIZE,  # Transformer大小
    NUM_TRANSFORMER_LAYERS,  # Transformer层数
    NUM_HEADS,  # 注意力头数
    BATCH_SIZES,  # 批次大小
    BATCH_STRATEGY,  # 批次策略
    EPOCHS,  # 训练轮数
    RANDOM_SEED,  # 随机种子
    LEARNING_RATE,  # 学习率
    RANKING_LENGTH,  # 排序长度
    RENORMALIZE_CONFIDENCES,  # 重新归一化置信度
    LOSS_TYPE,  # 损失类型
    SIMILARITY_TYPE,  # 相似度类型
    NUM_NEG,  # 负样本数量
    EVAL_NUM_EXAMPLES,  # 评估样本数量
    EVAL_NUM_EPOCHS,  # 评估轮数
    NEGATIVE_MARGIN_SCALE,  # 负边距缩放
    REGULARIZATION_CONSTANT,  # 正则化常数
    SCALE_LOSS,  # 损失缩放
    USE_MAX_NEG_SIM,  # 使用最大负相似度
    MAX_NEG_SIM,  # 最大负相似度
    MAX_POS_SIM,  # 最大正相似度
    EMBEDDING_DIMENSION,  # 嵌入维度
    DROP_RATE_DIALOGUE,  # 对话丢弃率
    DROP_RATE_LABEL,  # 标签丢弃率
    DROP_RATE,  # 丢弃率
    DROP_RATE_ATTENTION,  # 注意力丢弃率
    CONNECTION_DENSITY,  # 连接密度
    KEY_RELATIVE_ATTENTION,  # 键相对注意力
    VALUE_RELATIVE_ATTENTION,  # 值相对注意力
    MAX_RELATIVE_POSITION,  # 最大相对位置
    CROSS_ENTROPY,  # 交叉熵
    AUTO,  # 自动
    BALANCED,  # 平衡
    TENSORBOARD_LOG_DIR,  # TensorBoard日志目录
    TENSORBOARD_LOG_LEVEL,  # TensorBoard日志级别
    CHECKPOINT_MODEL,  # 检查点模型
    ENCODING_DIMENSION,  # 编码维度
    UNIDIRECTIONAL_ENCODER,  # 单向编码器
    SEQUENCE,  # 序列
    SENTENCE,  # 句子
    SEQUENCE_LENGTH,  # 序列长度
    DENSE_DIMENSION,  # 密集维度
    CONCAT_DIMENSION,  # 连接维度
    SPARSE_INPUT_DROPOUT,  # 稀疏输入丢弃
    DENSE_INPUT_DROPOUT,  # 密集输入丢弃
    MASKED_LM,  # 掩码语言模型
    MASK,  # 掩码
    HIDDEN_LAYERS_SIZES,  # 隐藏层大小
    FEATURIZERS,  # 特征化器
    ENTITY_RECOGNITION,  # 实体识别
    CONSTRAIN_SIMILARITIES,  # 约束相似度
    MODEL_CONFIDENCE,  # 模型置信度
    SOFTMAX,  # Softmax
    BILOU_FLAG,  # BILOU标志
    EPOCH_OVERRIDE,  # 轮数覆盖
    USE_GPU,  # 使用GPU
)

# 日志记录器
logger = logging.getLogger(__name__)

# 常量定义
E2E_CONFIDENCE_THRESHOLD = "e2e_confidence_threshold"  # 端到端置信度阈值
LABEL_KEY = LABEL  # 标签键
LABEL_SUB_KEY = IDS  # 标签子键
LENGTH = "length"  # 长度
INDICES = "indices"  # 索引

# 需要编码的句子特征
SENTENCE_FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]  # 意图、文本、动作名称、动作文本

# 需要编码的序列特征
SEQUENCE_FEATURES_TO_ENCODE = [TEXT, ACTION_TEXT, f"{LABEL}_{ACTION_TEXT}"]  # 文本、动作文本、标签动作文本

# 需要编码的标签特征
LABEL_FEATURES_TO_ENCODE = [
    f"{LABEL}_{ACTION_NAME}",  # 标签动作名称
    f"{LABEL}_{ACTION_TEXT}",  # 标签动作文本
    f"{LABEL}_{INTENT}",  # 标签意图
]

# 状态级别特征
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, ACTIVE_LOOP]  # 实体、槽位、活跃循环

# 预测特征
PREDICTION_FEATURES = STATE_LEVEL_FEATURES + SENTENCE_FEATURES_TO_ENCODE + [DIALOGUE]  # 状态级别特征 + 句子特征 + 对话


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class TEDPolicy(Policy):
    """Transformer Embedding Dialogue (TED) 策略。

    模型架构在 https://arxiv.org/abs/1910.00486 中有详细描述。
    简而言之，架构包含以下步骤：
        - 将用户输入（用户意图和实体）、先前的系统动作、
          槽位和活跃表单在每个时间步连接成输入向量，送入
          预Transformer嵌入层；
        - 将其输入到Transformer；
        - 对Transformer的输出应用密集层以获得每个时间步的
          对话嵌入；
        - 应用密集层为每个时间步创建系统动作的嵌入；
        - 计算对话嵌入和嵌入的系统动作之间的相似度。
          此步骤基于StarSpace (https://arxiv.org/abs/1709.03856) 的思想。
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回默认配置（参见父类的完整文档字符串）。"""
        # 更改默认参数时请确保更新文档
        return {
            # ## 使用的神经网络架构
            # 用户消息和标签嵌入层之前的隐藏层大小。
            # 隐藏层数量等于对应列表的长度。
            HIDDEN_LAYERS_SIZES: {  # 隐藏层大小
                TEXT: [],  # 文本
                ACTION_TEXT: [],  # 动作文本
                f"{LABEL}_{ACTION_TEXT}": [],  # 标签动作文本
            },
            # 用于稀疏特征的密集维度。
            DENSE_DIMENSION: {  # 密集维度
                TEXT: 128,  # 文本
                ACTION_TEXT: 128,  # 动作文本
                f"{LABEL}_{ACTION_TEXT}": 128,  # 标签动作文本
                INTENT: 20,  # 意图
                ACTION_NAME: 20,  # 动作名称
                f"{LABEL}_{ACTION_NAME}": 20,  # 标签动作名称
                ENTITIES: 20,  # 实体
                SLOTS: 20,  # 槽位
                ACTIVE_LOOP: 20,  # 活跃循环
            },
            # 用于连接序列和句子特征的默认维度。
            CONCAT_DIMENSION: {  # 连接维度
                TEXT: 128,  # 文本
                ACTION_TEXT: 128,  # 动作文本
                f"{LABEL}_{ACTION_TEXT}": 128,  # 标签动作文本
            },
            # 对话Transformer编码器之前嵌入向量的维度大小。
            ENCODING_DIMENSION: 50,  # 编码维度
            # Transformer编码器中的单元数
            TRANSFORMER_SIZE: {  # Transformer大小
                TEXT: 128,  # 文本
                ACTION_TEXT: 128,  # 动作文本
                f"{LABEL}_{ACTION_TEXT}": 128,  # 标签动作文本
                DIALOGUE: 128,  # 对话
            },
            # Transformer编码器中的层数
            NUM_TRANSFORMER_LAYERS: {  # Transformer层数
                TEXT: 1,  # 文本
                ACTION_TEXT: 1,  # 动作文本
                f"{LABEL}_{ACTION_TEXT}": 1,  # 标签动作文本
                DIALOGUE: 1,  # 对话
            },
            # Transformer中的注意力头数
            NUM_HEADS: 4,  # 注意力头数
            # 如果为'True'，在注意力中使用键相对嵌入
            KEY_RELATIVE_ATTENTION: False,  # 键相对注意力
            # 如果为'True'，在注意力中使用值相对嵌入
            VALUE_RELATIVE_ATTENTION: False,  # 值相对注意力
            # 相对嵌入的最大位置。仅在键或值相对注意力开启时生效
            MAX_RELATIVE_POSITION: 5,  # 最大相对位置
            # 对`text`、`action_text`和`label_action_text`使用单向或双向编码器
            UNIDIRECTIONAL_ENCODER: False,  # 单向编码器
            # ## 训练参数
            # 初始和最终批次大小：
            # 批次大小将在每个轮次线性增加。
            BATCH_SIZES: [64, 256],  # 批次大小
            # 创建批次时使用的策略。
            # 可以是'sequence'或'balanced'。
            BATCH_STRATEGY: BALANCED,  # 批次策略
            # 训练的轮数
            EPOCHS: 1,  # 训练轮数
            # 设置随机种子为任何'int'以获得可重现的结果
            RANDOM_SEED: None,  # 随机种子
            # 优化器的初始学习率
            LEARNING_RATE: 0.001,  # 学习率
            # ## 嵌入参数
            # 嵌入向量的维度大小
            EMBEDDING_DIMENSION: 20,  # 嵌入维度
            # 错误标签的数量。算法将在训练期间最小化
            # 它们与用户输入的相似度。
            NUM_NEG: 20,  # 负样本数量
            # 使用的相似度度量类型，可以是'auto'、'cosine'或'inner'。
            SIMILARITY_TYPE: AUTO,  # 相似度类型
            # 损失函数的类型，可以是'cross_entropy'或'margin'。
            LOSS_TYPE: CROSS_ENTROPY,  # 损失类型
            # 应预测置信度的顶级动作数量。
            # 如果应预测所有动作的置信度，则设置为`0`。
            # 所有其他动作的置信度将设置为0。
            RANKING_LENGTH: 0,  # 排序长度
            # 确定所选顶级动作的置信度是否应重新归一化，
            # 使其总和为1。默认情况下，我们不重新归一化，
            # 并按原样返回顶级动作的置信度。
            # 注意：重新归一化仅在通过`softmax`生成置信度时才有意义。
            RENORMALIZE_CONFIDENCES: False,  # 重新归一化置信度
            # 指示算法应尝试使正确标签的嵌入向量有多相似。
            # 对于'cosine'相似度类型，应为0.0 < ... < 1.0。
            MAX_POS_SIM: 0.8,  # 最大正相似度
            # 错误标签的最大负相似度。
            # 对于'cosine'相似度类型，应为-1.0 < ... < 1.0。
            MAX_NEG_SIM: -0.2,  # 最大负相似度
            # 如果为'True'，算法仅最小化错误意图标签上的最大相似度，
            # 仅在'loss_type'设置为'margin'时使用。
            USE_MAX_NEG_SIM: True,  # 使用最大负相似度
            # 如果为'True'，按正确预测的置信度反比例缩放损失
            SCALE_LOSS: True,  # 缩放损失
            # ## 正则化参数
            # 正则化的缩放
            REGULARIZATION_CONSTANT: 0.001,  # 正则化常数
            # 最小化不同标签嵌入之间最大相似度的重要性的缩放，
            # 仅在'loss_type'设置为'margin'时使用。
            NEGATIVE_MARGIN_SCALE: 0.8,  # 负边距缩放
            # 对话特征嵌入层的丢弃率。
            DROP_RATE_DIALOGUE: 0.1,  # 对话丢弃率
            # 话语级别特征嵌入层的丢弃率。
            DROP_RATE: 0.0,  # 丢弃率
            # 标签（如动作）特征嵌入层的丢弃率。
            DROP_RATE_LABEL: 0.0,  # 标签丢弃率
            # 注意力的丢弃率。
            DROP_RATE_ATTENTION: 0.0,  # 注意力丢弃率
            # 内部层中可训练权重的比例。
            CONNECTION_DENSITY: 0.2,  # 连接密度
            # 如果为'True'，对稀疏输入张量应用丢弃
            SPARSE_INPUT_DROPOUT: True,  # 稀疏输入丢弃
            # 如果为'True'，对密集输入张量应用丢弃
            DENSE_INPUT_DROPOUT: True,  # 密集输入丢弃
            # 如果为'True'，输入消息的随机令牌将被掩码。由于TED内部
            # 没有使用相关的损失项，掩码实际上只是应用于用户话语文本的输入丢弃。
            MASKED_LM: False,  # 掩码语言模型
            # ## 评估参数
            # 计算验证准确性的频率。
            # 小值可能损害性能。
            EVAL_NUM_EPOCHS: 20,  # 评估轮数
            # 用于保留验证集的示例数量
            # 大值可能损害性能，例如模型准确性。
            # 设置为0表示不验证。
            EVAL_NUM_EXAMPLES: 0,  # 评估示例数量
            # 如果您想使用tensorboard可视化训练和验证
            # 指标，请将此选项设置为有效的输出目录。
            TENSORBOARD_LOG_DIR: None,  # TensorBoard日志目录
            # 定义何时记录tensorboard的训练指标。
            # 在每个轮次后或每个训练步骤后。
            # 有效值：'epoch'和'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",  # TensorBoard日志级别
            # 执行模型检查点
            CHECKPOINT_MODEL: False,  # 检查点模型
            # 仅在策略足够自信时才选择e2e预测
            E2E_CONFIDENCE_THRESHOLD: 0.5,  # 端到端置信度阈值
            # 指定用作序列和句子特征的特征。
            # 默认情况下，使用管道中的所有特征。
            FEATURIZERS: [],  # 特征化器
            # 如果设置为true，则在用户话语中预测实体。
            ENTITY_RECOGNITION: True,  # 实体识别
            # 如果为'True'，对所有相似度项应用sigmoid并将其
            # 添加到损失函数中，以确保相似度值近似有界。仅在交叉熵损失内部使用。
            CONSTRAIN_SIMILARITIES: False,  # 约束相似度
            # 推理期间返回的模型置信度。目前，唯一
            # 可能的值是`softmax`。
            MODEL_CONFIDENCE: SOFTMAX,  # 模型置信度
            # 'BILOU_flag'确定是否使用BILOU标记。
            # 如果设置为'True'，标记更严格，但每个实体需要更多示例。
            # 经验法则：每个实体应该有超过100个示例。
            BILOU_FLAG: True,  # BILOU标志
            # 按逗号分割实体，这对于例如食谱中的成分列表有意义，
            # 但对于地址的部分没有意义
            SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,  # 按逗号分割实体
            # 策略的最大历史，默认无限制
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,  # 策略最大历史
            # 确定策略的重要性，较高值优先
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY,  # 策略优先级
            USE_GPU: True,  # 使用GPU
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        model: Optional[RasaModel] = None,
        featurizer: Optional[TrackerFeaturizer] = None,
        fake_features: Optional[Dict[Text, List[Features]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
    ) -> None:
        """声明具有默认值的实例变量。"""
        super().__init__(  # 调用父类初始化
            config, model_storage, resource, execution_context, featurizer=featurizer
        )
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(  # 初始化实体分割配置
            config[SPLIT_ENTITIES_BY_COMMA], SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )
        self._load_params(config)  # 加载参数

        self.model = model  # 模型

        self._entity_tag_specs = entity_tag_specs  # 实体标签规范

        self.fake_features = fake_features or defaultdict(list)  # 假特征
        # TED只有在假特征中只存在文本时才端到端，假特征代表
        # 当前版本训练TED的所有可能输入特征
        self.only_e2e = TEXT in self.fake_features and INTENT not in self.fake_features  # 仅端到端

        self._label_data: Optional[RasaModelData] = None  # 标签数据
        self.data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None  # 数据示例

        self.tmp_checkpoint_dir = None  # 临时检查点目录
        if self.config[CHECKPOINT_MODEL]:  # 如果启用检查点模型
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())  # 创建临时目录

    @staticmethod
    def model_class() -> Type[TED]:
        """获取策略要使用的模型架构类。

        Returns:
            所需的类。
        """
        return TED  # 返回TED类

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        """返回元数据文件名。"""
        return "ted_policy"  # 返回ted_policy

    def _load_params(self, config: Dict[Text, Any]) -> None:
        """加载参数。"""
        new_config = rasa.utils.train_utils.check_core_deprecated_options(config)  # 检查已弃用的选项
        self.config = new_config  # 设置配置
        self._auto_update_configuration()  # 自动更新配置

    def _auto_update_configuration(self) -> None:
        """处理参数的弃用和兼容性。"""
        self.config = rasa.utils.train_utils.update_confidence_type(self.config)  # 更新置信度类型
        rasa.utils.train_utils.validate_configuration_settings(self.config)  # 验证配置设置
        self.config = rasa.utils.train_utils.update_similarity_type(self.config)  # 更新相似度类型
        self.config = rasa.utils.train_utils.update_evaluation_parameters(self.config)  # 更新评估参数

    def _create_label_data(
        self,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Tuple[RasaModelData, List[Dict[Text, List[Features]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = (
            state_featurizer.encode_all_labels(domain, precomputations)
            if state_featurizer is not None
            else []
        )

        attribute_data, _ = convert_to_data_format(
            encoded_all_labels, featurizers=self.config[FEATURIZERS]
        )

        label_data = self._assemble_label_data(attribute_data, domain)

        return label_data, encoded_all_labels

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        """Constructs data regarding labels to be fed to the model.

        The resultant model data can possibly contain one or both of the
        keys - [`label_action_name`, `label_action_text`] but will definitely
        contain the `label` key.
        `label_action_*` will contain the sequence, sentence and mask features
        for corresponding labels and `label` will contain the numerical label ids.

        Args:
            attribute_data: Feature data for all labels.
            domain: Domain of the assistant.

        Returns:
            Features of labels ready to be fed to the model.
        """
        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE_LENGTH,
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE,
        )
        label_ids = np.arange(domain.num_actions)
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
        return label_data

    @staticmethod
    def _should_extract_entities(
        entity_tags: List[List[Dict[Text, List[Features]]]]
    ) -> bool:
        for turns_tags in entity_tags:
            for turn_tags in turns_tags:
                # if turn_tags are empty or all entity tag indices are `0`
                # it means that all the inputs only contain NO_ENTITY_TAG
                if turn_tags and np.any(turn_tags[ENTITY_TAGS][0].features):
                    return True
        return False

    def _create_data_for_entities(
        self, entity_tags: Optional[List[List[Dict[Text, List[Features]]]]]
    ) -> Optional[Data]:
        if not self.config[ENTITY_RECOGNITION]:
            return None

        # check that there are real entity tags
        if entity_tags and self._should_extract_entities(entity_tags):
            entity_tags_data, _ = convert_to_data_format(entity_tags)
            return entity_tags_data

        # there are no "real" entity tags
        logger.debug(
            f"Entity recognition cannot be performed, "
            f"set '{ENTITY_RECOGNITION}' config parameter to 'False'."
        )
        self.config[ENTITY_RECOGNITION] = False

        return None

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List[Features]]]],
        label_ids: Optional[np.ndarray] = None,
        entity_tags: Optional[List[List[Dict[Text, List[Features]]]]] = None,
        encoded_all_labels: Optional[List[Dict[Text, List[Features]]]] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            tracker_state_features: a dictionary of attributes
                (INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, ACTIVE_LOOP)
                to a list of features for all dialogue turns in all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers
            entity_tags: a dictionary of entity type (ENTITY_TAGS) to a list of features
                containing entity tag ids for text user inputs otherwise empty dict
                for all dialogue turns in all training trackers
            encoded_all_labels: a list of dictionaries containing attribute features
                for label ids

        Returns:
            RasaModelData
        """
        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)

        if label_ids is not None and encoded_all_labels is not None:
            label_ids = np.array(
                [np.expand_dims(seq_label_ids, -1) for seq_label_ids in label_ids]
            )
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(label_ids, number_of_dimensions=3)],
            )

            attribute_data, self.fake_features = convert_to_data_format(
                tracker_state_features, featurizers=self.config[FEATURIZERS]
            )

            entity_tags_data = self._create_data_for_entities(entity_tags)
            if entity_tags_data is not None:
                model_data.add_data(entity_tags_data)
        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features,
                self.fake_features,
                featurizers=self.config[FEATURIZERS],
            )

        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        model_data.add_lengths(ACTION_TEXT, SEQUENCE_LENGTH, ACTION_TEXT, SEQUENCE)

        # add the dialogue lengths
        attribute_present = next(iter(list(attribute_data.keys())))
        dialogue_lengths = np.array(
            [
                np.size(np.squeeze(f, -1))
                for f in model_data.data[attribute_present][MASK][0]
            ]
        )
        model_data.data[DIALOGUE][LENGTH] = [
            FeatureArray(dialogue_lengths, number_of_dimensions=1)
        ]

        # make sure all keys are in the same order during training and prediction
        model_data.sort()

        return model_data

    @staticmethod
    def _get_trackers_for_training(
        trackers: List[TrackerWithCachedStates],
    ) -> List[TrackerWithCachedStates]:
        """Filters out the list of trackers which should not be used for training.

        Args:
            trackers: All trackers available for training.

        Returns:
            Trackers which should be used for training.
        """
        # By default, we train on all available trackers.
        return trackers

    def _prepare_for_training(
        self,
        trackers: List[TrackerWithCachedStates],
        domain: Domain,
        precomputations: MessageContainerForCoreFeaturization,
        **kwargs: Any,
    ) -> Tuple[RasaModelData, np.ndarray]:
        """Prepares data to be fed into the model.

        Args:
            trackers: List of training trackers to be featurized.
            domain: Domain of the assistant.
            precomputations: Contains precomputed features and attributes.
            **kwargs: Any other arguments.

        Returns:
            Featurized data to be fed to the model and corresponding label ids.
        """
        training_trackers = self._get_trackers_for_training(trackers)
        # dealing with training data
        tracker_state_features, label_ids, entity_tags = self._featurize_for_training(
            training_trackers,
            domain,
            precomputations=precomputations,
            bilou_tagging=self.config[BILOU_FLAG],
            **kwargs,
        )

        if not tracker_state_features:
            return RasaModelData(), label_ids

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, precomputations=precomputations
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, entity_tags, encoded_all_labels
        )

        if self.config[ENTITY_RECOGNITION]:
            self._entity_tag_specs = (
                self.featurizer.state_featurizer.entity_tag_specs
                if self.featurizer.state_featurizer is not None
                else []
            )

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        return model_data, label_ids

    def run_training(
        self, model_data: RasaModelData, label_ids: Optional[np.ndarray] = None
    ) -> None:
        """Feeds the featurized training data to the model.

        Args:
            model_data: Featurized training data.
            label_ids: Label ids corresponding to the data points in `model_data`.
                These may or may not be used by the function depending
                on how the policy is trained.
        """
        if not self.finetune_mode:
            # This means the model wasn't loaded from a
            # previously trained model and hence needs
            # to be instantiated.
            self.model = self.model_class()(
                model_data.get_signature(),
                self.config,
                isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
                self._label_data,
                self._entity_tag_specs,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE])
            )
        (
            data_generator,
            validation_data_generator,
        ) = rasa.utils.train_utils.create_data_generators(
            model_data,
            self.config[BATCH_SIZES],
            self.config[EPOCHS],
            self.config[BATCH_STRATEGY],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[RANDOM_SEED],
        )
        callbacks = rasa.utils.train_utils.create_common_callbacks(
            self.config[EPOCHS],
            self.config[TENSORBOARD_LOG_DIR],
            self.config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        if self.model is None:
            raise ModelNotFound("No model was detected prior to training.")

        self.model.fit(
            data_generator,
            epochs=self.config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> Resource:
        """训练策略（参见父类的完整文档字符串）。"""
        if not training_trackers:  # 如果没有训练跟踪器
            rasa.shared.utils.io.raise_warning(  # 发出警告
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return self._resource  # 返回资源

        training_trackers = SupportedData.trackers_for_supported_data(  # 获取支持数据的跟踪器
            self.supported_data(), training_trackers
        )

        model_data, label_ids = self._prepare_for_training(  # 准备训练数据
            training_trackers, domain, precomputations
        )

        if model_data.is_empty():  # 如果模型数据为空
            rasa.shared.utils.io.raise_warning(  # 发出警告
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return self._resource  # 返回资源

        with (  # 使用GPU或CPU
            contextlib.nullcontext() if self.config["use_gpu"] else tf.device("/cpu:0")
        ):
            self.run_training(model_data, label_ids)  # 运行训练

        self.persist()  # 持久化模型

        return self._resource  # 返回资源

    def _featurize_tracker(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> List[List[Dict[Text, List[Features]]]]:
        # construct two examples in the batch to be fed to the model -
        # one by featurizing last user text
        # and second - an optional one (see conditions below),
        # the first example in the constructed batch either does not contain user input
        # or uses intent or text based on whether TED is e2e only.
        tracker_state_features = self._featurize_for_prediction(
            tracker,
            domain,
            precomputations=precomputations,
            use_text_for_last_user_input=self.only_e2e,
            rule_only_data=rule_only_data,
        )
        # the second - text, but only after user utterance and if not only e2e
        if (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and TEXT in self.fake_features
            and not self.only_e2e
        ):
            tracker_state_features += self._featurize_for_prediction(
                tracker,
                domain,
                precomputations=precomputations,
                use_text_for_last_user_input=True,
                rule_only_data=rule_only_data,
            )
        return tracker_state_features

    def _pick_confidence(
        self, confidences: np.ndarray, similarities: np.ndarray, domain: Domain
    ) -> Tuple[np.ndarray, bool]:
        # the confidences and similarities have shape (batch-size x number of actions)
        # batch-size can only be 1 or 2;
        # in the case batch-size==2, the first example contain user intent as features,
        # the second - user text as features
        if confidences.shape[0] > 2:
            raise ValueError(
                "We cannot pick prediction from batches of size more than 2."
            )
        # we use heuristic to pick correct prediction
        if confidences.shape[0] == 2:
            # we use similarities to pick appropriate input,
            # since it seems to be more accurate measure,
            # policy is trained to maximize the similarity not the confidence
            non_e2e_action_name = domain.action_names_or_texts[
                np.argmax(confidences[0])
            ]
            logger.debug(f"User intent lead to '{non_e2e_action_name}'.")
            e2e_action_name = domain.action_names_or_texts[np.argmax(confidences[1])]
            logger.debug(f"User text lead to '{e2e_action_name}'.")
            if (
                np.max(confidences[1]) > self.config[E2E_CONFIDENCE_THRESHOLD]
                # TODO maybe compare confidences is better
                and np.max(similarities[1]) > np.max(similarities[0])
            ):
                logger.debug(f"TED predicted '{e2e_action_name}' based on user text.")
                return confidences[1], True

            logger.debug(f"TED predicted '{non_e2e_action_name}' based on user intent.")
            return confidences[0], False

        # by default the first example in a batch is the one to use for prediction
        predicted_action_name = domain.action_names_or_texts[np.argmax(confidences[0])]
        basis_for_prediction = "text" if self.only_e2e else "intent"
        logger.debug(
            f"TED predicted '{predicted_action_name}' "
            f"based on user {basis_for_prediction}."
        )
        return confidences[0], self.only_e2e

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一个动作（参见父类的完整文档字符串）。"""
        if self.model is None:  # 如果模型为空
            return self._prediction(self._default_predictions(domain))  # 返回默认预测

        # 从跟踪器创建模型数据
        tracker_state_features = self._featurize_tracker(  # 特征化跟踪器
            tracker, domain, precomputations, rule_only_data=rule_only_data
        )
        model_data = self._create_model_data(tracker_state_features)  # 创建模型数据
        outputs = self.model.run_inference(model_data)  # 运行推理

        if isinstance(outputs["similarities"], np.ndarray):  # 如果相似度是numpy数组
            # 取序列中的最后一个预测
            similarities = outputs["similarities"][:, -1, :]  # 获取相似度
        else:
            raise TypeError(  # 抛出类型错误
                "model output for `similarities` " "should be a numpy array"
            )
        if isinstance(outputs["scores"], np.ndarray):  # 如果分数是numpy数组
            confidences = outputs["scores"][:, -1, :]  # 获取置信度
        else:
            raise TypeError("model output for `scores` should be a numpy array")  # 抛出类型错误
        # 从批次中取正确的预测
        confidence, is_e2e_prediction = self._pick_confidence(  # 选择置信度
            confidences, similarities, domain
        )

        # 对置信度进行排序和掩码（如果需要）
        ranking_length = self.config[RANKING_LENGTH]  # 排序长度
        if 0 < ranking_length < len(confidence):  # 如果需要排序
            renormalize = (  # 重新归一化
                self.config[RENORMALIZE_CONFIDENCES]
                and self.config[MODEL_CONFIDENCE] == SOFTMAX
            )
            _, confidence = train_utils.rank_and_mask(  # 排序和掩码
                confidence, ranking_length=ranking_length, renormalize=renormalize
            )

        optional_events = self._create_optional_event_for_entities(  # 创建实体的可选事件
            outputs, is_e2e_prediction, precomputations, tracker
        )

        return self._prediction(  # 返回预测
            confidence.tolist(),  # 置信度列表
            is_end_to_end_prediction=is_e2e_prediction,  # 是否端到端预测
            optional_events=optional_events,  # 可选事件
            diagnostic_data=outputs.get(DIAGNOSTIC_DATA),  # 诊断数据
        )

    def _create_optional_event_for_entities(
        self,
        prediction_output: Dict[Text, tf.Tensor],
        is_e2e_prediction: bool,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        tracker: DialogueStateTracker,
    ) -> Optional[List[Event]]:
        if tracker.latest_action_name != ACTION_LISTEN_NAME or not is_e2e_prediction:
            # entities belong only to the last user message
            # and only if user text was used for prediction,
            # a user message always comes after action listen
            return None

        if not self.config[ENTITY_RECOGNITION]:
            # entity recognition is not turned on, no entities can be predicted
            return None

        # The batch dimension of entity prediction is not the same as batch size,
        # rather it is the number of last (if max history featurizer else all)
        # text inputs in the batch
        # therefore, in order to pick entities from the latest user message
        # we need to pick entities from the last batch dimension of entity prediction
        predicted_tags, confidence_values = rasa.utils.train_utils.entity_label_to_tags(
            prediction_output,
            self._entity_tag_specs,
            self.config[BILOU_FLAG],
            prediction_index=-1,
        )

        if ENTITY_ATTRIBUTE_TYPE not in predicted_tags:
            # no entities detected
            return None

        # entities belong to the last message of the tracker
        # convert the predicted tags to actual entities
        text = tracker.latest_message.text if tracker.latest_message is not None else ""
        if precomputations is not None:
            parsed_message = precomputations.lookup_message(user_text=text)
        else:
            parsed_message = Message(data={TEXT: text})
        tokens = parsed_message.get(TOKENS_NAMES[TEXT])
        entities = EntityExtractorMixin.convert_predictions_into_entities(
            text,
            tokens,
            predicted_tags,
            self.split_entities_config,
            confidences=confidence_values,
        )

        # add the extractor name
        for entity in entities:
            entity[EXTRACTOR] = "TEDPolicy"

        return [EntitiesAdded(entities)]

    def persist(self) -> None:
        """Persists the policy to a storage."""
        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called without a trained model present. "
                "Nothing to persist then!"
            )
            return

        with self._model_storage.write_to(self._resource) as model_path:
            model_filename = self._metadata_filename()
            tf_model_file = model_path / f"{model_filename}.tf_model"

            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            self.featurizer.persist(model_path)

            if self.config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # Save an empty file to flag that this model has been
                # produced using checkpointing
                checkpoint_marker = model_path / f"{model_filename}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            self.model.save(str(tf_model_file))

            self.persist_model_utilities(model_path)

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        model_filename = self._metadata_filename()
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.priority.json", self.priority
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.meta.json", self.config
        )
        # save data example
        serialize_nested_feature_arrays(
            self.data_example,
            str(model_path / f"{model_filename}.data_example.st"),
            str(model_path / f"{model_filename}.data_example_metadata.json"),
        )
        # save label data
        serialize_nested_feature_arrays(
            dict(self._label_data.data) if self._label_data is not None else {},
            str(model_path / f"{model_filename}.label_data.st"),
            str(model_path / f"{model_filename}.label_data_metadata.json"),
        )
        # save fake features
        metadata = save_features(
            self.fake_features, str(model_path / f"{model_filename}.fake_features.st")
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.fake_features_metadata.json", metadata
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.entity_tag_specs.json", entity_tag_specs
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        tf_model_file = model_path / f"{cls._metadata_filename()}.tf_model"

        # load data example
        loaded_data = deserialize_nested_feature_arrays(
            str(model_path / f"{cls._metadata_filename()}.data_example.st"),
            str(model_path / f"{cls._metadata_filename()}.data_example_metadata.json"),
        )
        # load label data
        loaded_label_data = deserialize_nested_feature_arrays(
            str(model_path / f"{cls._metadata_filename()}.label_data.st"),
            str(model_path / f"{cls._metadata_filename()}.label_data_metadata.json"),
        )
        label_data = RasaModelData(data=loaded_label_data)

        # load fake features
        metadata = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.fake_features_metadata.json"
        )
        fake_features = load_features(
            str(model_path / f"{cls._metadata_filename()}.fake_features.st"), metadata
        )

        priority = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.priority.json"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.entity_tag_specs.json"
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
        model_config = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.meta.json"
        )

        return {
            "tf_model_file": tf_model_file,
            "loaded_data": loaded_data,
            "fake_features": fake_features,
            "label_data": label_data,
            "priority": priority,
            "entity_tag_specs": entity_tag_specs,
            "model_config": model_config,
        }

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> TEDPolicy:
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
        cls,
        model_path: Path,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> TEDPolicy:
        featurizer = TrackerFeaturizer.load(model_path)

        if not (model_path / f"{cls._metadata_filename()}.data_example.st").is_file():
            return cls(
                config,
                model_storage,
                resource,
                execution_context,
                featurizer=featurizer,
            )

        model_utilities = cls._load_model_utilities(model_path)

        config = cls._update_loaded_params(config)
        if execution_context.is_finetuning and EPOCH_OVERRIDE in config:
            config[EPOCHS] = config.get(EPOCH_OVERRIDE)

        (
            model_data_example,
            predict_data_example,
        ) = cls._construct_model_initialization_data(model_utilities["loaded_data"])

        model = None

        with (contextlib.nullcontext() if config["use_gpu"] else tf.device("/cpu:0")):
            model = cls._load_tf_model(
                model_utilities,
                model_data_example,
                predict_data_example,
                featurizer,
                execution_context.is_finetuning,
            )

        return cls._load_policy_with_model(
            config,
            model_storage,
            resource,
            execution_context,
            featurizer=featurizer,
            model_utilities=model_utilities,
            model=model,
        )

    @classmethod
    def _load_policy_with_model(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: TrackerFeaturizer,
        model: TED,
        model_utilities: Dict[Text, Any],
    ) -> TEDPolicy:
        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            model=model,
            featurizer=featurizer,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
        )

    @classmethod
    def _load_tf_model(
        cls,
        model_utilities: Dict[Text, Any],
        model_data_example: RasaModelData,
        predict_data_example: RasaModelData,
        featurizer: TrackerFeaturizer,
        should_finetune: bool,
    ) -> TED:
        model = cls.model_class().load(
            str(model_utilities["tf_model_file"]),
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            config=model_utilities["model_config"],
            max_history_featurizer_is_used=isinstance(
                featurizer, MaxHistoryTrackerFeaturizer
            ),
            label_data=model_utilities["label_data"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            finetune_mode=should_finetune,
        )
        return model

    @classmethod
    def _construct_model_initialization_data(
        cls, loaded_data: Dict[Text, Dict[Text, List[FeatureArray]]]
    ) -> Tuple[RasaModelData, RasaModelData]:
        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        predict_data_example = RasaModelData(
            label_key=LABEL_KEY,
            label_sub_key=LABEL_SUB_KEY,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                # we need to remove label features for prediction if they are present
                if feature_name in PREDICTION_FEATURES
            },
        )
        return model_data_example, predict_data_example

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = rasa.utils.train_utils.update_confidence_type(meta)
        meta = rasa.utils.train_utils.update_similarity_type(meta)

        return meta


class TED(TransformerRasaModel):
    """来自 https://arxiv.org/abs/1910.00486 的TED模型架构。"""

    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        config: Dict[Text, Any],
        max_history_featurizer_is_used: bool,
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> None:
        """初始化TED模型。

        Args:
            data_signature: 输入数据的数据签名
            config: 模型配置
            max_history_featurizer_is_used: 如果为'True'，只使用最后一个对话轮次
            label_data: 标签数据
            entity_tag_specs: 实体标签规范
        """
        super().__init__("TED", config, data_signature, label_data)  # 调用父类初始化

        self.max_history_featurizer_is_used = max_history_featurizer_is_used  # 是否使用最大历史特征化器

        self.predict_data_signature = {  # 预测数据签名
            feature_name: features
            for feature_name, features in data_signature.items()
            if feature_name in PREDICTION_FEATURES  # 如果在预测特征中
        }

        self._entity_tag_specs = entity_tag_specs  # 实体标签规范

        # 指标
        self.action_loss = tf.keras.metrics.Mean(name="loss")  # 动作损失
        self.action_acc = tf.keras.metrics.Mean(name="acc")  # 动作准确率
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")  # 实体损失
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")  # 实体F1分数
        self.metrics_to_log += ["loss", "acc"]  # 要记录的指标
        if self.config[ENTITY_RECOGNITION]:  # 如果启用实体识别
            self.metrics_to_log += ["e_loss", "e_f1"]  # 添加实体指标

        # 高效预测所需
        self.all_labels_embed: Optional[tf.Tensor] = None  # 所有标签嵌入

        self._prepare_layers()  # 准备层

    def _check_data(self) -> None:
        if not any(key in [INTENT, TEXT] for key in self.data_signature.keys()):
            raise RasaException(
                f"No user features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

        if not any(
            key in [ACTION_NAME, ACTION_TEXT] for key in self.data_signature.keys()
        ):
            raise ValueError(
                f"No action features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    # ---CREATING LAYERS HELPERS---

    def _prepare_layers(self) -> None:
        for name in self.data_signature.keys():
            self._prepare_input_layers(
                name, self.data_signature[name], is_label_attribute=False
            )
            self._prepare_encoding_layers(name)

        for name in self.label_signature.keys():
            self._prepare_input_layers(
                name, self.label_signature[name], is_label_attribute=True
            )
            self._prepare_encoding_layers(name)

        self._tf_layers[
            f"transformer.{DIALOGUE}"
        ] = rasa_layers.prepare_transformer_layer(
            attribute_name=DIALOGUE,
            config=self.config,
            num_layers=self.config[NUM_TRANSFORMER_LAYERS][DIALOGUE],
            units=self.config[TRANSFORMER_SIZE][DIALOGUE],
            drop_rate=self.config[DROP_RATE_DIALOGUE],
            # use bidirectional transformer, because
            # we will invert dialogue sequence so that the last turn is located
            # at the first position and would always have
            # exactly the same positional encoding
            unidirectional=not self.max_history_featurizer_is_used,
        )

        self._prepare_label_classification_layers(DIALOGUE)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_input_layers(
        self,
        attribute_name: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        is_label_attribute: bool = False,
    ) -> None:
        """Prepares feature processing layers for sentence/sequence-level features.

        Distinguishes between label features and other features, not applying input
        dropout to the label ones.
        """
        # Disable input dropout in the config to be used if this is a label attribute.
        if is_label_attribute:
            config_to_use = self.config.copy()
            config_to_use.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )
        else:
            config_to_use = self.config
        # Attributes with sequence-level features also have sentence-level features,
        # all these need to be combined and further processed.
        if attribute_name in SEQUENCE_FEATURES_TO_ENCODE:
            self._tf_layers[
                f"sequence_layer.{attribute_name}"
            ] = rasa_layers.RasaSequenceLayer(
                attribute_name, attribute_signature, config_to_use
            )
        # Attributes without sequence-level features require some actual feature
        # processing only if they have sentence-level features. Attributes with no
        # sequence- and sentence-level features (dialogue, entity_tags, label) are
        # skipped here.
        elif SENTENCE in attribute_signature:
            self._tf_layers[
                f"sparse_dense_concat_layer.{attribute_name}"
            ] = rasa_layers.ConcatenateSparseDenseFeatures(
                attribute=attribute_name,
                feature_type=SENTENCE,
                feature_type_signature=attribute_signature[SENTENCE],
                config=config_to_use,
            )

    def _prepare_encoding_layers(self, name: Text) -> None:
        """Create Ffnn encoding layer used just before combining all dialogue features.

        Args:
            name: attribute name
        """
        # create encoding layers only for the features which should be encoded;
        if name not in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            return
        # check that there are SENTENCE features for the attribute name in data
        if (
            name in SENTENCE_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.data_signature[name]
        ):
            return
        #  same for label_data
        if (
            name in LABEL_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.label_signature[name]
        ):
            return

        self._prepare_ffnn_layer(
            f"{name}",
            [self.config[ENCODING_DIMENSION]],
            self.config[DROP_RATE_DIALOGUE],
            prefix="encoding_layer",
        )

    # ---GRAPH BUILDING HELPERS---

    @staticmethod
    def _compute_dialogue_indices(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> None:
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32)
        # wrap in a list, because that's the structure of tf_batch_data
        tf_batch_data[DIALOGUE][INDICES] = [
            (
                tf.map_fn(
                    tf.range,
                    dialogue_lengths,
                    fn_output_signature=tf.RaggedTensorSpec(
                        shape=[None], dtype=tf.int32
                    ),
                )
            ).values
        ]

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]
        # labels cannot have all features "fake"
        all_labels_encoded = {}
        for key in self.tf_label_data.keys():
            if key != LABEL_KEY:
                attribute_features, _, _ = self._encode_real_features_per_attribute(
                    self.tf_label_data, key
                )
                all_labels_encoded[key] = attribute_features

        x = self._collect_label_attribute_encodings(all_labels_encoded)

        # additional sequence axis is artifact of our RasaModelData creation
        # TODO check whether this should be solved in data creation
        x = tf.squeeze(x, axis=1)

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    @staticmethod
    def _collect_label_attribute_encodings(
        all_labels_encoded: Dict[Text, tf.Tensor]
    ) -> tf.Tensor:
        # Initialize with at least one attribute first
        # so that the subsequent TF ops are simplified.
        all_attributes_present = list(all_labels_encoded.keys())
        x = all_labels_encoded.pop(all_attributes_present[0])

        # Add remaining attributes
        for attribute in all_labels_encoded:
            x += all_labels_encoded.get(attribute)
        return x

    def _embed_dialogue(
        self,
        dialogue_in: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """Creates dialogue level embedding and mask.

        Args:
            dialogue_in: The encoded dialogue.
            tf_batch_data: Batch in model data format.

        Returns:
            The dialogue embedding, the mask, and (for diagnostic purposes)
            also the attention weights.
        """
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)
        mask = rasa_layers.compute_mask(dialogue_lengths)

        if self.max_history_featurizer_is_used:
            # invert dialogue sequence so that the last turn would always have
            # exactly the same positional encoding
            dialogue_in = tf.reverse_sequence(dialogue_in, dialogue_lengths, seq_axis=1)

        dialogue_transformed, attention_weights = self._tf_layers[
            f"transformer.{DIALOGUE}"
        ](dialogue_in, 1 - mask, self._training)
        dialogue_transformed = tf.nn.gelu(dialogue_transformed)

        if self.max_history_featurizer_is_used:
            # pick last vector if max history featurizer is used, since we inverted
            # dialogue sequence, the last vector is actually the first one
            dialogue_transformed = dialogue_transformed[:, :1, :]
            mask = tf.expand_dims(self._last_token(mask, dialogue_lengths), 1)
        elif not self._training:
            # during prediction we don't care about previous dialogue turns,
            # so to save computation time, use only the last one
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, dialogue_lengths), 1
            )
            mask = tf.expand_dims(self._last_token(mask, dialogue_lengths), 1)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask, dialogue_transformed, attention_weights

    def _encode_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The input is a representation of 4d tensor of
        # shape (batch-size x dialogue-len x sequence-len x units) in 3d of shape
        # (sum of dialogue history length for all tensors in the batch x
        # max sequence length x number of features).

        # However, some dialogue turns contain non existent state features,
        # e.g. `intent` and `text` features are mutually exclusive,
        # as well as `action_name` and `action_text` are mutually exclusive,
        # or some dialogue turns don't contain any `slots`.
        # In order to create 4d full tensors, we created "fake" zero features for
        # these non existent state features. And filtered them during batch generation.
        # Therefore the first dimensions for different attributes are different.
        # It could happen that some batches don't contain "real" features at all,
        # e.g. large number of stories don't contain any `slots`.
        # Therefore actual input tensors will be empty.
        # Since we need actual numbers to create dialogue turn features, we create
        # zero tensors in `_encode_fake_features_per_attribute` for these attributes.
        return tf.cond(
            tf.shape(tf_batch_data[attribute][SENTENCE][0])[0] > 0,
            lambda: self._encode_real_features_per_attribute(tf_batch_data, attribute),
            lambda: self._encode_fake_features_per_attribute(tf_batch_data, attribute),
        )

    def _encode_fake_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Returns dummy outputs for fake features of a given attribute.

        Needs to match the outputs of `_encode_real_features_per_attribute` in shape
        but these outputs will be filled with zeros.

        Args:
            tf_batch_data: Maps each attribute to its features and masks.
            attribute: The attribute whose fake features will be "processed", e.g.
                `ACTION_NAME`, `INTENT`.

        Returns:
            attribute_features: A tensor of shape `(batch_size, dialogue_length, units)`
                filled with zeros.
            text_output: Only for `TEXT` attribute (otherwise an empty tensor): A tensor
                of shape `(combined batch_size & dialogue_length, max seq length,
                units)` filled with zeros.
            text_sequence_lengths: Only for `TEXT` attribute, otherwise an empty tensor:
                Of hape `(combined batch_size & dialogue_length, 1)`, filled with zeros.
        """
        # we need to create real zero tensors with appropriate batch and dialogue dim
        # because they are passed to dialogue transformer
        attribute_mask = tf_batch_data[attribute][MASK][0]

        # determine all dimensions so that fake features of the correct shape can be
        # created
        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        if attribute in set(SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE):
            units = self.config[ENCODING_DIMENSION]
        else:
            # state-level attributes don't use an encoding layer, hence their size is
            # just the output size of the corresponding sparse+dense feature combining
            # layer
            units = self._tf_layers[
                f"sparse_dense_concat_layer.{attribute}"
            ].output_units

        attribute_features = tf.zeros(
            (batch_dim, dialogue_dim, units), dtype=tf.float32
        )

        # Only for user text, the transformer output and sequence lengths also have to
        # be created (here using fake features) to enable entity recognition training
        # and prediction.
        if attribute == TEXT:
            # we just need to get the correct last dimension size from the prepared
            # transformer
            text_units = self._tf_layers[f"sequence_layer.{attribute}"].output_units
            text_output = tf.zeros((0, 0, text_units), dtype=tf.float32)
            text_sequence_lengths = tf.zeros((0,), dtype=tf.int32)
        else:
            # simulate None with empty tensor of zeros
            text_output = tf.zeros((0,))
            text_sequence_lengths = tf.zeros((0,))

        return attribute_features, text_output, text_sequence_lengths

    @staticmethod
    def _create_last_dialogue_turns_mask(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> tf.Tensor:
        # Since max_history_featurizer_is_used is True,
        # we need to find the locations of last dialogue turns in
        # (combined batch dimension and dialogue length,) dimension,
        # so that we can use `_sequence_lengths` as a boolean  mask to pick
        # which ones are "real" textual input in these last dialogue turns.

        # In order to do that we can use given `dialogue_lengths`.
        # For example:
        # If we have `dialogue_lengths = [2, 1, 3]`, than
        # `dialogue_indices = [0, 1, 0, 0, 1, 2]` here we can spot that `0`
        # always indicates the first dialogue turn,
        # which means that previous dialogue turn is the last dialogue turn.
        # Combining this with the fact that the last element in
        # `dialogue_indices` is always the last dialogue turn, we can add
        # a `0` to the end, getting
        # `_dialogue_indices = [0, 1, 0, 0, 1, 2, 0]`.
        # Then removing the first element
        # `_last_dialogue_turn_inverse_indicator = [1, 0, 0, 1, 2, 0]`
        # we see that `0` points to the last dialogue turn.
        # We convert all positive numbers to `True` and take
        # the inverse mask to get
        # `last_dialogue_mask = [0, 1, 1, 0, 0, 1],
        # which precisely corresponds to the fact that first dialogue is of
        # length 2, the second 1 and the third 3.
        last_dialogue_turn_mask = tf.math.logical_not(
            tf.cast(
                tf.concat(
                    [
                        tf_batch_data[DIALOGUE][INDICES][0],
                        tf.zeros((1,), dtype=tf.int32),
                    ],
                    axis=0,
                )[1:],
                dtype=tf.bool,
            )
        )
        # get only the indices of real inputs
        return tf.boolean_mask(
            last_dialogue_turn_mask,
            tf.reshape(tf_batch_data[attribute][SEQUENCE_LENGTH][0], (-1,)),
        )

    def _encode_real_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encodes features for a given attribute.

        Args:
            tf_batch_data: Maps each attribute to its features and masks.
            attribute: the attribute we will encode features for
                (e.g., ACTION_NAME, INTENT)

        Returns:
            attribute_features: A tensor of shape `(batch_size, dialogue_length, units)`
                with all features for `attribute` processed and combined. If sequence-
                level features are present, the sequence dimension is eliminated using
                a transformer.
            text_output: Only for `TEXT` attribute (otherwise an empty tensor): A tensor
                of shape `(combined batch_size & dialogue_length, max seq length,
                units)` containing token-level embeddings further used for entity
                extraction from user text. Similar to `attribute_features` but returned
                for all tokens, not just for the last one.
            text_sequence_lengths: Only for `TEXT` attribute, otherwise an empty tensor:
                Shape `(combined batch_size & dialogue_length, 1)`, containing the
                sequence length for user text examples in `text_output`. The sequence
                length is effectively the number of tokens + 1 (to account also for
                sentence-level features). Needed for entity extraction from user text.
        """
        # simulate None with empty tensor of zeros
        text_output = tf.zeros((0,))
        text_sequence_lengths = tf.zeros((0,))

        if attribute in SEQUENCE_FEATURES_TO_ENCODE:
            # get lengths of real token sequences as a 3D tensor
            sequence_feature_lengths = self._get_sequence_feature_lengths(
                tf_batch_data, attribute
            )

            # sequence_feature_lengths contain `0` for "fake" features, while
            # tf_batch_data[attribute] contains only "real" features. Hence, we need to
            # get rid of the lengths that are 0. This step produces a 1D tensor.
            sequence_feature_lengths = tf.boolean_mask(
                sequence_feature_lengths, sequence_feature_lengths
            )

            attribute_features, _, _, _, _, _ = self._tf_layers[
                f"sequence_layer.{attribute}"
            ](
                (
                    tf_batch_data[attribute][SEQUENCE],
                    tf_batch_data[attribute][SENTENCE],
                    sequence_feature_lengths,
                ),
                training=self._training,
            )

            combined_sentence_sequence_feature_lengths = sequence_feature_lengths + 1

            # Only for user text, the transformer output and sequence lengths also have
            # to be returned to enable entity recognition training and prediction.
            if attribute == TEXT:
                text_output = attribute_features
                text_sequence_lengths = combined_sentence_sequence_feature_lengths

                if self.max_history_featurizer_is_used:
                    # get the location of all last dialogue inputs
                    last_dialogue_turns_mask = self._create_last_dialogue_turns_mask(
                        tf_batch_data, attribute
                    )
                    # pick outputs that correspond to the last dialogue turns
                    text_output = tf.boolean_mask(text_output, last_dialogue_turns_mask)
                    text_sequence_lengths = tf.boolean_mask(
                        text_sequence_lengths, last_dialogue_turns_mask
                    )

            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = tf.expand_dims(
                self._last_token(
                    attribute_features, combined_sentence_sequence_feature_lengths
                ),
                axis=1,
            )

        # for attributes without sequence-level features, all we need is to combine the
        # sparse and dense sentence-level features into one
        else:
            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = self._tf_layers[
                f"sparse_dense_concat_layer.{attribute}"
            ]((tf_batch_data[attribute][SENTENCE],), training=self._training)

        if attribute in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            attribute_features = self._tf_layers[f"encoding_layer.{attribute}"](
                attribute_features, self._training
            )

        # attribute features have shape
        # (combined batch dimension and dialogue length x 1 x units)
        # convert them back to their original shape of
        # batch size x dialogue length x units
        attribute_features = self._convert_to_original_shape(
            attribute_features, tf_batch_data, attribute
        )

        return attribute_features, text_output, text_sequence_lengths

    @staticmethod
    def _convert_to_original_shape(
        attribute_features: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        attribute: Text,
    ) -> tf.Tensor:
        """Transform attribute features back to original shape.

        Given shape: (combined batch and dialogue dimension x 1 x units)
        Original shape: (batch x dialogue length x units)

        Args:
            attribute_features: the "real" features to convert
            tf_batch_data: dictionary mapping every attribute to its features and masks
            attribute: the attribute we will encode features for
                (e.g., ACTION_NAME, INTENT)

        Returns:
            The converted attribute features
        """
        # in order to convert the attribute features with shape
        # (combined batch-size and dialogue length x 1 x units)
        # to a shape of (batch-size x dialogue length x units)
        # we use tf.scatter_nd. Therefore, we need the target shape and the indices
        # mapping the values of attribute features to the position in the resulting
        # tensor.

        # attribute_mask has shape batch x dialogue_len x 1
        attribute_mask = tf_batch_data[attribute][MASK][0]

        if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
            dialogue_lengths = tf.cast(
                tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32
            )
            dialogue_indices = tf_batch_data[DIALOGUE][INDICES][0]
        else:
            # for labels, dialogue length is a fake dim and equal to 1
            dialogue_lengths = tf.ones((tf.shape(attribute_mask)[0],), dtype=tf.int32)
            dialogue_indices = tf.zeros((tf.shape(attribute_mask)[0],), dtype=tf.int32)

        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        units = attribute_features.shape[-1]

        # attribute_mask has shape (batch x dialogue_len x 1), remove last dimension
        attribute_mask = tf.cast(tf.squeeze(attribute_mask, axis=-1), dtype=tf.int32)
        # sum of attribute mask contains number of dialogue turns with "real" features
        non_fake_dialogue_lengths = tf.reduce_sum(attribute_mask, axis=-1)
        # create the batch indices
        batch_indices = tf.repeat(tf.range(batch_dim), non_fake_dialogue_lengths)

        # attribute_mask has shape (batch x dialogue_len x 1), while
        # dialogue_indices has shape (combined_dialogue_len,)
        # in order to find positions of real input we need to flatten
        # attribute mask to (combined_dialogue_len,)
        dialogue_indices_mask = tf.boolean_mask(
            attribute_mask, tf.sequence_mask(dialogue_lengths, dtype=tf.int32)
        )
        # pick only those indices that contain "real" input
        dialogue_indices = tf.boolean_mask(dialogue_indices, dialogue_indices_mask)

        indices = tf.stack([batch_indices, dialogue_indices], axis=1)

        shape = tf.convert_to_tensor([batch_dim, dialogue_dim, units])
        attribute_features = tf.squeeze(attribute_features, axis=1)

        return tf.scatter_nd(indices, attribute_features, shape)

    def _process_batch_data(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Encodes batch data.

        Combines intent and text and action name and action text if both are present.

        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks

        Returns:
             Tensor: encoding of all features in the batch, combined;
        """
        # encode each attribute present in tf_batch_data
        text_output = None
        text_sequence_lengths = None
        batch_encoded = {}
        for attribute in tf_batch_data.keys():
            if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
                (
                    attribute_features,
                    _text_output,
                    _text_sequence_lengths,
                ) = self._encode_features_per_attribute(tf_batch_data, attribute)

                batch_encoded[attribute] = attribute_features
                if attribute == TEXT:
                    text_output = _text_output
                    text_sequence_lengths = _text_sequence_lengths

        # if both action text and action name are present, combine them; otherwise,
        # return the one which is present

        if (
            batch_encoded.get(ACTION_TEXT) is not None
            and batch_encoded.get(ACTION_NAME) is not None
        ):
            batch_action = batch_encoded.pop(ACTION_TEXT) + batch_encoded.pop(
                ACTION_NAME
            )
        elif batch_encoded.get(ACTION_TEXT) is not None:
            batch_action = batch_encoded.pop(ACTION_TEXT)
        else:
            batch_action = batch_encoded.pop(ACTION_NAME)
        # same for user input
        if (
            batch_encoded.get(INTENT) is not None
            and batch_encoded.get(TEXT) is not None
        ):
            batch_user = batch_encoded.pop(INTENT) + batch_encoded.pop(TEXT)
        elif batch_encoded.get(TEXT) is not None:
            batch_user = batch_encoded.pop(TEXT)
        else:
            batch_user = batch_encoded.pop(INTENT)

        batch_features = [batch_user, batch_action]
        # once we have user input and previous action,
        # add all other attributes (SLOTS, ACTIVE_LOOP, etc.) to batch_features;
        for key in batch_encoded.keys():
            batch_features.append(batch_encoded.get(key))

        batch_features = tf.concat(batch_features, axis=-1)

        return batch_features, text_output, text_sequence_lengths

    def _reshape_for_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The first dim of the output of the text sequence transformer is the same
        # as number of "real" features for `text` at the last dialogue turns
        # (let's call it `N`),
        # which corresponds to the first dim of the tag ids tensor.
        # To calculate the loss for entities we need the output of the text
        # sequence transformer (shape: N x sequence length x units),
        # the output of the dialogue transformer
        # (shape: batch size x dialogue length x units) and the tag ids for the
        # entities (shape: N x sequence length - 1 x units)
        # In order to process the tensors, they need to have the same shape.
        # Convert the output of the dialogue transformer to shape
        # (N x 1 x units).

        # Note: The CRF layer cannot handle 4D tensors. E.g. we cannot use the shape
        # batch size x dialogue length x sequence length x units

        # convert the output of the dialogue transformer
        # to shape (real entity dim x 1 x units)
        attribute_mask = tf_batch_data[TEXT][MASK][0]
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)

        if self.max_history_featurizer_is_used:
            # pick outputs that correspond to the last dialogue turns
            attribute_mask = tf.expand_dims(
                self._last_token(attribute_mask, dialogue_lengths), axis=1
            )
        dialogue_transformer_output = tf.boolean_mask(
            dialogue_transformer_output, tf.squeeze(attribute_mask, axis=-1)
        )

        # boolean mask removed axis=1, add it back
        dialogue_transformer_output = tf.expand_dims(
            dialogue_transformer_output, axis=1
        )

        # broadcast the dialogue transformer output sequence-length-times to get the
        # same shape as the text sequence transformer output
        dialogue_transformer_output = tf.tile(
            dialogue_transformer_output, (1, tf.shape(text_output)[1], 1)
        )

        # concat the output of the dialogue transformer to the output of the text
        # sequence transformer (adding context)
        # resulting shape (N x sequence length x 2 units)
        # N = number of "real" features for `text` at the last dialogue turns
        text_transformed = tf.concat(
            [text_output, dialogue_transformer_output], axis=-1
        )
        text_mask = rasa_layers.compute_mask(text_sequence_lengths)

        # add zeros to match the shape of text_transformed, because
        # max sequence length might differ, since it is calculated dynamically
        # based on a subset of sequence lengths
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(text_mask)[1]
        text_mask = tf.pad(text_mask, [[0, 0], [0, sequence_diff], [0, 0]])

        # remove additional dims and sentence features
        text_sequence_lengths = tf.reshape(text_sequence_lengths, (-1,)) - 1

        return text_transformed, text_mask, text_sequence_lengths

    # ---TRAINING---

    def _batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:
        # It could happen that some batches don't contain "real" features for `text`,
        # e.g. large number of stories are intent only.
        # Therefore actual `text_output` will be empty.
        # We cannot create a loss with empty tensors.
        # Since we need actual numbers to create a full loss, we output
        # zero in this case.
        return tf.cond(
            tf.shape(text_output)[0] > 0,
            lambda: self._real_batch_loss_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            ),
            lambda: tf.constant(0.0),
        )

    def _real_batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:

        text_transformed, text_mask, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_output,
            text_sequence_lengths,
        )

        tag_ids = tf_batch_data[ENTITY_TAGS][IDS][0]
        # add a zero (no entity) for the sentence features to match the shape of inputs
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(tag_ids)[1]
        tag_ids = tf.pad(tag_ids, [[0, 0], [0, sequence_diff], [0, 0]])

        loss, f1, _ = self._calculate_entity_loss(
            text_transformed,
            tag_ids,
            text_mask,
            text_sequence_lengths,
            ENTITY_ATTRIBUTE_TYPE,
        )

        self.entity_loss.update_state(loss)
        self.entity_f1.update_state(f1)

        return loss

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)
        labels_embed = tf.gather(all_labels_embed, indices)

        return labels_embed

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
        self._compute_dialogue_indices(tf_batch_data)

        all_label_ids, all_labels_embed = self._create_all_labels_embed()

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        labels_embed = self._get_labels_embed(label_ids, all_labels_embed)

        dialogue_in, text_output, text_sequence_lengths = self._process_batch_data(
            tf_batch_data
        )
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
            _,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        losses = []

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            labels_embed,
            label_ids,
            all_labels_embed,
            all_label_ids,
            dialogue_mask,
        )
        losses.append(loss)

        if (
            self.config[ENTITY_RECOGNITION]
            and text_output is not None
            and text_sequence_lengths is not None
        ):
            losses.append(
                self._batch_loss_entities(
                    tf_batch_data,
                    dialogue_transformer_output,
                    text_output,
                    text_sequence_lengths,
                )
            )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return tf.math.add_n(losses)

    # ---PREDICTION---
    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        _, self.all_labels_embed = self._create_all_labels_embed()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )
        self._compute_dialogue_indices(tf_batch_data)

        dialogue_in, text_output, text_sequence_lengths = self._process_batch_data(
            tf_batch_data
        )
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
            attention_weights,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        sim_all, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        predictions = {
            "scores": scores,
            "similarities": sim_all,
            DIAGNOSTIC_DATA: {"attention_weights": attention_weights},
        }

        if (
            self.config[ENTITY_RECOGNITION]
            and text_output is not None
            and text_sequence_lengths is not None
        ):
            pred_ids, confidences = self._batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            )
            name = ENTITY_ATTRIBUTE_TYPE
            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

        return predictions

    def _batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # It could happen that current prediction turn don't contain
        # "real" features for `text`,
        # Therefore actual `text_output` will be empty.
        # We cannot predict entities with empty tensors.
        # Since we need to output some tensors of the same shape, we output
        # zero tensors.
        return tf.cond(
            tf.shape(text_output)[0] > 0,
            lambda: self._real_batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            ),
            lambda: (
                # the output is of shape (batch_size, max_seq_len)
                tf.zeros(tf.shape(text_output)[:2], dtype=tf.int32),
                tf.zeros(tf.shape(text_output)[:2], dtype=tf.float32),
            ),
        )

    def _real_batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        text_transformed, _, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_output,
            text_sequence_lengths,
        )

        name = ENTITY_ATTRIBUTE_TYPE

        _logits = self._tf_layers[f"embed.{name}.logits"](text_transformed)

        return self._tf_layers[f"crf.{name}"](_logits, text_sequence_lengths)
