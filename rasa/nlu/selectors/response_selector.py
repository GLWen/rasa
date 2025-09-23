# =============================================================================
# Rasa NLU Response Selector 响应选择器模块
# 本模块实现了基于监督嵌入的响应选择器，用于从候选响应中选择最合适的响应
# 响应选择器是 Rasa NLU 管道中的重要组件，用于处理检索式对话系统中的响应选择任务
# =============================================================================

# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入标准库模块
import copy  # 深拷贝模块，用于创建对象的完全独立副本
import logging  # 日志记录模块，用于记录程序运行状态和调试信息
from rasa.nlu.featurizers.featurizer import Featurizer  # 特征化器基类，提供特征提取的通用接口

# 导入科学计算库
import numpy as np  # 数值计算库，提供多维数组和数学运算功能
import tensorflow as tf  # 深度学习框架，用于构建和训练神经网络模型

# 导入类型注解模块，提供类型提示功能
from typing import Any, Dict, Optional, Text, Tuple, Union, List, Type

# 导入 Rasa 核心模块
from rasa.engine.graph import ExecutionContext  # 执行上下文，包含节点名称和执行模式信息
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.shared.constants import DIAGNOSTIC_DATA  # 诊断数据常量，用于存储模型诊断信息
from rasa.shared.nlu.training_data import util  # 训练数据工具，提供数据处理和转换功能
import rasa.shared.utils.io  # 共享工具模块，提供文件读写和序列化功能
from rasa.shared.exceptions import InvalidConfigException  # 配置异常类，用于处理配置错误
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息
from rasa.nlu.classifiers.diet_classifier import (  # DIET 分类器相关导入
    DIET,  # DIET 模型基类，提供双意图和实体转换器的核心功能
    LABEL_KEY,  # 标签键，用于标识标签数据的主键
    LABEL_SUB_KEY,  # 标签子键，用于标识标签数据的子键
    SENTENCE,  # 句子特征，表示句子级别的特征数据
    SEQUENCE,  # 序列特征，表示序列级别的特征数据
    DIETClassifier,  # DIET 分类器，用于意图分类和实体提取
)
from rasa.nlu.extractors.extractor import EntityTagSpec  # 实体标签规范，定义实体标签的格式和属性
from rasa.utils.tensorflow import rasa_layers  # Rasa TensorFlow 层，提供自定义的神经网络层
from rasa.utils.tensorflow.constants import (  # TensorFlow 常量导入
    LABEL,  # 标签常量，用于标识标签相关的配置
    HIDDEN_LAYERS_SIZES,  # 隐藏层大小，定义神经网络隐藏层的尺寸
    SHARE_HIDDEN_LAYERS,  # 共享隐藏层，控制是否在不同输入间共享隐藏层权重
    TRANSFORMER_SIZE,  # 转换器大小，定义 Transformer 模型的隐藏单元数
    NUM_TRANSFORMER_LAYERS,  # 转换器层数，定义 Transformer 的层数
    NUM_HEADS,  # 注意力头数，定义多头注意力机制的头数
    BATCH_SIZES,  # 批次大小，定义训练时的批次大小范围
    BATCH_STRATEGY,  # 批次策略，定义批次创建的策略（序列或平衡）
    EPOCHS,  # 训练轮数，定义模型训练的轮次数
    RANDOM_SEED,  # 随机种子，用于确保结果的可重现性
    LEARNING_RATE,  # 学习率，定义优化器的学习率
    RANKING_LENGTH,  # 排名长度，定义返回的顶级预测数量
    RENORMALIZE_CONFIDENCES,  # 重新归一化置信度，控制是否对置信度进行重新归一化
    LOSS_TYPE,  # 损失类型，定义损失函数的类型（交叉熵或边距）
    SIMILARITY_TYPE,  # 相似度类型，定义相似度计算的方法
    NUM_NEG,  # 负样本数量，定义训练时使用的负样本数量
    SPARSE_INPUT_DROPOUT,  # 稀疏输入丢弃，控制是否对稀疏输入应用丢弃
    DENSE_INPUT_DROPOUT,  # 密集输入丢弃，控制是否对密集输入应用丢弃
    MASKED_LM,  # 掩码语言模型，控制是否使用掩码语言模型训练
    ENTITY_RECOGNITION,  # 实体识别，控制是否启用实体识别功能
    INTENT_CLASSIFICATION,  # 意图分类，控制是否启用意图分类功能
    EVAL_NUM_EXAMPLES,  # 评估示例数量，定义用于验证的示例数量
    EVAL_NUM_EPOCHS,  # 评估轮数，定义评估的频率
    UNIDIRECTIONAL_ENCODER,  # 单向编码器，控制是否使用单向编码器
    DROP_RATE,  # 丢弃率，定义编码器的丢弃率
    DROP_RATE_ATTENTION,  # 注意力丢弃率，定义注意力层的丢弃率
    CONNECTION_DENSITY,  # 连接密度，定义内部层中可训练权重的比例
    NEGATIVE_MARGIN_SCALE,  # 负边距缩放，定义负边距损失的重要性
    REGULARIZATION_CONSTANT,  # 正则化常数，定义正则化的规模
    SCALE_LOSS,  # 损失缩放，控制是否根据置信度缩放损失
    USE_MAX_NEG_SIM,  # 使用最大负相似度，控制是否仅最小化最大负相似度
    MAX_NEG_SIM,  # 最大负相似度，定义错误标签的最大负相似度
    MAX_POS_SIM,  # 最大正相似度，定义正确标签的最大正相似度
    EMBEDDING_DIMENSION,  # 嵌入维度，定义嵌入向量的维度大小
    BILOU_FLAG,  # BILOU 标志，控制是否使用 BILOU 标记方案
    KEY_RELATIVE_ATTENTION,  # 键相对注意力，控制是否在注意力中使用键相对嵌入
    VALUE_RELATIVE_ATTENTION,  # 值相对注意力，控制是否在注意力中使用值相对嵌入
    MAX_RELATIVE_POSITION,  # 最大相对位置，定义相对嵌入的最大位置
    RETRIEVAL_INTENT,  # 检索意图，定义要训练的检索意图名称
    USE_TEXT_AS_LABEL,  # 使用文本作为标签，控制是否使用响应文本作为标签
    CROSS_ENTROPY,  # 交叉熵，定义交叉熵损失类型
    AUTO,  # 自动，定义自动选择模式
    BALANCED,  # 平衡，定义平衡批次策略
    TENSORBOARD_LOG_DIR,  # TensorBoard 日志目录，定义 TensorBoard 日志输出目录
    TENSORBOARD_LOG_LEVEL,  # TensorBoard 日志级别，定义 TensorBoard 日志记录级别
    CONCAT_DIMENSION,  # 连接维度，定义连接序列和句子特征的维度
    FEATURIZERS,  # 特征化器，定义用于特征提取的特征化器列表
    CHECKPOINT_MODEL,  # 检查点模型，控制是否执行模型检查点
    DENSE_DIMENSION,  # 密集维度，定义密集特征的默认维度
    CONSTRAIN_SIMILARITIES,  # 约束相似度，控制是否约束相似度值
    MODEL_CONFIDENCE,  # 模型置信度，定义模型置信度的计算方式
    SOFTMAX,  # Softmax，定义 Softmax 激活函数
)
from rasa.nlu.constants import (  # NLU 常量导入
    RESPONSE_SELECTOR_PROPERTY_NAME,  # 响应选择器属性名，用于在消息中存储响应选择器相关数据
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,  # 响应选择器检索意图，存储所有检索意图列表
    RESPONSE_SELECTOR_RESPONSES_KEY,  # 响应选择器响应键，存储候选响应列表
    RESPONSE_SELECTOR_PREDICTION_KEY,  # 响应选择器预测键，存储预测结果
    RESPONSE_SELECTOR_RANKING_KEY,  # 响应选择器排名键，存储响应排名
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,  # 响应选择器话语动作键，存储话语动作名称
    RESPONSE_SELECTOR_DEFAULT_INTENT,  # 响应选择器默认意图，定义默认的检索意图
    DEFAULT_TRANSFORMER_SIZE,  # 默认转换器大小，定义转换器的默认大小
)
from rasa.shared.nlu.constants import (  # 共享 NLU 常量导入
    TEXT,  # 文本常量，用于标识文本属性
    INTENT,  # 意图常量，用于标识意图属性
    RESPONSE,  # 响应常量，用于标识响应属性
    INTENT_RESPONSE_KEY,  # 意图响应键，用于标识意图响应键属性
    INTENT_NAME_KEY,  # 意图名称键，用于标识意图名称属性
    PREDICTED_CONFIDENCE_KEY,  # 预测置信度键，用于标识预测置信度属性
)

from rasa.utils.tensorflow.model_data import RasaModelData  # Rasa 模型数据，用于管理模型的特征和标签数据
from rasa.utils.tensorflow.models import RasaModel  # Rasa 模型，提供 Rasa 模型的基础功能

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True  # 注册为意图分类器组件类型，标记为可训练组件
)
class ResponseSelector(DIETClassifier):
    """使用监督嵌入的响应选择器。

    响应选择器是 Rasa NLU 管道中的重要组件，用于从候选响应中选择最合适的响应。
    它通过将用户输入和候选响应嵌入到同一向量空间中，然后通过监督学习来训练模型，
    使其能够最大化正确响应与用户输入之间的相似性，同时最小化错误响应的相似性。

    核心功能：
    1. 将用户输入和候选响应映射到共享的嵌入空间
    2. 通过监督学习训练嵌入函数，使相关输入-响应对具有高相似性
    3. 提供响应排名，包括最佳匹配和替代选项
    4. 支持两种架构：DIET2BOW（文本到词袋）和 DIET2DIET（文本到文本）

    技术特点：
    - 基于 starspace 思想：https://arxiv.org/abs/1709.03856
    - 支持 Transformer 架构和传统神经网络
    - 提供多种相似度计算方法和损失函数
    - 支持掩码语言模型训练
    - 可配置的负采样策略

    依赖要求：
    - 需要在管道中前置特征化器组件
    - 建议使用 CountVectorsFeaturizer 进行稀疏特征提取
    - 可选择性地使用 SpacyNLP 和 SpacyTokenizer 进行文本预处理

    在此实现中，`mu` 参数的处理方式与原始 starspace 不同，
    并且添加了额外的隐藏层和丢弃机制以提高模型性能。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """获取此组件运行前必须包含在管道中的组件类型。
        
        该方法定义了组件的依赖关系，确保在响应选择之前文本已经被正确特征化。
        特征化器负责将原始文本转换为模型可以处理的数值特征。
        
        Returns:
            必需的组件类型列表，包含特征化器组件
        """
        return [Featurizer]  # 需要特征化器组件，用于将文本转换为数值特征

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置参数。
        
        该方法定义了 ResponseSelector 的所有可配置参数及其默认值。
        配置参数主要来自 DIET 分类器，并添加了响应选择器特有的参数。
        这些参数控制模型的架构、训练过程、评估策略等各个方面。
        
        注意：更改默认参数时请确保更新相关文档。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        return {
            **DIETClassifier.get_default_config(),  # 继承 DIET 分类器的基础配置
            # ## 神经网络架构配置
            # 用户消息和标签的嵌入层之前的隐藏层大小
            # 隐藏层数量等于对应列表的长度，每个数字代表该层的单元数
            HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: [256, 128]},  # 隐藏层大小，定义文本和标签的隐藏层结构
            # 是否在输入词和响应之间共享隐藏层权重
            # True 表示共享权重，False 表示独立权重
            SHARE_HIDDEN_LAYERS: False,  # 共享隐藏层，控制是否在不同输入间共享隐藏层参数
            # 转换器中的单元数，None 表示使用默认值
            TRANSFORMER_SIZE: None,  # 转换器大小，定义 Transformer 模型的隐藏单元数
            # 转换器层数，0 表示不使用转换器
            NUM_TRANSFORMER_LAYERS: 0,  # 转换器层数，定义 Transformer 的层数
            # 转换器中注意力头数，用于多头注意力机制
            NUM_HEADS: 4,  # 注意力头数，定义多头注意力机制的头数
            # 如果为 'True'，在注意力中使用键相对嵌入
            KEY_RELATIVE_ATTENTION: False,  # 键相对注意力，控制是否使用键相对位置嵌入
            # 如果为 'True'，在注意力中使用值相对嵌入
            VALUE_RELATIVE_ATTENTION: False,  # 值相对注意力，控制是否使用值相对位置嵌入
            # 相对嵌入的最大位置。仅在键或值相对注意力开启时生效
            MAX_RELATIVE_POSITION: 5,  # 最大相对位置，定义相对位置嵌入的最大距离
            # 使用单向或双向编码器
            UNIDIRECTIONAL_ENCODER: False,  # 单向编码器，控制是否使用单向编码器
            # ## 训练参数配置
            # 初始和最终批次大小：批次大小将在每个轮次线性增加
            BATCH_SIZES: [64, 256],  # 批次大小，定义训练时的批次大小范围
            # 创建批次时使用的策略，可以是 'sequence' 或 'balanced'
            BATCH_STRATEGY: BALANCED,  # 批次策略，定义批次创建的策略
            # 训练的轮次数，控制模型训练的迭代次数
            EPOCHS: 300,  # 训练轮数，定义模型训练的轮次数
            # 设置随机种子为任何 'int' 以获得可重现的结果
            RANDOM_SEED: None,  # 随机种子，用于确保结果的可重现性
            # 优化器的初始学习率，控制参数更新的步长
            LEARNING_RATE: 0.001,  # 学习率，定义优化器的学习率
            # ## 嵌入参数配置
            # 嵌入向量的维度大小，影响模型的表达能力和计算复杂度
            EMBEDDING_DIMENSION: 20,  # 嵌入维度，定义嵌入向量的维度大小
            # 如果没有密集特征时使用的默认密集维度
            DENSE_DIMENSION: {TEXT: 512, LABEL: 512},  # 密集维度，定义密集特征的默认维度
            # 用于连接序列和句子特征的默认维度
            CONCAT_DIMENSION: {TEXT: 512, LABEL: 512},  # 连接维度，定义特征连接的维度
            # 错误标签的数量。算法将在训练期间最小化它们与用户输入的相似性
            NUM_NEG: 20,  # 负样本数量，定义训练时使用的负样本数量
            # 使用的相似度度量类型，'auto'、'cosine' 或 'inner'
            SIMILARITY_TYPE: AUTO,  # 相似度类型，定义相似度计算的方法
            # 损失函数的类型，'cross_entropy' 或 'margin'
            LOSS_TYPE: CROSS_ENTROPY,  # 损失类型，定义损失函数的类型
            # 应预测置信度的顶级动作数量，0 表示报告所有意图的置信度
            RANKING_LENGTH: 10,  # 排名长度，定义返回的顶级预测数量
            # 确定所选顶级动作的置信度是否应重新归一化，使它们总和为 1
            # 注意：重新归一化仅在通过 `softmax` 生成置信度时才有意义
            RENORMALIZE_CONFIDENCES: False,  # 重新归一化置信度，控制是否对置信度进行重新归一化
            # 指示算法应尝试使正确标签的嵌入向量相似
            # 对于 'cosine' 相似度类型，应为 0.0 < ... < 1.0
            MAX_POS_SIM: 0.8,  # 最大正相似度，定义正确标签的最大相似度目标
            # 错误标签的最大负相似度
            # 对于 'cosine' 相似度类型，应为 -1.0 < ... < 1.0
            MAX_NEG_SIM: -0.4,  # 最大负相似度，定义错误标签的最大负相似度目标
            # 如果为 'True'，算法仅最小化错误意图标签上的最大相似度
            # 仅在 'loss_type' 设置为 'margin' 时使用
            USE_MAX_NEG_SIM: True,  # 使用最大负相似度，控制是否仅最小化最大负相似度
            # 与正确预测的置信度成反比地缩放损失
            SCALE_LOSS: True,  # 损失缩放，控制是否根据置信度缩放损失
            # ## 正则化参数配置
            # 正则化的规模，控制 L2 正则化的强度
            REGULARIZATION_CONSTANT: 0.002,  # 正则化常数，定义正则化的规模
            # 内部层中可训练权重的比例，1.0 表示所有权重都可训练
            CONNECTION_DENSITY: 1.0,  # 连接密度，定义内部层中可训练权重的比例
            # 最小化不同标签嵌入之间最大相似度的重要性的规模
            NEGATIVE_MARGIN_SCALE: 0.8,  # 负边距缩放，定义负边距损失的重要性
            # 编码器的丢弃率，用于防止过拟合
            DROP_RATE: 0.2,  # 丢弃率，定义编码器的丢弃率
            # 注意力的丢弃率，用于防止过拟合
            DROP_RATE_ATTENTION: 0,  # 注意力丢弃率，定义注意力层的丢弃率
            # 如果为 'True'，对稀疏输入张量应用丢弃
            SPARSE_INPUT_DROPOUT: False,  # 稀疏输入丢弃，控制是否对稀疏输入应用丢弃
            # 如果为 'True'，对密集输入张量应用丢弃
            DENSE_INPUT_DROPOUT: False,  # 密集输入丢弃，控制是否对密集输入应用丢弃
            # ## 评估参数配置
            # 计算验证准确性的频率，小值可能损害性能
            EVAL_NUM_EPOCHS: 20,  # 评估轮数，定义评估的频率
            # 用于保留验证集的示例数量，大值可能损害性能
            EVAL_NUM_EXAMPLES: 0,  # 评估示例数量，定义用于验证的示例数量
            # ## 响应选择器特有配置
            # 如果为 'True'，输入消息的随机标记将被掩码，模型应预测这些标记
            MASKED_LM: False,  # 掩码语言模型，控制是否使用掩码语言模型训练
            # 此响应选择器要训练的意图名称，None 表示训练所有检索意图
            RETRIEVAL_INTENT: None,  # 检索意图，定义要训练的检索意图名称
            # 布尔标志，检查响应的实际文本是否应用作训练模型的基本真实标签
            USE_TEXT_AS_LABEL: False,  # 使用文本作为标签，控制是否使用响应文本作为标签
            # 如果要使用 tensorboard 可视化训练和验证指标，请将此选项设置为有效的输出目录
            TENSORBOARD_LOG_DIR: None,  # TensorBoard 日志目录，定义 TensorBoard 日志输出目录
            # 定义何时记录 tensorboard 的训练指标，有效值：'epoch' 和 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",  # TensorBoard 日志级别，定义 TensorBoard 日志记录级别
            # 指定用作序列和句子特征的特征，默认使用管道中的所有特征
            FEATURIZERS: [],  # 特征化器，定义用于特征提取的特征化器列表
            # 执行模型检查点，用于保存训练过程中的模型状态
            CHECKPOINT_MODEL: False,  # 检查点模型，控制是否执行模型检查点
            # 如果为 'True'，对所有相似度项应用 sigmoid 并将其添加到损失函数中
            # 仅在交叉熵损失内部使用
            CONSTRAIN_SIMILARITIES: False,  # 约束相似度，控制是否约束相似度值
            # 推理期间返回的模型置信度。目前，唯一可能的值是 `softmax`
            MODEL_CONFIDENCE: SOFTMAX,  # 模型置信度，定义模型置信度的计算方式
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
        """初始化响应选择器实例。

        该构造函数初始化 ResponseSelector 实例，设置响应选择器特有的配置参数，
        并调用父类 DIETClassifier 的初始化方法。响应选择器专门用于处理检索式对话
        系统中的响应选择任务，通过监督学习训练嵌入函数来匹配用户输入和候选响应。

        Args:
            config: 组件的配置字典，包含所有可配置参数
            model_storage: 模型存储接口，用于持久化和加载模型
            resource: 资源管理对象，用于定位和访问模型资源
            execution_context: 执行上下文，包含节点名称和执行模式信息
            index_label_id_mapping: 标签索引映射，用于编码标签和索引之间的转换
            entity_tag_specs: 实体标签规范列表，定义实体标签的格式和属性
            model: 预训练的模型实例，如果提供则直接使用
            all_retrieval_intents: 所有检索意图列表，包含数据中定义的所有检索意图
            responses: 响应字典，包含所有候选响应及其元数据
            sparse_feature_sizes: 稀疏特征大小，定义模型训练时稀疏特征的维度
        """
        component_config = config  # 保存组件配置

        # 以下属性不能为 ResponseSelector 调整，强制设置为固定值
        component_config[INTENT_CLASSIFICATION] = True  # 启用意图分类功能
        component_config[ENTITY_RECOGNITION] = False  # 禁用实体识别功能，响应选择器不处理实体
        component_config[BILOU_FLAG] = None  # 禁用 BILOU 标记，响应选择器不使用实体标记

        # 初始化响应选择器特有的实例变量
        self.responses = responses or {}  # 响应字典，存储所有候选响应
        self.all_retrieval_intents = all_retrieval_intents or []  # 所有检索意图列表
        self.retrieval_intent = None  # 当前检索意图，None 表示处理所有检索意图
        self.use_text_as_label = False  # 是否使用文本作为标签，False 表示使用意图响应键

        # 调用父类 DIETClassifier 的初始化方法
        super().__init__(
            component_config,  # 传递配置
            model_storage,  # 传递模型存储
            resource,  # 传递资源管理
            execution_context,  # 传递执行上下文
            index_label_id_mapping,  # 传递标签索引映射
            entity_tag_specs,  # 传递实体标签规范
            model,  # 传递模型实例
            sparse_feature_sizes=sparse_feature_sizes,  # 传递稀疏特征大小
        )

    @property
    def label_key(self) -> Text:
        """返回标签键。

        该方法返回用于标识标签数据的主键，用于在模型数据中定位标签相关信息。
        
        Returns:
            标签键字符串，用于标识标签数据的主键
        """
        return LABEL_KEY

    @property
    def label_sub_key(self) -> Text:
        """返回标签子键。

        该方法返回用于标识标签数据的子键，用于在模型数据中定位标签的详细信息。
        
        Returns:
            标签子键字符串，用于标识标签数据的子键
        """
        return LABEL_SUB_KEY

    @staticmethod
    def model_class(  # type: ignore[override]
        use_text_as_label: bool,
    ) -> Type[RasaModel]:
        """返回模型类。

        该方法根据是否使用文本作为标签来选择相应的模型类。
        响应选择器支持两种架构：
        1. DIET2BOW：文本到词袋架构，适用于使用意图响应键作为标签的情况
        2. DIET2DIET：文本到文本架构，适用于使用响应文本作为标签的情况

        Args:
            use_text_as_label: 是否使用文本作为标签，True 表示使用响应文本，False 表示使用意图响应键

        Returns:
            相应的模型类，用于创建模型实例
        """
        if use_text_as_label:
            return DIET2DIET  # 返回 DIET2DIET 模型类，用于文本到文本的响应选择
        else:
            return DIET2BOW  # 返回 DIET2BOW 模型类，用于文本到词袋的响应选择

    def _load_selector_params(self) -> None:
        """加载响应选择器特有的配置参数。

        该方法从组件配置中提取响应选择器特有的参数，并设置到实例变量中。
        这些参数控制响应选择器的行为，包括检索意图和标签使用方式。
        """
        self.retrieval_intent = self.component_config[RETRIEVAL_INTENT]  # 获取检索意图配置
        self.use_text_as_label = self.component_config[USE_TEXT_AS_LABEL]  # 获取是否使用文本作为标签的配置

    def _warn_about_transformer_and_hidden_layers_enabled(
        self, selector_name: Text
    ) -> None:
        """警告用户启用了转换器但未禁用隐藏层。

        响应选择器的默认配置指定了相当大的隐藏层大小，但这适用于不使用转换器的情况。
        如果存在转换器，根据我们的经验，在特征组合层和转换器之间不使用隐藏层时效果最佳。
        该方法会检查配置的一致性并发出相应的警告。

        Args:
            selector_name: 选择器名称，用于在警告信息中标识具体的组件
        """
        default_config = self.get_default_config()  # 获取默认配置
        hidden_layers_is_at_default_value = (
            self.component_config[HIDDEN_LAYERS_SIZES]
            == default_config[HIDDEN_LAYERS_SIZES]
        )  # 检查隐藏层大小是否为默认值
        config_for_disabling_hidden_layers: Dict[Text, List[Any]] = {
            k: [] for k, _ in default_config[HIDDEN_LAYERS_SIZES].items()
        }  # 创建禁用隐藏层的配置（空列表）
        
        # 如果隐藏层未被禁用，则发出警告
        if (
            self.component_config[HIDDEN_LAYERS_SIZES]
            != config_for_disabling_hidden_layers
        ):
            # 根据用户对隐藏层配置的操作生成更具体的警告信息
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
        """修正转换器大小以确保训练不会中断，并通知用户。

        如果使用转换器，默认的 `transformer_size` 会导致问题。
        我们需要设置一个合理的默认值，以确保模型正常工作。

        Args:
            selector_name: 选择器名称，用于在警告信息中标识具体的组件
        """
        if (
            self.component_config[TRANSFORMER_SIZE] is None
            or self.component_config[TRANSFORMER_SIZE] < 1
        ):  # 检查转换器大小是否无效
            rasa.shared.utils.io.raise_warning(
                f"`{TRANSFORMER_SIZE}` is set to "
                f"`{self.component_config[TRANSFORMER_SIZE]}` for "
                f"{selector_name}, but a positive size is required when using "
                f"`{NUM_TRANSFORMER_LAYERS} > 0`. {selector_name} will proceed, using "
                f"`{TRANSFORMER_SIZE}={DEFAULT_TRANSFORMER_SIZE}`. "
                f"Alternatively, specify a different value in the component's config.",
                category=UserWarning,
            )
            self.component_config[TRANSFORMER_SIZE] = DEFAULT_TRANSFORMER_SIZE  # 设置默认转换器大小

    def _check_config_params_when_transformer_enabled(self) -> None:
        """检查并修正启用转换器时的配置参数。

        这是必要的，因为各个配置参数的默认值是相互依赖的，
        当启用转换器时，某些默认值应该改变。

        该方法会检查转换器相关的配置参数，并在需要时发出警告或自动修正。
        """
        if self.component_config[NUM_TRANSFORMER_LAYERS] > 0:  # 如果启用了转换器
            selector_name = "ResponseSelector" + (
                f"({self.retrieval_intent})" if self.retrieval_intent else ""
            )  # 构建选择器名称
            self._warn_about_transformer_and_hidden_layers_enabled(selector_name)  # 检查隐藏层配置
            self._warn_and_correct_transformer_size(selector_name)  # 检查转换器大小配置

    def _check_config_parameters(self) -> None:
        """检查组件配置是否合理，并在需要时进行修正。

        该方法首先调用父类的配置检查方法，然后加载响应选择器特有的参数，
        最后检查转换器相关的配置参数。
        """
        super()._check_config_parameters()  # 调用父类的配置检查方法
        self._load_selector_params()  # 加载响应选择器特有的参数
        # 在检查完一般 DIET 相关参数后，也检查响应选择器特有的参数
        self._check_config_params_when_transformer_enabled()

    def _set_message_property(
        self, message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:
        """设置消息的响应选择器属性。

        该方法将响应选择器的预测结果和相关信息添加到消息对象中，
        包括检索意图列表和预测结果字典。

        Args:
            message: 消息对象，用于存储预测结果
            prediction_dict: 预测结果字典，包含响应选择器的预测信息
            selector_key: 选择器键，用于标识特定的响应选择器
        """
        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})  # 获取现有的选择器属性
        message_selector_properties[
            RESPONSE_SELECTOR_RETRIEVAL_INTENTS
        ] = self.all_retrieval_intents  # 添加所有检索意图列表
        message_selector_properties[selector_key] = prediction_dict  # 添加预测结果字典
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,  # 设置响应选择器属性名
            message_selector_properties,  # 设置更新后的属性值
            add_to_output=True,  # 添加到输出中
        )

    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """准备训练数据。

        该方法对训练数据进行预处理，包括数据过滤、标签编码提取和模型数据创建。
        它还会执行数据完整性检查，确保训练数据的质量和一致性。

        Args:
            training_data: 要预处理的训练数据

        Returns:
            预处理后的模型数据，用于模型训练
        """
        # 在过滤之前收集数据中存在的所有检索意图
        self.all_retrieval_intents = list(training_data.retrieval_intents)

        if self.retrieval_intent:  # 如果指定了特定的检索意图
            training_data = training_data.filter_training_examples(
                lambda ex: self.retrieval_intent == ex.get(INTENT)
            )  # 过滤出指定检索意图的训练示例
        else:
            # 检索意图参数保持默认值
            logger.info(
                "Retrieval intent parameter was left to its default value. This "
                "response selector will be trained on training examples combining "
                "all retrieval intents."
            )  # 记录信息，说明将训练所有检索意图

        label_attribute = RESPONSE if self.use_text_as_label else INTENT_RESPONSE_KEY  # 根据配置选择标签属性

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=label_attribute
        )  # 创建标签ID到索引的映射

        self.responses = training_data.responses  # 保存响应数据

        if not label_id_index_mapping:  # 如果没有标签可以训练
            return RasaModelData()  # 返回空的模型数据

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)  # 创建索引到标签ID的映射

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=label_attribute
        )  # 创建标签数据

        model_data = self._create_model_data(
            training_data.intent_examples,  # 使用意图示例
            label_id_index_mapping,  # 使用标签映射
            label_attribute=label_attribute,  # 使用标签属性
        )  # 创建模型数据

        self._check_input_dimension_consistency(model_data)  # 检查输入维度一致性

        return model_data

    def _resolve_intent_response_key(
        self, label: Dict[Text, Optional[Text]]
    ) -> Optional[Text]:
        """根据标签解析意图响应键。

        该方法根据预测的标签查找对应的意图响应键。它首先检查预测的标签是否直接匹配
        模板键，然后检查是否匹配响应文本。这确保了能够正确找到对应的响应。

        Args:
            label: 选择器预测的标签，包含标签名称等信息

        Returns:
            在已知响应中找到的标签匹配项对应的意图响应键。
            总是保证有匹配项，否则这种情况应该已经在之前被捕获并发出警告。
        """
        for key, responses in self.responses.items():  # 遍历所有响应

            # 首先检查预测的标签是否就是键本身
            search_key = util.template_key_to_intent_response_key(key)  # 将模板键转换为意图响应键
            if search_key == label.get("name"):  # 如果匹配
                return search_key  # 返回匹配的键

            # 否则遍历响应以检查文本是否有直接匹配
            for response in responses:  # 遍历该键下的所有响应
                if response.get(TEXT, "") == label.get("name"):  # 如果响应文本匹配
                    return search_key  # 返回对应的键
        return None  # 如果没有找到匹配项，返回 None

    def process(self, messages: List[Message]) -> List[Message]:
        """为消息选择最可能的响应。

        该方法是响应选择器的核心处理逻辑，它接收用户消息列表，为每个消息
        选择最合适的响应，并将预测结果添加到消息中。预测结果包括最佳响应、
        响应排名、置信度等信息。

        Args:
            messages: 包含最新用户消息的列表

        Returns:
            包含增强后的消息的列表，每个消息都包含最可能的响应、
            相关的意图响应键及其与输入的相似度
        """
        for message in messages:  # 遍历每个消息
            out = self._predict(message)  # 对消息进行预测
            top_label, label_ranking = self._predict_label(out)  # 获取最佳标签和标签排名

            # 获取顶级预测标签的确切意图响应键和关联响应
            label_intent_response_key = (
                self._resolve_intent_response_key(top_label)  # 解析意图响应键
                or top_label[INTENT_NAME_KEY]  # 如果解析失败，使用标签名称
            )
            label_responses = self.responses.get(
                util.intent_response_key_to_template_key(label_intent_response_key)  # 获取对应的响应
            )

            if label_intent_response_key and not label_responses:  # 如果找不到响应
                # 响应似乎不可用，可能是训练数据的问题，使用备用方案
                rasa.shared.utils.io.raise_warning(
                    f"Unable to fetch responses for {label_intent_response_key} "
                    f"This means that there is likely an issue with the training data."
                    f"Please make sure you have added responses for this intent."
                )
                label_responses = [{TEXT: label_intent_response_key}]  # 使用标签作为备用响应

            for label in label_ranking:  # 为排名中的每个标签添加意图响应键
                label[INTENT_RESPONSE_KEY] = (
                    self._resolve_intent_response_key(label) or label[INTENT_NAME_KEY]
                )  # 解析或使用标签名称
                # 移除 "name" 键，因为它要么与 "intent_response_key" 相同，
                # 要么是响应文本，在排名中不需要
                label.pop(INTENT_NAME_KEY)

            selector_key = (
                self.retrieval_intent  # 如果指定了检索意图
                if self.retrieval_intent
                else RESPONSE_SELECTOR_DEFAULT_INTENT  # 否则使用默认意图
            )

            logger.debug(
                f"Adding following selector key to message property: {selector_key}"
            )  # 记录调试信息

            utter_action_key = util.intent_response_key_to_template_key(
                label_intent_response_key
            )  # 将意图响应键转换为模板键
            prediction_dict = {
                RESPONSE_SELECTOR_PREDICTION_KEY: {  # 预测结果
                    RESPONSE_SELECTOR_RESPONSES_KEY: label_responses,  # 响应列表
                    PREDICTED_CONFIDENCE_KEY: top_label[PREDICTED_CONFIDENCE_KEY],  # 预测置信度
                    INTENT_RESPONSE_KEY: label_intent_response_key,  # 意图响应键
                    RESPONSE_SELECTOR_UTTER_ACTION_KEY: utter_action_key,  # 话语动作键
                },
                RESPONSE_SELECTOR_RANKING_KEY: label_ranking,  # 响应排名
            }

            self._set_message_property(message, prediction_dict, selector_key)  # 设置消息属性

            if (
                self._execution_context.should_add_diagnostic_data  # 如果需要添加诊断数据
                and out  # 并且有预测输出
                and DIAGNOSTIC_DATA in out  # 并且包含诊断数据
            ):
                message.add_diagnostic_data(
                    self._execution_context.node_name, out.get(DIAGNOSTIC_DATA)
                )  # 添加诊断数据

        return messages

    def persist(self) -> None:
        """将模型持久化到指定目录。

        该方法将响应选择器模型及其相关数据保存到模型存储中，包括响应数据、
        检索意图列表等。这些数据在模型加载时会被恢复。

        注意：如果模型为 None，则不执行任何操作。
        """
        if self.model is None:  # 如果模型不存在
            return None  # 直接返回，不执行持久化

        with self._model_storage.write_to(self._resource) as model_path:  # 打开模型存储写入上下文
            file_name = self.__class__.__name__  # 获取类名作为文件名前缀

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.responses.json", self.responses
            )  # 保存响应数据到 JSON 文件

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.retrieval_intents.json",
                self.all_retrieval_intents,
            )  # 保存检索意图列表到 JSON 文件

        super().persist()  # 调用父类的持久化方法

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
        """加载模型类。

        该方法根据配置选择适当的模型类并加载预训练的模型。
        它创建预测数据示例，然后调用相应模型类的加载方法。

        Args:
            tf_model_file: TensorFlow 模型文件路径
            model_data_example: 模型数据示例
            label_data: 标签数据
            entity_tag_specs: 实体标签规范列表
            config: 配置字典
            finetune_mode: 是否为微调模式

        Returns:
            加载的模型实例
        """
        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,  # 使用相同的标签键
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name  # 只包含文本特征
            },
        )  # 创建预测数据示例
        return cls.model_class(config[USE_TEXT_AS_LABEL]).load(  # 根据配置选择模型类并加载
            tf_model_file,  # 模型文件路径
            model_data_example,  # 模型数据示例
            predict_data_example,  # 预测数据示例
            data_signature=model_data_example.get_signature(),  # 数据签名
            label_data=label_data,  # 标签数据
            entity_tag_specs=entity_tag_specs,  # 实体标签规范
            config=copy.deepcopy(config),  # 配置的深拷贝
            finetune_mode=finetune_mode,  # 微调模式
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        """实例化模型类。

        该方法根据配置创建新的模型实例，用于训练或预测。

        Args:
            model_data: 模型数据，包含特征和标签信息

        Returns:
            新创建的模型实例
        """
        return self.model_class(self.use_text_as_label)(  # 根据配置选择模型类并实例化
            data_signature=model_data.get_signature(),  # 数据签名
            label_data=self._label_data,  # 标签数据
            entity_tag_specs=self._entity_tag_specs,  # 实体标签规范
            config=self.component_config,  # 组件配置
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
        """从指定目录加载训练好的模型。

        该方法从模型存储中加载预训练的响应选择器模型，包括模型权重、
        响应数据和检索意图列表。如果加载失败，则创建新的未训练实例。

        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            **kwargs: 其他关键字参数

        Returns:
            加载的响应选择器实例
        """
        model = super().load(  # 调用父类的加载方法
            config, model_storage, resource, execution_context, **kwargs
        )

        try:
            with model_storage.read_from(resource) as model_path:  # 打开模型存储读取上下文
                file_name = cls.__name__  # 获取类名作为文件名前缀
                responses = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.responses.json"
                )  # 读取响应数据
                all_retrieval_intents = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.retrieval_intents.json"
                )  # 读取检索意图列表
                model.responses = responses  # 设置响应数据
                model.all_retrieval_intents = all_retrieval_intents  # 设置检索意图列表
                return model  # 返回加载的模型
        except ValueError:  # 如果加载失败
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )  # 记录调试信息
            return cls(config, model_storage, resource, execution_context)  # 创建新的未训练实例


class DIET2BOW(DIET):
    """DIET2BOW 转换器实现。

    该类实现了文本到词袋（Text-to-Bag-of-Words）的响应选择架构。
    它使用 DIET 模型将用户输入文本转换为嵌入向量，然后与候选响应的词袋表示
    进行相似度计算，从而选择最合适的响应。

    主要特点：
    - 用户输入使用完整的文本特征（序列和句子特征）
    - 候选响应使用词袋表示（BOW）
    - 通过监督学习训练嵌入函数
    - 支持掩码语言模型训练
    """

    def _create_metrics(self) -> None:
        """创建训练指标。

        该方法创建用于跟踪训练过程的指标，包括损失和准确率。
        指标按顺序排列：先输出损失，再输出准确率。
        """
        # self.metrics 保持顺序
        # 先输出损失
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")  # 掩码损失
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")  # 响应损失
        # 再输出准确率
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")  # 掩码准确率
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")  # 响应准确率

    def _update_metrics_to_log(self) -> None:
        """更新要记录的指标。

        该方法根据配置和日志级别确定哪些指标需要记录。
        掩码语言模型相关的指标只有在启用时才记录。
        """
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG  # 检查是否为调试级别

        if self.config[MASKED_LM]:  # 如果启用了掩码语言模型
            self.metrics_to_log.append("m_acc")  # 添加掩码准确率
            if debug_log_level:  # 如果是调试级别
                self.metrics_to_log.append("m_loss")  # 添加掩码损失

        self.metrics_to_log.append("r_acc")  # 添加响应准确率
        if debug_log_level:  # 如果是调试级别
            self.metrics_to_log.append("r_loss")  # 添加响应损失

        self._log_metric_info()  # 记录指标信息

    def _log_metric_info(self) -> None:
        """记录指标信息。

        该方法记录训练过程中将要记录的指标信息，帮助用户了解训练状态。
        """
        metric_name = {"t": "total", "m": "mask", "r": "response"}  # 指标名称映射
        logger.debug("Following metrics will be logged during training: ")  # 记录指标信息
        for metric in self.metrics_to_log:  # 遍历要记录的指标
            parts = metric.split("_")  # 分割指标名称
            name = f"{metric_name[parts[0]]} {parts[1]}"  # 构建可读的指标名称
            logger.debug(f"  {metric} ({name})")  # 记录指标

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:
        """更新标签相关指标。

        该方法更新响应选择相关的损失和准确率指标。

        Args:
            loss: 损失值
            acc: 准确率值
        """
        self.response_loss.update_state(loss)  # 更新响应损失
        self.response_acc.update_state(acc)  # 更新响应准确率


class DIET2DIET(DIET):
    """DIET2DIET 转换器实现。

    该类实现了文本到文本（Text-to-Text）的响应选择架构。
    它使用 DIET 模型将用户输入文本和候选响应文本都转换为嵌入向量，
    然后计算它们之间的相似度，从而选择最合适的响应。

    主要特点：
    - 用户输入和候选响应都使用完整的文本特征（序列和句子特征）
    - 通过 Transformer 架构处理文本特征
    - 支持共享隐藏层权重以节省参数
    - 支持掩码语言模型训练
    """

    def _check_data(self) -> None:
        """检查数据完整性。

        该方法检查训练数据是否包含必要的特征，并验证配置的一致性。
        如果数据不完整或配置不一致，会抛出相应的异常。

        Raises:
            InvalidConfigException: 如果缺少必要的特征
            ValueError: 如果配置不一致
        """
        if TEXT not in self.data_signature:  # 如果缺少文本特征
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:  # 如果缺少标签特征
            raise InvalidConfigException(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if (
            self.config[SHARE_HIDDEN_LAYERS]  # 如果启用了共享隐藏层
            and self.data_signature[TEXT][SENTENCE]  # 并且文本句子特征
            != self.data_signature[LABEL][SENTENCE]  # 与标签句子特征不匹配
        ):
            raise ValueError(
                "If hidden layer weights are shared, data signatures "
                "for text_features and label_features must coincide."
            )

    def _create_metrics(self) -> None:
        """创建训练指标。

        该方法创建用于跟踪训练过程的指标，包括损失和准确率。
        指标按顺序排列：先输出损失，再输出准确率。
        """
        # self.metrics 保持顺序
        # 先输出损失
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")  # 掩码损失
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")  # 响应损失
        # 再输出准确率
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")  # 掩码准确率
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")  # 响应准确率

    def _update_metrics_to_log(self) -> None:
        """更新要记录的指标。

        该方法根据配置和日志级别确定哪些指标需要记录。
        掩码语言模型相关的指标只有在启用时才记录。
        """
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG  # 检查是否为调试级别

        if self.config[MASKED_LM]:  # 如果启用了掩码语言模型
            self.metrics_to_log.append("m_acc")  # 添加掩码准确率
            if debug_log_level:  # 如果是调试级别
                self.metrics_to_log.append("m_loss")  # 添加掩码损失

        self.metrics_to_log.append("r_acc")  # 添加响应准确率
        if debug_log_level:  # 如果是调试级别
            self.metrics_to_log.append("r_loss")  # 添加响应损失

        self._log_metric_info()  # 记录指标信息

    def _log_metric_info(self) -> None:
        """记录指标信息。

        该方法记录训练过程中将要记录的指标信息，帮助用户了解训练状态。
        """
        metric_name = {"t": "total", "m": "mask", "r": "response"}  # 指标名称映射
        logger.debug("Following metrics will be logged during training: ")  # 记录指标信息
        for metric in self.metrics_to_log:  # 遍历要记录的指标
            parts = metric.split("_")  # 分割指标名称
            name = f"{metric_name[parts[0]]} {parts[1]}"  # 构建可读的指标名称
            logger.debug(f"  {metric} ({name})")  # 记录指标

    def _prepare_layers(self) -> None:
        """准备模型层。

        该方法准备用于处理文本和标签特征的神经网络层。
        它创建序列层来处理不同的特征类型，并可选地设置掩码语言模型训练。
        """
        self.text_name = TEXT  # 设置文本属性名
        self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL  # 根据配置设置标签属性名

        # 为用户文本和响应文本准备层，这些层组合不同的特征类型，
        # 使用转换器嵌入所有内容，并可选地执行掩码语言建模。
        # 为标签特征省略输入丢弃。
        label_config = self.config.copy()  # 复制配置
        label_config.update({SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False})  # 禁用标签特征的输入丢弃
        for attribute, config in [
            (self.text_name, self.config),  # 文本特征使用原始配置
            (self.label_name, label_config),  # 标签特征使用修改后的配置
        ]:
            self._tf_layers[
                f"sequence_layer.{attribute}"
            ] = rasa_layers.RasaSequenceLayer(
                attribute, self.data_signature[attribute], config
            )  # 创建序列层

        if self.config[MASKED_LM]:  # 如果启用了掩码语言模型
            self._prepare_mask_lm_loss(self.text_name)  # 准备掩码语言模型损失

        self._prepare_label_classification_layers(predictor_attribute=self.text_name)  # 准备标签分类层

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """创建所有标签的嵌入。

        该方法为所有标签创建嵌入向量，用于响应选择。
        它首先获取标签ID，然后通过序列层处理标签特征，最后提取句子级别的嵌入。

        Returns:
            包含标签ID和嵌入向量的元组
        """
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]  # 获取所有标签ID

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )  # 获取序列特征长度

        # 组合所有特征类型并使用转换器嵌入
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                self.tf_label_data[LABEL][SEQUENCE],  # 序列特征
                self.tf_label_data[LABEL][SENTENCE],  # 句子特征
                sequence_feature_lengths,  # 序列长度
            ),
            training=self._training,  # 训练模式
        )

        # 最后一个标记取自具有真实特征的最后一个位置，由以下因素确定：
        # - 真实标记的数量，即序列级特征的序列长度
        # - 句子级特征的存在或不存在（反映在这些特征的有效序列长度为1或0）
        # 我们需要组合两个长度来正确获取最后一个位置
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            self.tf_label_data, LABEL
        )  # 获取句子特征长度
        sentence_label = self._last_token(
            label_transformed, sequence_feature_lengths + sentence_feature_lengths
        )  # 提取最后一个标记

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](sentence_label)  # 创建标签嵌入

        return all_label_ids, all_labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """计算给定批次的损失。

        该方法计算响应选择模型的损失，包括掩码语言模型损失（如果启用）
        和响应选择损失。它处理文本和标签特征，并计算相应的损失。

        Args:
            batch_in: 输入批次数据

        Returns:
            批次的损失值
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)  # 转换批次数据格式

        # 处理文本的所有特征
        sequence_feature_lengths_text = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )  # 获取文本序列特征长度
        (
            text_transformed,  # 转换后的文本特征
            text_in,  # 原始文本输入
            _,  # 未使用的输出
            text_seq_ids,  # 文本序列ID
            mlm_mask_booleanean_text,  # 掩码语言模型掩码
            _,  # 未使用的输出
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],  # 文本序列特征
                tf_batch_data[TEXT][SENTENCE],  # 文本句子特征
                sequence_feature_lengths_text,  # 序列长度
            ),
            training=self._training,  # 训练模式
        )

        # 处理标签的所有特征
        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )  # 获取标签序列特征长度
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                tf_batch_data[LABEL][SEQUENCE],  # 标签序列特征
                tf_batch_data[LABEL][SENTENCE],  # 标签句子特征
                sequence_feature_lengths_label,  # 序列长度
            ),
            training=self._training,  # 训练模式
        )

        losses = []  # 损失列表

        if self.config[MASKED_LM]:  # 如果启用了掩码语言模型
            loss, acc = self._mask_loss(
                text_transformed,  # 转换后的文本特征
                text_in,  # 原始文本输入
                text_seq_ids,  # 文本序列ID
                mlm_mask_booleanean_text,  # 掩码语言模型掩码
                self.text_name,  # 文本属性名
            )  # 计算掩码语言模型损失

            self.mask_loss.update_state(loss)  # 更新掩码损失
            self.mask_acc.update_state(acc)  # 更新掩码准确率
            losses.append(loss)  # 添加损失到列表

        # 获取用于标签分类的句子特征向量。向量从具有真实特征的最后一个位置提取。
        # 为了确定这个位置，我们组合序列级和句子级特征的序列长度。
        sentence_feature_lengths_text = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )  # 获取文本句子特征长度
        sentence_vector_text = self._last_token(
            text_transformed,  # 转换后的文本特征
            sequence_feature_lengths_text + sentence_feature_lengths_text,  # 组合长度
        )  # 提取最后一个标记

        # 以相同方式提取标签属性的句子向量
        sentence_feature_lengths_label = self._get_sentence_feature_lengths(
            tf_batch_data, LABEL
        )  # 获取标签句子特征长度
        sentence_vector_label = self._last_token(
            label_transformed,  # 转换后的标签特征
            sequence_feature_lengths_label + sentence_feature_lengths_label,  # 组合长度
        )  # 提取最后一个标记
        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]  # 获取标签ID

        loss, acc = self._calculate_label_loss(
            sentence_vector_text, sentence_vector_label, label_ids
        )  # 计算标签损失
        self.response_loss.update_state(loss)  # 更新响应损失
        self.response_acc.update_state(acc)  # 更新响应准确率
        losses.append(loss)  # 添加损失到列表

        return tf.math.add_n(losses)  # 返回总损失

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """预测给定批次的输出。

        该方法对输入批次进行预测，返回响应选择的相似度分数。
        它处理文本特征，计算与所有标签的相似度，并返回预测结果。

        Args:
            batch_in: 输入批次数据

        Returns:
            包含预测结果的字典，包括相似度分数和诊断数据
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )  # 转换批次数据格式

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )  # 获取序列特征长度
        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],  # 文本序列特征
                tf_batch_data[TEXT][SENTENCE],  # 文本句子特征
                sequence_feature_lengths,  # 序列长度
            ),
            training=self._training,  # 训练模式
        )  # 处理文本特征

        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,  # 注意力权重
                "text_transformed": text_transformed,  # 转换后的文本特征
            }
        }  # 创建预测结果字典

        if self.all_labels_embed is None:  # 如果标签嵌入未创建
            _, self.all_labels_embed = self._create_all_labels()  # 创建所有标签的嵌入

        # 获取用于意图分类的句子特征向量
        sentence_vector = self._last_token(text_transformed, sequence_feature_lengths)  # 提取最后一个标记
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)  # 创建句子嵌入

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],  # 句子嵌入（添加维度）
            self.all_labels_embed[tf.newaxis, :, :],  # 所有标签嵌入（添加维度）
        )  # 计算相似度和置信度
        predictions["i_scores"] = scores  # 添加相似度分数

        return predictions
