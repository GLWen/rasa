# =============================================================================
# Rasa NLU DIET Classifier 双意图和实体转换器分类器模块
# 本模块实现了 DIET (Dual Intent and Entity Transformer) 分类器，
# 用于同时进行意图分类和实体提取的多任务学习模型
# =============================================================================

# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入标准库模块
import copy  # 深拷贝功能，用于创建对象的完全独立副本
import logging  # 日志记录模块，用于记录程序运行状态和调试信息
from collections import defaultdict  # 默认字典，提供默认值的字典类型
from pathlib import Path  # 路径处理模块，提供面向对象的文件系统路径操作
from typing import Any, Dict, List, Optional, Text, Tuple, Union, TypeVar, Type  # 类型注解模块，提供类型提示功能

# 导入科学计算库
import numpy as np  # 数值计算库，提供多维数组和数学运算功能
import scipy.sparse  # 稀疏矩阵库，用于高效存储和处理稀疏数据
import tensorflow as tf  # 深度学习框架，用于构建和训练神经网络模型

# 导入 Rasa 核心模块
from rasa.exceptions import ModelNotFound  # 模型未找到异常，当模型文件不存在时抛出
from rasa.nlu.featurizers.featurizer import Featurizer  # 特征化器基类，提供特征提取的通用接口
from rasa.engine.graph import ExecutionContext, GraphComponent  # 图执行上下文和组件，用于组件管理和执行控制
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.nlu.extractors.extractor import EntityExtractorMixin  # 实体提取器混入，提供实体提取的通用接口
from rasa.nlu.classifiers.classifier import IntentClassifier  # 意图分类器基类，提供意图分类的通用接口
import rasa.shared.utils.io  # 共享工具模块，提供文件读写和序列化功能
import rasa.nlu.utils.bilou_utils as bilou_utils  # BILOU 标签工具，用于实体标签的 BILOU 标记处理
from rasa.shared.constants import DIAGNOSTIC_DATA  # 诊断数据常量，用于存储模型诊断信息
from rasa.nlu.extractors.extractor import EntityTagSpec  # 实体标签规范，定义实体标签的格式和映射
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH  # 标签排名长度，定义标签排名的最大长度
from rasa.utils import train_utils  # 训练工具，提供模型训练相关的辅助函数
from rasa.utils.tensorflow import rasa_layers  # Rasa TensorFlow 层，提供自定义的 TensorFlow 层实现
from rasa.utils.tensorflow.feature_array import (
    FeatureArray,  # 特征数组类，用于封装和管理特征数据
    serialize_nested_feature_arrays,  # 序列化嵌套特征数组，用于模型持久化
    deserialize_nested_feature_arrays,  # 反序列化嵌套特征数组，用于模型加载
)
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel  # Rasa 模型基类，提供模型的基础功能
from rasa.utils.tensorflow.model_data import (
    RasaModelData,  # Rasa 模型数据类，用于管理训练和预测数据
    FeatureSignature,  # 特征签名类，用于描述特征的结构和类型
)
from rasa.nlu.constants import TOKENS_NAMES, DEFAULT_TRANSFORMER_SIZE  # NLU 常量，定义标记名称和默认转换器大小

# 导入 NLU 常量
from rasa.shared.nlu.constants import (
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,  # 按逗号分割实体的默认值，用于实体分割配置
    TEXT,  # 文本常量，表示消息的文本内容属性
    INTENT,  # 意图常量，表示消息的意图属性
    INTENT_RESPONSE_KEY,  # 意图响应键，用于响应选择器中的意图标识
    ENTITIES,  # 实体常量，表示消息的实体属性
    ENTITY_ATTRIBUTE_TYPE,  # 实体属性类型，表示实体的类型信息
    ENTITY_ATTRIBUTE_GROUP,  # 实体属性组，表示实体的分组信息
    ENTITY_ATTRIBUTE_ROLE,  # 实体属性角色，表示实体的角色信息
    NO_ENTITY_TAG,  # 非实体标签，表示非实体的标记
    SPLIT_ENTITIES_BY_COMMA,  # 按逗号分割实体，用于实体分割配置
)

# 导入异常和训练数据
from rasa.shared.exceptions import InvalidConfigException  # 无效配置异常，当配置参数无效时抛出
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息

# 导入 TensorFlow 常量
from rasa.utils.tensorflow.constants import (
    DROP_SMALL_LAST_BATCH,  # 丢弃小批次标志，决定是否丢弃最后一个不完整的批次
    LABEL,  # 标签常量，用于标识标签相关的数据
    IDS,  # ID 常量，用于标识数据项的唯一标识符
    HIDDEN_LAYERS_SIZES,  # 隐藏层大小配置，定义神经网络隐藏层的尺寸
    RENORMALIZE_CONFIDENCES,  # 重新归一化置信度标志，决定是否对置信度进行重新归一化
    SHARE_HIDDEN_LAYERS,  # 共享隐藏层标志，决定是否在不同任务间共享隐藏层权重
    TRANSFORMER_SIZE,  # 转换器大小，定义 Transformer 模型的隐藏单元数
    NUM_TRANSFORMER_LAYERS,  # 转换器层数，定义 Transformer 模型的层数
    NUM_HEADS,  # 注意力头数，定义多头注意力机制的头数
    BATCH_SIZES,  # 批次大小配置，定义训练时的批次大小范围
    BATCH_STRATEGY,  # 批次策略，定义批次创建的策略（序列或平衡）
    EPOCHS,  # 训练轮数，定义模型训练的轮数
    RANDOM_SEED,  # 随机种子，用于确保结果的可重现性
    LEARNING_RATE,  # 学习率，定义优化器的学习率
    RANKING_LENGTH,  # 排名长度，定义返回的标签排名数量
    LOSS_TYPE,  # 损失类型，定义损失函数的类型（交叉熵或边距）
    SIMILARITY_TYPE,  # 相似度类型，定义相似度计算的方式（自动、余弦或内积）
    NUM_NEG,  # 负样本数，定义训练时使用的负样本数量
    SPARSE_INPUT_DROPOUT,  # 稀疏输入丢弃率，定义稀疏输入的丢弃概率
    DENSE_INPUT_DROPOUT,  # 密集输入丢弃率，定义密集输入的丢弃概率
    MASKED_LM,  # 掩码语言模型标志，决定是否使用掩码语言模型训练
    ENTITY_RECOGNITION,  # 实体识别标志，决定是否进行实体识别任务
    TENSORBOARD_LOG_DIR,  # TensorBoard 日志目录，用于存储训练日志
    INTENT_CLASSIFICATION,  # 意图分类标志，决定是否进行意图分类任务
    EVAL_NUM_EXAMPLES,  # 评估样本数，定义用于验证的样本数量
    EVAL_NUM_EPOCHS,  # 评估轮数，定义验证评估的频率
    UNIDIRECTIONAL_ENCODER,  # 单向编码器标志，决定是否使用单向编码器
    DROP_RATE,  # 丢弃率，定义一般层的丢弃概率
    DROP_RATE_ATTENTION,  # 注意力丢弃率，定义注意力层的丢弃概率
    CONNECTION_DENSITY,  # 连接密度，定义层中可训练权重的比例
    NEGATIVE_MARGIN_SCALE,  # 负边距缩放，定义负样本边距损失的重要性
    REGULARIZATION_CONSTANT,  # 正则化常数，定义正则化项的权重
    SCALE_LOSS,  # 缩放损失标志，决定是否按置信度缩放损失
    USE_MAX_NEG_SIM,  # 使用最大负相似度标志，决定是否使用最大负相似度
    MAX_NEG_SIM,  # 最大负相似度，定义负样本的最大相似度阈值
    MAX_POS_SIM,  # 最大正相似度，定义正样本的最大相似度阈值
    EMBEDDING_DIMENSION,  # 嵌入维度，定义嵌入向量的维度
    BILOU_FLAG,  # BILOU 标志，决定是否使用 BILOU 标记方案
    KEY_RELATIVE_ATTENTION,  # 键相对注意力标志，决定是否在注意力中使用键相对嵌入
    VALUE_RELATIVE_ATTENTION,  # 值相对注意力标志，决定是否在注意力中使用值相对嵌入
    MAX_RELATIVE_POSITION,  # 最大相对位置，定义相对嵌入的最大位置范围
    AUTO,  # 自动常量，用于自动选择参数值
    BALANCED,  # 平衡常量，用于平衡批次策略
    CROSS_ENTROPY,  # 交叉熵常量，用于交叉熵损失函数
    TENSORBOARD_LOG_LEVEL,  # TensorBoard 日志级别，定义日志记录的详细程度
    CONCAT_DIMENSION,  # 连接维度，定义特征连接时的维度
    FEATURIZERS,  # 特征化器配置，定义使用的特征化器列表
    CHECKPOINT_MODEL,  # 检查点模型标志，决定是否保存模型检查点
    SEQUENCE,  # 序列常量，用于标识序列特征
    SENTENCE,  # 句子常量，用于标识句子特征
    SEQUENCE_LENGTH,  # 序列长度常量，用于标识序列长度信息
    DENSE_DIMENSION,  # 密集维度，定义密集特征的维度
    MASK,  # 掩码常量，用于标识掩码信息
    CONSTRAIN_SIMILARITIES,  # 约束相似度标志，决定是否约束相似度值
    MODEL_CONFIDENCE,  # 模型置信度类型，定义置信度计算的方式
    SOFTMAX,  # Softmax 常量，用于 Softmax 激活函数
    RUN_EAGERLY,  # 急切运行标志，决定是否使用急切执行模式
)

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)

# 特征类型常量
SPARSE = "sparse"  # 稀疏特征类型，用于标识稀疏矩阵特征
DENSE = "dense"    # 密集特征类型，用于标识密集数组特征

# 标签相关常量
LABEL_KEY = LABEL      # 标签键，用于标识标签数据的主键
LABEL_SUB_KEY = IDS    # 标签子键，用于标识标签数据的子键

# 可能的实体标签类型，定义实体可以具有的所有属性类型
POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]

# DIET 分类器类型变量，用于类型注解中的泛型约束
DIETClassifierT = TypeVar("DIETClassifierT", bound="DIETClassifier")


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,  # 意图分类器组件类型，用于意图识别任务
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,   # 实体提取器组件类型，用于实体识别任务
    ],
    is_trainable=True,  # 可训练组件，支持模型训练和微调
)
class DIETClassifier(GraphComponent, IntentClassifier, EntityExtractorMixin):
    """用于意图分类和实体提取的多任务学习模型。

    DIET 是双意图和实体转换器（Dual Intent and Entity Transformer），
    它基于共享的 Transformer 架构同时进行意图分类和实体提取。
    
    核心架构特点：
    1. 共享 Transformer：两个任务共享同一个 Transformer 编码器
    2. 实体识别：通过条件随机场（CRF）层在转换器输出上预测实体标签序列
    3. 意图分类：使用 __CLS__ 标记的转换器输出进行意图分类
    4. 语义空间：意图标签和文本嵌入到统一的语义向量空间中
    5. 损失函数：使用点积损失最大化与目标标签的相似性，最小化与负样本的相似性
    
    这种设计使得模型能够同时学习文本的全局语义（意图）和局部结构（实体），
    提高了多任务学习的效率和效果。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """获取此组件运行前必须包含在管道中的组件类型。
        
        该方法定义了组件的依赖关系，确保在模型训练和预测之前
        文本已经被正确特征化，为模型提供必要的输入特征。
        
        Returns:
            必需的组件类型列表，包含特征化器组件
        """
        return [Featurizer]  # 需要特征化器组件，用于将文本转换为数值特征

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置参数。
        
        该方法定义了 DIETClassifier 的所有可配置参数及其默认值。
        配置参数涵盖了模型架构、训练参数、正则化设置、评估配置等方面。
        
        注意：更改默认参数时请确保更新相关文档。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        # 更改默认参数时请确保更新文档
        return {
            # =================================================================
            # 神经网络架构配置
            # =================================================================
            # 用户消息和标签的嵌入层之前的隐藏层大小配置
            # 隐藏层数等于对应列表的长度，空列表表示不使用隐藏层
            HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},  # 隐藏层大小，分别为文本和标签定义隐藏层尺寸
            # 是否在用户消息和标签之间共享隐藏层权重
            # True 表示共享权重，False 表示独立权重
            SHARE_HIDDEN_LAYERS: False,  # 共享隐藏层标志，影响模型参数数量和训练复杂度
            # 转换器中的单元数，定义 Transformer 模型的隐藏维度
            TRANSFORMER_SIZE: DEFAULT_TRANSFORMER_SIZE,  # 转换器大小，影响模型的表达能力和计算复杂度
            # 转换器层数，定义 Transformer 模型的深度
            NUM_TRANSFORMER_LAYERS: 2,  # 转换器层数，更多层数通常提供更强的表达能力但增加计算成本
            # 转换器中的注意力头数，定义多头注意力机制的头数
            NUM_HEADS: 4,  # 注意力头数，更多头数可以捕获不同类型的注意力模式
            # 如果为 'True'，在注意力中使用键相对嵌入，提供位置感知的注意力
            KEY_RELATIVE_ATTENTION: False,  # 键相对注意力标志，影响注意力的位置编码方式
            # 如果为 'True'，在注意力中使用值相对嵌入，提供位置感知的值计算
            VALUE_RELATIVE_ATTENTION: False,  # 值相对注意力标志，影响注意力值的计算方式
            # 相对嵌入的最大位置范围，仅在键或值相对注意力开启时生效
            MAX_RELATIVE_POSITION: 5,  # 最大相对位置，定义相对位置编码的最大范围
            # 使用单向或双向编码器，单向编码器只能看到前面的标记
            UNIDIRECTIONAL_ENCODER: False,  # 单向编码器标志，影响模型对序列信息的利用方式
            # =================================================================
            # 训练参数配置
            # =================================================================
            # 初始和最终批次大小配置，批次大小将在每个轮次线性增加
            # 从初始值逐渐增加到最终值，有助于稳定训练过程
            BATCH_SIZES: [64, 256],  # 批次大小范围，[初始批次大小, 最终批次大小]
            # 创建批次时使用的策略，影响批次中样本的组成方式
            # 'sequence' 表示按序列顺序，'balanced' 表示平衡采样
            BATCH_STRATEGY: BALANCED,  # 批次策略，平衡策略有助于提高训练稳定性
            # 训练的轮数，定义模型训练的总轮数
            EPOCHS: 300,  # 训练轮数，更多轮数通常提供更好的性能但增加训练时间
            # 设置随机种子为任何 'int' 以获得可重现的结果
            # None 表示使用随机种子，具体数值确保结果可重现
            RANDOM_SEED: None,  # 随机种子，用于控制随机性，确保结果可重现
            # 优化器的初始学习率，控制参数更新的步长
            LEARNING_RATE: 0.001,  # 学习率，影响模型收敛速度和最终性能
            # =================================================================
            # 嵌入参数配置
            # =================================================================
            # 嵌入向量的维度大小，定义语义空间的维度
            EMBEDDING_DIMENSION: 20,  # 嵌入维度，影响模型的表达能力和计算复杂度
            # 用于稀疏特征的密集维度，将稀疏特征转换为密集表示
            DENSE_DIMENSION: {TEXT: 128, LABEL: 20},  # 密集维度，分别为文本和标签定义密集特征维度
            # 用于连接序列和句子特征的默认维度，定义特征融合后的维度
            CONCAT_DIMENSION: {TEXT: 128, LABEL: 20},  # 连接维度，用于特征连接时的维度控制
            # 错误标签的数量，算法将在训练期间最小化它们与用户输入的相似性
            NUM_NEG: 20,  # 负样本数，影响对比学习的难度和效果
            # 使用的相似性度量类型，可以是 'auto'、'cosine' 或 'inner'
            SIMILARITY_TYPE: AUTO,  # 相似度类型，自动选择最适合的相似度计算方法
            # 损失函数的类型，可以是 'cross_entropy' 或 'margin'
            LOSS_TYPE: CROSS_ENTROPY,  # 损失类型，交叉熵损失适用于多分类任务
            # 应报告置信度的顶级意图数量，控制返回的意图排名数量
            # 如果应报告所有意图的置信度，则设置为 0
            RANKING_LENGTH: LABEL_RANKING_LENGTH,  # 排名长度，定义返回的意图排名数量
            # 指示算法应尝试使正确标签的嵌入向量有多相似
            # 对于 'cosine' 相似度类型，应为 0.0 < ... < 1.0
            MAX_POS_SIM: 0.8,  # 最大正相似度，控制正样本的相似度目标
            # 错误标签的最大负相似度，控制负样本的相似度阈值
            # 对于 'cosine' 相似度类型，应为 -1.0 < ... < 1.0
            MAX_NEG_SIM: -0.4,  # 最大负相似度，控制负样本的相似度上限
            # 如果为 'True'，算法仅最小化错误意图标签上的最大相似度
            # 仅在 'loss_type' 设置为 'margin' 时使用
            USE_MAX_NEG_SIM: True,  # 使用最大负相似度标志，影响边距损失的计算方式
            # 如果为 'True'，按正确预测的置信度反比例缩放损失
            SCALE_LOSS: False,  # 缩放损失标志，影响损失函数的权重分配
            # =================================================================
            # 正则化参数配置
            # =================================================================
            # 正则化的规模，控制 L2 正则化项的权重
            REGULARIZATION_CONSTANT: 0.002,  # 正则化常数，防止过拟合，提高模型泛化能力
            # 最小化不同标签嵌入之间最大相似度的重要性规模
            # 仅在 'loss_type' 设置为 'margin' 时使用
            NEGATIVE_MARGIN_SCALE: 0.8,  # 负边距缩放，控制负样本边距损失的重要性
            # 编码器的丢弃率，在训练时随机丢弃部分神经元
            DROP_RATE: 0.2,  # 丢弃率，防止过拟合，提高模型鲁棒性
            # 注意力的丢弃率，在注意力层中应用丢弃
            DROP_RATE_ATTENTION: 0,  # 注意力丢弃率，通常设置为 0 以保持注意力机制完整性
            # 内部层中可训练权重的比例，控制模型的稀疏性
            CONNECTION_DENSITY: 0.2,  # 连接密度，影响模型的参数数量和计算复杂度
            # 如果为 'True'，对稀疏输入张量应用丢弃
            SPARSE_INPUT_DROPOUT: True,  # 稀疏输入丢弃标志，对稀疏特征应用丢弃
            # 如果为 'True'，对密集输入张量应用丢弃
            DENSE_INPUT_DROPOUT: True,  # 密集输入丢弃标志，对密集特征应用丢弃
            # =================================================================
            # 评估参数配置
            # =================================================================
            # 计算验证准确性的频率，控制验证评估的间隔
            # 小值可能会损害性能，因为频繁验证会增加计算开销
            EVAL_NUM_EPOCHS: 20,  # 评估轮数，每 20 个轮次进行一次验证评估
            # 用于保留验证集的示例数量，控制验证集的大小
            # 大值可能会损害性能，例如模型准确性，因为减少了训练数据
            # 设置为 0 表示无验证，跳过验证过程
            EVAL_NUM_EXAMPLES: 0,  # 评估样本数，0 表示不进行验证
            # =================================================================
            # 模型配置参数
            # =================================================================
            # 如果为 'True'，训练意图分类并预测意图
            INTENT_CLASSIFICATION: True,  # 意图分类标志，启用意图分类任务
            # 如果为 'True'，训练命名实体识别并预测实体
            ENTITY_RECOGNITION: True,  # 实体识别标志，启用实体识别任务
            # 如果为 'True'，输入消息的随机标记将被掩码，模型应预测这些标记
            MASKED_LM: False,  # 掩码语言模型标志，启用掩码语言模型预训练
            # 'BILOU_flag' 确定是否使用 BILOU 标记方案
            # 如果设置为 'True'，标记更严格，但每个实体需要更多示例
            # 经验法则：每个实体应该有超过 100 个示例
            BILOU_FLAG: True,  # BILOU 标志，使用 BILOU 标记方案进行实体识别
            # 如果要使用 tensorboard 可视化训练和验证指标
            # 请将此选项设置为有效的输出目录
            TENSORBOARD_LOG_DIR: None,  # TensorBoard 日志目录，None 表示不记录日志
            # 定义何时记录 tensorboard 的训练指标
            # 可以在每个轮次后或每个训练步骤后
            # 有效值：'epoch' 和 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",  # TensorBoard 日志级别，按轮次记录日志
            # 执行模型检查点，定期保存模型状态
            CHECKPOINT_MODEL: False,  # 检查点模型标志，是否保存训练检查点
            # 指定用作序列和句子特征的特征化器
            # 默认使用管道中的所有特征化器
            FEATURIZERS: [],  # 特征化器列表，空列表表示使用所有可用的特征化器
            # 按逗号分割实体，这对于成分列表等有意义
            # 但对于地址的各个部分没有意义
            SPLIT_ENTITIES_BY_COMMA: True,  # 按逗号分割实体标志，是否按逗号分割实体
            # 如果为 'True'，对所有相似性项应用 sigmoid 并将其添加到损失函数中
            # 以确保相似性值近似有界。仅在交叉熵损失内部使用
            CONSTRAIN_SIMILARITIES: False,  # 约束相似度标志，是否约束相似度值范围
            # 推理期间返回的模型置信度。目前唯一可能的值是 `softmax`
            MODEL_CONFIDENCE: SOFTMAX,  # 模型置信度类型，使用 Softmax 计算置信度
            # 确定所选顶级意图的置信度是否应重新归一化，使其总和为 1
            # 默认情况下，我们不重新归一化，按原样返回顶级意图的置信度
            # 注意：重新归一化仅在通过 `softmax` 生成置信度时才有意义
            RENORMALIZE_CONFIDENCES: False,  # 重新归一化置信度标志，是否重新归一化置信度
            # 确定是否构建模型图
            # 当模型仅训练或推断几个步骤时，这是有利的
            # 因为图的编译往往比运行它花费更多时间
            # 建议不要调整优化参数
            RUN_EAGERLY: False,  # 急切运行标志，是否使用急切执行模式
            # 确定如果最后一个批次包含少于一半批次大小的示例是否应丢弃
            DROP_SMALL_LAST_BATCH: False,  # 丢弃小批次标志，是否丢弃不完整的最后批次
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
        """初始化 DIET 分类器实例。
        
        该方法创建 DIET 分类器的新实例，设置所有必要的属性和配置。
        它验证配置参数，初始化模型相关属性，并准备训练和预测所需的数据结构。
        
        Args:
            config: 组件配置字典，包含所有模型和训练参数
            model_storage: 模型存储接口，用于模型的持久化和加载
            resource: 资源管理对象，用于标识和管理模型资源
            execution_context: 执行上下文，包含节点名称和执行模式信息
            index_label_id_mapping: 索引到标签ID的映射，用于标签的编码和解码
            entity_tag_specs: 实体标签规范列表，定义实体标签的格式和映射
            model: Rasa模型实例，预训练的模型（用于微调或预测）
            sparse_feature_sizes: 稀疏特征大小，定义稀疏特征的维度信息
        """
        # 检查是否配置了训练轮数，如果没有配置则发出警告
        if EPOCHS not in config:
            rasa.shared.utils.io.raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        # 设置基本属性
        self.component_config = config  # 组件配置，存储所有配置参数
        self._model_storage = model_storage  # 模型存储，用于模型的持久化操作
        self._resource = resource  # 资源管理，用于标识和管理模型资源
        self._execution_context = execution_context  # 执行上下文，包含执行环境信息

        # 检查配置参数的有效性和一致性
        self._check_config_parameters()

        # 设置标签映射和实体标签规范
        self.index_label_id_mapping = index_label_id_mapping or {}  # 索引到标签ID的映射，用于标签转换
        self._entity_tag_specs = entity_tag_specs  # 实体标签规范，定义实体标签的格式

        # 设置模型实例
        self.model = model  # 模型实例，用于训练和预测

        # 设置检查点目录，如果启用了检查点功能
        self.tmp_checkpoint_dir = None
        if self.component_config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

        # 初始化数据相关属性
        self._label_data: Optional[RasaModelData] = None  # 标签数据，存储标签的特征信息
        self._data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None  # 数据示例，用于模型持久化

        # 初始化实体分割配置，设置实体分割的相关参数
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            self.component_config[SPLIT_ENTITIES_BY_COMMA],
            SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
        )

        # 设置微调模式和稀疏特征大小
        self.finetune_mode = self._execution_context.is_finetuning  # 微调模式标志，表示是否进行微调
        self._sparse_feature_sizes = sparse_feature_sizes  # 稀疏特征大小，定义稀疏特征的维度

    # =================================================================
    # 初始化辅助方法
    # =================================================================
    
    def _check_masked_lm(self) -> None:
        """检查掩码语言模型配置的有效性。
        
        该方法验证掩码语言模型配置是否与转换器层数配置兼容。
        如果转换器层数为 0，则不能使用掩码语言模型。
        
        Raises:
            ValueError: 如果配置不兼容
        """
        if (
            self.component_config[MASKED_LM]  # 如果启用了掩码语言模型
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0  # 但转换器层数为 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

    def _check_share_hidden_layers_sizes(self) -> None:
        """检查共享隐藏层配置的有效性。
        
        该方法验证当启用隐藏层权重共享时，所有隐藏层的大小是否一致。
        如果共享权重，则所有隐藏层的大小必须相同。
        
        Raises:
            ValueError: 如果隐藏层大小不一致
        """
        if self.component_config.get(SHARE_HIDDEN_LAYERS):  # 如果启用了隐藏层权重共享
            first_hidden_layer_sizes = next(
                iter(self.component_config[HIDDEN_LAYERS_SIZES].values())
            )
            # 检查所有隐藏层大小是否相同
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
        """检查并更新配置参数。
        
        该方法执行一系列配置检查和更新操作：
        1. 检查已弃用的选项
        2. 验证掩码语言模型配置
        3. 验证共享隐藏层配置
        4. 更新置信度类型
        5. 验证配置设置
        6. 更新相似度类型
        7. 更新评估参数
        """
        # 检查已弃用的配置选项
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        # 检查掩码语言模型配置
        self._check_masked_lm()
        # 检查共享隐藏层配置
        self._check_share_hidden_layers_sizes()

        # 更新置信度类型配置
        self.component_config = train_utils.update_confidence_type(
            self.component_config
        )

        # 验证配置设置的有效性
        train_utils.validate_configuration_settings(self.component_config)

        # 更新相似度类型配置
        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )
        # 更新评估参数配置
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
        """创建新的未训练组件实例。
        
        这是一个类方法，用于创建新的 DIETClassifier 实例。
        通常在训练开始时调用，创建一个全新的、未训练的组件。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            
        Returns:
            新的未训练的 DIETClassifier 实例
        """
        return cls(config, model_storage, resource, execution_context)

    @property
    def label_key(self) -> Optional[Text]:
        """获取标签键，如果意图分类已激活则返回标签键。
        
        Returns:
            标签键或 None（如果意图分类未激活）
        """
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @property
    def label_sub_key(self) -> Optional[Text]:
        """获取标签子键，如果意图分类已激活则返回标签子键。
        
        Returns:
            标签子键或 None（如果意图分类未激活）
        """
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        """获取模型类。
        
        Returns:
            DIET 模型类
        """
        return DIET

    # =================================================================
    # 训练数据辅助方法
    # =================================================================
    
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """创建标签ID到索引的映射字典。
        
        该方法从训练数据中提取所有唯一的标签ID，并为每个标签分配一个索引。
        这用于将标签转换为数值，以便模型处理。
        
        Args:
            training_data: 训练数据对象
            attribute: 属性名称（如 INTENT）
            
        Returns:
            标签ID到索引的映射字典
        """
        # 从训练数据中提取所有唯一的标签ID，排除 None 值
        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        # 为每个标签ID分配一个索引，按字母顺序排序以确保一致性
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        """反转映射字典。
        
        该方法将键值对反转，用于将索引映射回标签ID。
        
        Args:
            mapping: 原始映射字典
            
        Returns:
            反转后的映射字典
        """
        return {value: key for key, value in mapping.items()}

    def _create_entity_tag_specs(
        self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """创建实体标签规范及其相应的标签ID映射。
        
        该方法为每个可能的实体标签类型（类型、角色、组）创建标签规范。
        根据 BILOU 标志的设置，使用不同的标签映射策略。
        
        Args:
            training_data: 训练数据对象
            
        Returns:
            实体标签规范列表
        """
        _tag_specs = []  # 初始化标签规范列表

        # 遍历所有可能的实体标签类型
        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:  # 如果使用 BILOU 标记方案
                # 使用 BILOU 工具构建标签ID映射
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:  # 如果使用简单标记方案
                # 使用自定义方法构建标签ID映射
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            # 如果成功创建了标签映射，则添加到规范列表中
            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,  # 标签名称
                        tags_to_ids=tag_id_index_mapping,  # 标签到ID的映射
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),  # ID到标签的映射
                        num_tags=len(tag_id_index_mapping),  # 标签数量
                    )
                )

        return _tag_specs

    @staticmethod
    def _tag_id_index_mapping_for(
        tag_name: Text, training_data: TrainingData
    ) -> Optional[Dict[Text, int]]:
        """为指定标签名称创建标签ID到索引的映射。
        
        该方法根据标签名称从训练数据中提取相应的标签，并创建ID到索引的映射。
        支持实体类型、角色和组三种标签类型。
        
        Args:
            tag_name: 标签名称（ENTITY_ATTRIBUTE_TYPE、ENTITY_ATTRIBUTE_ROLE 或 ENTITY_ATTRIBUTE_GROUP）
            training_data: 训练数据对象
            
        Returns:
            标签ID到索引的映射字典，如果没有标签则返回 None
        """
        # 根据标签名称选择相应的标签集合
        if tag_name == ENTITY_ATTRIBUTE_ROLE:  # 如果是角色标签
            distinct_tags = training_data.entity_roles
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:  # 如果是组标签
            distinct_tags = training_data.entity_groups
        else:  # 如果是类型标签（默认）
            distinct_tags = training_data.entities

        # 排除非实体标签和 None 值
        distinct_tags = distinct_tags - {NO_ENTITY_TAG} - {None}

        # 如果没有标签，返回 None
        if not distinct_tags:
            return None

        # 为每个标签分配一个索引，从 1 开始（0 保留给非实体标签）
        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tags), 1)
        }
        # NO_ENTITY_TAG 对应非实体，应该对应索引 0
        # 这对于正确的填充预测是必需的
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        """为指定标签查找对应的示例消息。
        
        该方法在示例列表中查找具有指定属性值的消息。
        用于获取特定标签的训练示例。
        
        Args:
            label: 要查找的标签值
            examples: 示例消息列表
            attribute: 属性名称（如 INTENT、ACTION_NAME 等）
            
        Returns:
            匹配的示例消息，如果未找到则返回 None
        """
        for ex in examples:  # 遍历所有示例
            if ex.get(attribute) == label:  # 如果找到匹配的标签
                return ex
        return None  # 未找到匹配的示例

    def _check_labels_features_exist(
        self, labels_example: List[Message], attribute: Text
    ) -> bool:
        """检查所有标签是否都有特征设置。
        
        该方法验证所有标签示例是否都包含必要的特征。
        用于确保训练数据的完整性。
        
        Args:
            labels_example: 标签示例消息列表
            attribute: 属性名称
            
        Returns:
            如果所有标签都有特征则返回 True，否则返回 False
        """
        return all(
            label_example.features_present(
                attribute, self.component_config[FEATURIZERS]
            )
            for label_example in labels_example
        )

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Dict[Text, Union[scipy.sparse.spmatrix, np.ndarray]]:
        """从消息中提取特征。
        
        该方法从消息对象中提取稀疏和密集特征，包括序列特征和句子特征。
        同时验证特征维度的一致性。
        
        Args:
            message: 消息对象
            attribute: 属性名称（如 TEXT、INTENT 等）
            
        Returns:
            包含各种特征的字典
            
        Raises:
            ValueError: 当稀疏和密集特征的序列维度不匹配时
        """
        # 获取稀疏特征（序列和句子级别）
        (
            sparse_sequence_features,
            sparse_sentence_features,
        ) = message.get_sparse_features(attribute, self.component_config[FEATURIZERS])
        
        # 获取密集特征（序列和句子级别）
        dense_sequence_features, dense_sentence_features = message.get_dense_features(
            attribute, self.component_config[FEATURIZERS]
        )

        # 验证序列特征的维度一致性
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
        
        # 验证句子特征的维度一致性
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

        # 如果不使用 transformer 且不进行实体识别，为了加速训练，
        # 只使用句子特征作为特征向量。在这种设置下，我们不会使用序列特征。
        # 将这些特征传递到实际训练过程需要相当长的时间。
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute not in [INTENT, INTENT_RESPONSE_KEY]
        ):
            sparse_sequence_features = None
            dense_sequence_features = None

        # 构建输出特征字典
        out = {}

        # 添加稀疏句子特征
        if sparse_sentence_features is not None:
            out[f"{SPARSE}_{SENTENCE}"] = sparse_sentence_features.features
        
        # 添加稀疏序列特征
        if sparse_sequence_features is not None:
            out[f"{SPARSE}_{SEQUENCE}"] = sparse_sequence_features.features
        
        # 添加密集句子特征
        if dense_sentence_features is not None:
            out[f"{DENSE}_{SENTENCE}"] = dense_sentence_features.features
        
        # 添加密集序列特征
        if dense_sequence_features is not None:
            out[f"{DENSE}_{SEQUENCE}"] = dense_sequence_features.features

        return out

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """检查如果隐藏层共享时特征是否具有相同的维度。
        
        当启用隐藏层共享时，文本特征和标签特征必须具有相同的维度。
        该方法验证这种一致性，确保模型能够正确训练。
        
        Args:
            model_data: 模型数据对象
            
        Raises:
            ValueError: 当文本特征和标签特征维度不匹配时
        """
        if self.component_config.get(SHARE_HIDDEN_LAYERS):  # 如果启用了隐藏层共享
            # 获取文本和标签的句子特征数量
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            
            # 获取文本和标签的序列特征数量
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            # 检查句子特征维度是否匹配
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
        """收集预计算的编码特征。
        
        该方法从标签示例中提取特征，并将它们组织成序列特征和句子特征。
        用于处理已经预计算好的特征编码。
        
        Args:
            label_examples: 标签示例消息列表
            attribute: 属性名称，默认为 INTENT
            
        Returns:
            包含序列特征和句子特征的元组
        """
        features = defaultdict(list)  # 使用默认字典存储特征

        # 遍历所有标签示例，提取特征
        for e in label_examples:
            label_features = self._extract_features(e, attribute)
            for feature_key, feature_value in label_features.items():
                features[feature_key].append(feature_value)
        
        # 分离序列特征和句子特征
        sequence_features = []
        sentence_features = []
        for feature_name, feature_value in features.items():
            if SEQUENCE in feature_name:  # 如果是序列特征
                sequence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
            else:  # 如果是句子特征
                sentence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
        return sequence_features, sentence_features

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[FeatureArray]:
        """计算标签的独热编码表示。
        
        当没有找到标签特征时，该方法为标签计算默认的独热编码特征。
        每个标签对应一个独热向量，其中只有一个位置为1，其他位置为0。
        
        Args:
            labels_example: 标签示例消息列表
            
        Returns:
            包含独热编码特征的 FeatureArray 列表
        """
        logger.debug("No label features found. Computing default label features.")

        # 创建单位矩阵作为独热编码的基础
        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # 为独热标签添加序列维度
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
        """创建以词袋形式编码标签ID的矩阵。
        
        为每个标签找到一个训练示例，并从相应的消息对象中获取编码特征。
        如果特征已经计算过，则从消息对象中获取；否则计算标签的独热编码作为特征向量。
        
        Args:
            training_data: 训练数据对象
            label_id_dict: 标签名称到ID的映射字典
            attribute: 属性名称
            
        Returns:
            包含标签数据的 RasaModelData 对象
            
        Raises:
            ValueError: 当没有标签特征存在时
        """
        # 为每个标签收集一个示例
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # 根据标签索引对元组列表进行排序
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        
        # 收集特征，如果存在预计算的特征则使用，否则动态计算
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            sentence_features = self._compute_default_label_features(labels_example)

        # 创建标签数据对象
        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)
        
        # 检查是否有标签特征存在
        if label_data.does_feature_not_exist(
            LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        # 创建标签ID数组
        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # 显式添加最后一个维度到 label_ids
        # 以正确跟踪动态序列
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

        # 添加序列长度信息
        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        """使用默认的标签特征。
        
        该方法从预存储的标签数据中获取默认的标签特征。
        用于在预测时快速获取标签特征。
        
        Args:
            label_ids: 标签ID数组
            
        Returns:
            标签特征数组列表
        """
        if self._label_data is None:  # 如果没有标签数据
            return []

        # 获取句子级别的标签特征
        feature_arrays = self._label_data.get(LABEL, SENTENCE)
        all_label_features = feature_arrays[0]
        
        # 根据标签ID选择相应的特征
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
        """准备训练数据并创建 RasaModelData 对象。
        
        该方法将训练数据转换为模型可以使用的格式，包括特征提取、标签处理等。
        
        Args:
            training_data: 训练数据消息列表
            label_id_dict: 标签ID字典，可选
            label_attribute: 标签属性名称，可选
            training: 是否为训练模式，默认为 True
            
        Returns:
            准备好的 RasaModelData 对象
        """
        from rasa.utils.tensorflow import model_data_utils

        # 确定需要考虑的属性
        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # 在预测时没有意图标签，只在训练时添加
            attributes_to_consider.append(label_attribute)
        if (
            training
            and self.component_config[ENTITY_RECOGNITION]
            and self._entity_tag_specs
        ):
            # 只在训练时添加实体作为标签，且仅当有实体训练数据时
            attributes_to_consider.append(ENTITIES)

        # 在训练时只使用设置了标签属性的训练示例
        if training and label_attribute is not None:
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        # 过滤出有文本特征的训练数据
        training_data = [
            message
            for message in training_data
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]

        # 如果没有训练数据，返回空的 RasaModelData
        if not training_data:
            return RasaModelData()

        # 对训练示例进行特征化
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
        
        # 将特征转换为数据格式
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )

        # 创建模型数据对象
        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )
        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        
        # 当前实现尚未考虑更新标签属性的稀疏特征大小，因此移除它们
        sparse_feature_sizes = self._remove_label_sparse_feature_sizes(
            sparse_feature_sizes=sparse_feature_sizes, label_attribute=label_attribute
        )
        model_data.add_sparse_feature_sizes(sparse_feature_sizes)

        # 添加标签特征
        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # 确保训练和预测时所有键的顺序相同
        # 因为我们在从模型数据构建实际张量时依赖于键和子键的顺序
        model_data.sort()

        return model_data

    @staticmethod
    def _remove_label_sparse_feature_sizes(
        sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        label_attribute: Optional[Text] = None,
    ) -> Dict[Text, Dict[Text, List[int]]]:
        """移除标签属性的稀疏特征大小。
        
        该方法从稀疏特征大小字典中移除指定的标签属性。
        用于清理不需要的标签特征信息。
        
        Args:
            sparse_feature_sizes: 稀疏特征大小字典
            label_attribute: 要移除的标签属性名称
            
        Returns:
            清理后的稀疏特征大小字典
        """
        if label_attribute in sparse_feature_sizes:  # 如果标签属性存在
            del sparse_feature_sizes[label_attribute]  # 删除该属性
        return sparse_feature_sizes

    def _add_label_features(
        self,
        model_data: RasaModelData,
        training_data: List[Message],
        label_attribute: Text,
        label_id_dict: Dict[Text, int],
        training: bool = True,
    ) -> None:
        """添加标签特征到模型数据中。
        
        该方法将标签特征添加到模型数据中，包括标签ID和特征信息。
        支持意图分类和实体识别的标签处理。
        
        Args:
            model_data: 模型数据对象
            training_data: 训练数据消息列表
            label_attribute: 标签属性名称
            label_id_dict: 标签ID字典
            training: 是否为训练模式
        """
        label_ids = []
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # 收集所有标签ID
            for example in training_data:
                if example.get(label_attribute):
                    label_ids.append(label_id_dict[example.get(label_attribute)])
            
            # 显式添加最后一个维度到 label_ids
            # 以正确跟踪动态序列
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

        # 如果没有标签特征存在，从 _label_data 获取默认特征
        if (
            label_attribute
            and model_data.does_feature_not_exist(label_attribute, SENTENCE)
            and model_data.does_feature_not_exist(label_attribute, SEQUENCE)
        ):
            model_data.add_features(
                LABEL, SENTENCE, self._use_default_label_features(np.array(label_ids))
            )

        # 由于 label_attribute 可以有不同的值（如 INTENT 或 RESPONSE），
        # 将特征复制到 LABEL 键下，以便在模型内部更容易访问标签特征
        model_data.update_key(label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(label_attribute, MASK, LABEL, MASK)

        # 添加序列长度信息
        model_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """准备训练数据。
        
        对训练数据执行完整性检查，提取标签的编码。
        
        Args:
            training_data: 训练数据对象
            
        Returns:
            预处理后的 RasaModelData 对象
        """
        # 如果启用了 BILOU 标记和实体识别，应用 BILOU 模式
        if (
            self.component_config[BILOU_FLAG]
            and self.component_config[ENTITY_RECOGNITION]
        ):
            bilou_utils.apply_bilou_schema(training_data)

        # 获取标签ID到索引的映射
        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # 没有标签可以训练
            return RasaModelData()

        # 创建索引到标签ID的映射
        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        # 创建标签数据
        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )

        # 创建实体标签规范
        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        # 确定标签属性
        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )
        
        # 创建模型数据
        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        # 检查输入维度一致性
        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        """检查是否有足够的标签进行训练。
        
        该方法验证模型数据中是否包含至少2个不同的标签。
        这是进行有监督学习的基本要求。
        
        Args:
            model_data: 模型数据对象
            
        Returns:
            如果有足够的标签则返回 True，否则返回 False
        """
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
        """对单个消息进行预测。
        
        该方法使用训练好的模型对输入消息进行预测，返回预测结果。
        
        Args:
            message: 要预测的消息对象
            
        Returns:
            预测结果字典，如果模型未训练或数据为空则返回 None
        """
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # 从消息创建会话数据并转换为批次大小为1的数据
        model_data = self._create_model_data([message], training=False)
        if model_data.is_empty():
            return None
        return self.model.run_inference(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """预测提供消息的意图。
        
        该方法从模型预测输出中提取意图预测结果，包括最高置信度的标签和标签排名。
        
        Args:
            predict_out: 模型预测输出字典
            
        Returns:
            包含预测标签和标签排名的元组
        """
        label: Dict[Text, Any] = {"name": None, "confidence": 0.0}
        label_ranking: List[Dict[Text, Any]] = []

        if predict_out is None:
            return label, label_ranking

        # 获取意图相似度分数
        message_sim = predict_out["i_scores"]
        message_sim = message_sim.flatten()  # 将矩阵展平

        # 如果输入全为零，则不预测任何标签
        if message_sim.size == 0:
            return label, label_ranking

        # 对置信度进行排名
        ranking_length = self.component_config[RANKING_LENGTH]
        renormalize = (
            self.component_config[RENORMALIZE_CONFIDENCES]
            and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        )
        ranked_label_indices, message_sim = train_utils.rank_and_mask(
            message_sim, ranking_length=ranking_length, renormalize=renormalize
        )

        # 构建标签和排名
        casted_message_sim: List[float] = message_sim.tolist()  # 将numpy float转换为Python float
        top_label_idx = ranked_label_indices[0]
        label = {
            "name": self.index_label_id_mapping[top_label_idx],
            "confidence": casted_message_sim[top_label_idx],
        }

        # 构建标签排名
        ranking = [(idx, casted_message_sim[idx]) for idx in ranked_label_indices]
        label_ranking = [
            {"name": self.index_label_id_mapping[label_idx], "confidence": score}
            for label_idx, score in ranking
        ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        """预测消息中的实体。
        
        该方法从模型预测输出中提取实体预测结果，并将标签转换为实体格式。
        
        Args:
            predict_out: 模型预测输出字典
            message: 要预测的消息对象
            
        Returns:
            预测的实体列表
        """
        if predict_out is None:
            return []

        # 将实体标签转换为标签格式
        predicted_tags, confidence_values = train_utils.entity_label_to_tags(
            predict_out, self._entity_tag_specs, self.component_config[BILOU_FLAG]
        )

        # 将预测转换为实体格式
        entities = self.convert_predictions_into_entities(
            message.get(TEXT),
            message.get(TOKENS_NAMES[TEXT], []),
            predicted_tags,
            self.split_entities_config,
            confidence_values,
        )

        # 添加提取器名称
        entities = self.add_extractor_name(entities)
        # 将新预测的实体与现有实体合并
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
        """将模型持久化到指定目录。
        
        该方法将训练好的模型保存到模型存储中，包括模型权重、配置和数据示例。
        """
        if self.model is None:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__
            tf_model_file = model_path / f"{file_name}.tf_model"

            # 创建模型文件目录
            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            # 如果启用了检查点模型且存在临时检查点目录
            if self.component_config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # 保存一个空文件来标记该模型是通过检查点生成的
                checkpoint_marker = model_path / f"{file_name}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            # 保存模型
            self.model.save(str(tf_model_file))

            # 保存数据示例
            serialize_nested_feature_arrays(
                self._data_example,
                model_path / f"{file_name}.data_example.st",
                model_path / f"{file_name}.data_example_metadata.json",
            )
            # 保存标签数据
            serialize_nested_feature_arrays(
                dict(self._label_data.data) if self._label_data is not None else {},
                model_path / f"{file_name}.label_data.st",
                model_path / f"{file_name}.label_data_metadata.json",
            )

            # 保存稀疏特征大小
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.sparse_feature_sizes.json",
                self._sparse_feature_sizes,
            )
            
            # 保存索引到标签ID的映射
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.index_label_id_mapping.json",
                self.index_label_id_mapping,
            )

            # 保存实体标签规范
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
        """从存储中加载策略（完整文档字符串请参见父类）。
        
        Args:
            cls: 类类型
            config: 配置字典
            model_storage: 模型存储对象
            resource: 资源对象
            execution_context: 执行上下文
            **kwargs: 其他关键字参数
            
        Returns:
            加载的 DIETClassifier 实例
        """
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
        """从提供的目录加载训练好的模型。
        
        Args:
            cls: 类类型
            model_path: 模型路径
            config: 配置字典
            model_storage: 模型存储对象
            resource: 资源对象
            execution_context: 执行上下文
            
        Returns:
            加载的 DIETClassifier 实例
        """
        # 从文件加载模型数据
        (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            data_example,
            sparse_feature_sizes,
        ) = cls._load_from_files(model_path)

        # 更新配置类型
        config = train_utils.update_confidence_type(config)
        config = train_utils.update_similarity_type(config)

        # 加载模型
        model = cls._load_model(
            entity_tag_specs,
            label_data,
            config,
            data_example,
            model_path,
            finetune_mode=execution_context.is_finetuning,
        )

        # 创建并返回 DIETClassifier 实例
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
        """从文件加载模型数据。
        
        Args:
            cls: 类类型
            model_path: 模型路径
            
        Returns:
            包含模型数据的元组
        """
        file_name = cls.__name__

        # 加载数据示例
        data_example = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.data_example.st"),
            str(model_path / f"{file_name}.data_example_metadata.json"),
        )
        # 加载标签数据
        # 加载标签数据
        loaded_label_data = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.label_data.st"),
            str(model_path / f"{file_name}.label_data_metadata.json"),
        )
        label_data = RasaModelData(data=loaded_label_data)

        # 加载稀疏特征大小
        sparse_feature_sizes = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.sparse_feature_sizes.json"
        )
        
        # 加载索引到标签ID的映射
        index_label_id_mapping = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.index_label_id_mapping.json"
        )
        
        # 加载实体标签规范
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

        # 转换索引到标签ID的映射
        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        # 返回加载的模型数据
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
        """加载训练好的模型。
        
        Args:
            cls: 类类型
            entity_tag_specs: 实体标签规范列表
            label_data: 标签数据
            config: 配置字典
            data_example: 数据示例
            model_path: 模型路径
            finetune_mode: 是否为微调模式
            
        Returns:
            加载的 RasaModel 实例
        """
        file_name = cls.__name__
        tf_model_file = model_path / f"{file_name}.tf_model"

        # 确定标签键和子键
        label_key = LABEL_KEY if config[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if config[INTENT_CLASSIFICATION] else None

        # 创建模型数据示例
        model_data_example = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )

        # 加载模型类
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
        """加载模型类实例。
        
        Args:
            cls: 类类型
            tf_model_file: TensorFlow 模型文件路径
            model_data_example: 模型数据示例
            label_data: 标签数据
            entity_tag_specs: 实体标签规范列表
            config: 配置字典
            finetune_mode: 是否为微调模式
            
        Returns:
            加载的 RasaModel 实例
        """
        # 创建预测数据示例，只包含文本特征
        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        # 加载模型类实例
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
        """实例化模型类。
        
        Args:
            model_data: 模型数据对象
            
        Returns:
            实例化的 RasaModel 对象
        """
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
        """确保实体标签规范的顺序与 CRF 层顺序匹配。
        
        Args:
            entity_tag_specs: 实体标签规范列表
            
        Returns:
            有序的实体标签规范列表
        """
        if entity_tag_specs is None:
            return []

        # CRF 层的顺序
        crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        ordered_tag_spec = []

        # 按照 CRF 顺序重新排列标签规范
        for tag_name in crf_order:
            for tag_spec in entity_tag_specs:
                if tag_name == tag_spec.tag_name:
                    ordered_tag_spec.append(tag_spec)

        return ordered_tag_spec

    def _check_data(self) -> None:
        """检查数据签名是否有效。
        
        验证文本特征和标签特征是否存在，并检查共享隐藏层配置的一致性。
        
        Raises:
            InvalidConfigException: 当数据签名无效时
        """
        # 检查文本特征是否存在
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        
        # 如果启用意图分类，检查标签特征
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL not in self.data_signature:
                raise InvalidConfigException(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

            # 如果启用共享隐藏层，检查特征签名是否一致
            if self.config[SHARE_HIDDEN_LAYERS]:
                different_sentence_signatures = False
                different_sequence_signatures = False
                
                # 检查句子特征签名是否不同
                if (
                    SENTENCE in self.data_signature[TEXT]
                    and SENTENCE in self.data_signature[LABEL]
                ):
                    different_sentence_signatures = (
                        self.data_signature[TEXT][SENTENCE]
                        != self.data_signature[LABEL][SENTENCE]
                    )
                
                # 检查序列特征签名是否不同
                if (
                    SEQUENCE in self.data_signature[TEXT]
                    and SEQUENCE in self.data_signature[LABEL]
                ):
                    different_sequence_signatures = (
                        self.data_signature[TEXT][SEQUENCE]
                        != self.data_signature[LABEL][SEQUENCE]
                    )

                # 如果特征签名不同，抛出异常
                if different_sentence_signatures or different_sequence_signatures:
                    raise ValueError(
                        "If hidden layer weights are shared, data signatures "
                        "for text_features and label_features must coincide."
                    )

        # 检查实体识别配置
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
        """创建训练指标。
        
        该方法创建用于监控训练过程的损失和准确率指标。
        指标按创建顺序排列，损失指标在前，准确率指标在后。
        """
        # self.metrics 将按照创建顺序排列
        # 所以先创建损失指标，以便先输出损失
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")  # 掩码损失
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")  # 意图损失
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")  # 实体损失
        self.entity_group_loss = tf.keras.metrics.Mean(name="g_loss")  # 实体组损失
        self.entity_role_loss = tf.keras.metrics.Mean(name="r_loss")  # 实体角色损失
        
        # 然后创建准确率指标，以便后输出准确率
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")  # 掩码准确率
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")  # 意图准确率
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")  # 实体 F1 分数
        self.entity_group_f1 = tf.keras.metrics.Mean(name="g_f1")  # 实体组 F1 分数
        self.entity_role_f1 = tf.keras.metrics.Mean(name="r_f1")

    def _update_metrics_to_log(self) -> None:
        """更新要记录的指标列表。
        
        根据配置决定哪些指标需要记录到日志中。
        在调试模式下会记录更多详细信息。
        """
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        # 如果启用掩码语言模型，记录相关指标
        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")
        
        # 如果启用意图分类，记录相关指标
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log.append("i_acc")
            if debug_log_level:
                self.metrics_to_log.append("i_loss")
        
        # 如果启用实体识别，记录相关指标
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log.append(f"{name[0]}_f1")
                    if debug_log_level:
                        self.metrics_to_log.append(f"{name[0]}_loss")

        # 记录指标信息
        self._log_metric_info()

    def _log_metric_info(self) -> None:
        """记录指标信息到日志。
        
        该方法将指标名称映射到可读的格式，并记录到调试日志中。
        """
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
        """准备模型层。
        
        该方法为文本、标签和实体创建相应的神经网络层。
        包括序列层、变换器层、掩码语言模型层等。
        """
        # 对于用户文本，准备组合不同特征类型的层，
        # 使用变换器嵌入所有内容，并可选择进行掩码语言建模
        self.text_name = TEXT
        self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ] = rasa_layers.RasaSequenceLayer(
            self.text_name, self.data_signature[self.text_name], self.config
        )
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        # 意图标签的处理类似于用户文本，但没有变换器，
        # 没有掩码语言建模，并且没有应用 dropout
        # 只对整体标签嵌入应用 dropout，而不是对单个特征应用
        # 在所有标签特征组合后
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

            # 禁用对稀疏和密集标签特征应用的输入 dropout
            label_config = self.config.copy()
            label_config.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )

            # 创建特征组合层
            self._tf_layers[
                f"feature_combining_layer.{self.label_name}"
            ] = rasa_layers.RasaFeatureCombiningLayer(
                self.label_name, self.label_signature[self.label_name], label_config
            )

            # 准备前馈神经网络层
            self._prepare_ffnn_layer(
                self.label_name,
                self.config[HIDDEN_LAYERS_SIZES][self.label_name],
                self.config[DROP_RATE],
            )

            # 准备标签分类层
            self._prepare_label_classification_layers(predictor_attribute=TEXT)

        # 如果启用实体识别，准备实体识别层
        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_mask_lm_loss(self, name: Text) -> None:
        """准备掩码语言模型损失。
        
        Args:
            name: 层名称
        """
        # 用于嵌入掩码位置的预测标记
        self._prepare_embed_layers(f"{name}_lm_mask")

        # 用于嵌入被掩码的真实标记
        self._prepare_embed_layers(f"{name}_golden_token")

        # 掩码损失是额外的损失
        # 设置缩放为 False，这样它不会压倒其他损失
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _create_bow(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sequence_feature_lengths: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        """创建词袋表示。
        
        Args:
            sequence_features: 序列特征列表
            sentence_features: 句子特征列表
            sequence_feature_lengths: 序列特征长度
            name: 层名称
            
        Returns:
            词袋表示张量
        """
        # 通过特征组合层处理特征
        x, _ = self._tf_layers[f"feature_combining_layer.{name}"](
            (sequence_features, sentence_features, sequence_feature_lengths),
            training=self._training,
        )

        # 通过沿序列维度求和转换为词袋
        x = tf.reduce_sum(x, axis=1)

        # 通过前馈神经网络层处理
        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """创建所有标签的嵌入表示。
        
        Returns:
            包含标签ID和标签嵌入的元组
        """
        # 获取所有标签ID
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        # 获取序列特征长度
        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        # 创建词袋表示
        x = self._create_bow(
            self.tf_label_data[LABEL][SEQUENCE],
            self.tf_label_data[LABEL][SENTENCE],
            sequence_feature_lengths,
            self.label_name,
        )
        
        # 创建标签嵌入
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
        """计算掩码语言模型损失。
        
        Args:
            outputs: 模型输出张量
            inputs: 输入张量
            seq_ids: 序列ID张量
            mlm_mask_boolean: 掩码布尔张量
            name: 层名称
            
        Returns:
            掩码损失张量
        """
        # 确保掩码中至少有一个元素
        mlm_mask_boolean = tf.cond(
            tf.reduce_any(mlm_mask_boolean),
            lambda: mlm_mask_boolean,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(mlm_mask_boolean)),
        )

        mlm_mask_boolean = tf.squeeze(mlm_mask_boolean, -1)

        # 选择被掩码的元素，丢弃批次和序列维度
        # 有效地从形状 (batch_size, sequence_length, units) 切换到
        # (num_masked_elements, units)
        outputs = tf.boolean_mask(outputs, mlm_mask_boolean)
        inputs = tf.boolean_mask(inputs, mlm_mask_boolean)
        ids = tf.boolean_mask(seq_ids, mlm_mask_boolean)

        # 创建预测标记和真实标记的嵌入
        tokens_predicted_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        tokens_true_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        # 为了限制计算昂贵的损失计算，我们将 MLM 中的标签空间（即标记空间）
        # 限制为仅在此批次中被掩码的标记。因此减少了标记嵌入列表
        # (tokens_true_embed) 和减少的标签列表 (ids) 分别作为
        # all_labels_embed 和 all_labels 传递。将来，我们可以不那么严格，
        # 构建一个稍微更大的标签空间，也可以包括当前批次中未掩码的标记。
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
        """计算标签损失。
        
        Args:
            text_features: 文本特征张量
            label_features: 标签特征张量
            label_ids: 标签ID张量
            
        Returns:
            标签损失张量
        """
        # 创建所有标签的嵌入
        all_label_ids, all_labels_embed = self._create_all_labels()

        # 创建文本和标签的嵌入
        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        # 计算标签损失
        return self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """计算给定批次的损失。

        Args:
            batch_in: 批次数据

        Returns:
            给定批次的损失
        """
        # 将批次数据转换为模型数据格式
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        # 获取序列特征长度
        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )

        # 通过序列层处理文本特征
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

        # 句子级别特征的序列长度始终为1，但如果句子级别特征不存在，
        # 它们实际上可以为0
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        # 组合序列和句子特征长度
        combined_sequence_sentence_feature_lengths = (
            sequence_feature_lengths + sentence_feature_lengths
        )

        # 如果启用掩码语言模型且处于训练模式
        if self.config[MASKED_LM] and self._training:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, mlm_mask_boolean_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        # 如果启用意图分类
        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                combined_sequence_sentence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )
            losses.append(loss)

        # 如果启用实体识别
        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )

        # 返回所有损失的总和
        return tf.math.add_n(losses)

    def _batch_loss_intent(
        self,
        combined_sequence_sentence_feature_lengths_text: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        """计算意图分类的批次损失。
        
        Args:
            combined_sequence_sentence_feature_lengths_text: 组合的序列和句子特征长度
            text_transformed: 转换后的文本特征
            tf_batch_data: 批次数据
            
        Returns:
            意图分类损失
        """
        # 获取用于意图分类的句子特征向量
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        # 获取标签的序列特征长度
        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )

        # 获取标签ID和创建标签词袋表示
        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )
        
        # 计算标签损失和准确率
        loss, acc = self._calculate_label_loss(sentence_vector, label, label_ids)

        # 更新标签指标
        self._update_label_metrics(loss, acc)

        return loss

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:
        """更新标签指标。

        Args:
            loss: 损失值
            acc: 准确率值
        """
        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)

    def _batch_loss_entities(
        self,
        mask_combined_sequence_sentence: tf.Tensor,
        sequence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> List[tf.Tensor]:
        """计算实体识别的批次损失。
        
        Args:
            mask_combined_sequence_sentence: 组合的序列和句子掩码
            sequence_feature_lengths: 序列特征长度
            text_transformed: 转换后的文本特征
            tf_batch_data: 批次数据
            
        Returns:
            实体识别损失列表
        """
        losses = []

        entity_tags = None

        # 遍历所有实体标签规范
        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            # 获取标签ID
            tag_ids = tf_batch_data[ENTITIES][tag_spec.tag_name][0]
            # 为句子特征添加零（无实体）以匹配输入的形状
            tag_ids = tf.pad(tag_ids, [[0, 0], [0, 1], [0, 0]])

            # 计算实体损失
            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                tag_spec.tag_name,
                entity_tags,
            )

            # 如果是实体类型标签，将其用作角色和组CRF的额外输入
            if tag_spec.tag_name == ENTITY_ATTRIBUTE_TYPE:
                entity_tags = tf.one_hot(
                    tf.cast(tag_ids[:, :, 0], tf.int32), depth=tag_spec.num_tags
                )

            # 更新实体指标
            self._update_entity_metrics(loss, f1, tag_spec.tag_name)

            losses.append(loss)

        return losses

    def _update_entity_metrics(
        self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text
    ) -> None:
        """更新实体指标。
        
        Args:
            loss: 损失值
            f1: F1分数
            tag_name: 标签名称
        """
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
        """为预测准备模型。
        
        该方法在预测前准备模型，包括创建标签嵌入等。
        """
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, tf.Tensor]:
        """预测给定批次的输出。

        Args:
            batch_in: 批次数据

        Returns:
            要预测的输出
        """
        # 将批次数据转换为模型数据格式
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        # 获取序列和句子特征长度
        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        # 通过序列层处理文本特征
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
        
        # 创建预测结果字典
        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        # 如果启用意图分类，预测意图
        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(
                    sequence_feature_lengths + sentence_feature_lengths,
                    text_transformed,
                )
            )

        # 如果启用实体识别，预测实体
        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_feature_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
        self, sequence_feature_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:
        """预测实体的批次方法。
        
        Args:
            sequence_feature_lengths: 序列特征长度
            text_transformed: 转换后的文本特征
            
        Returns:
            实体预测结果字典
        """
        predictions: Dict[Text, tf.Tensor] = {}

        entity_tags = None

        # 遍历所有实体标签规范
        for tag_spec in self._entity_tag_specs:
            # 如果CRF层未训练，跳过
            if tag_spec.num_tags == 0:
                continue

            name = tag_spec.tag_name
            _input = text_transformed

            # 如果有实体标签，将其作为额外输入
            if entity_tags is not None:
                _tags = self._tf_layers[f"embed.{name}.tags"](entity_tags)
                _input = tf.concat([_input, _tags], axis=-1)

            # 计算logits
            _logits = self._tf_layers[f"embed.{name}.logits"](_input)
            
            # 通过CRF层预测
            pred_ids, confidences = self._tf_layers[f"crf.{name}"](
                _logits, sequence_feature_lengths
            )

            # 存储预测结果
            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

            # 如果是实体类型标签，将其用作角色和组CRF的额外输入
            if name == ENTITY_ATTRIBUTE_TYPE:
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
        self,
        combined_sequence_sentence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:
        """预测意图的批次方法。
        
        Args:
            combined_sequence_sentence_feature_lengths: 组合的序列和句子特征长度
            text_transformed: 转换后的文本特征
            
        Returns:
            意图预测结果字典
        """
        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # 获取用于意图分类的句子特征向量
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        # 计算相似度和置信度
        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        return {"i_scores": scores}
