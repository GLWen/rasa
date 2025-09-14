# 导入时间模块
import time
# 导入随机数模块
import random
# 导入TensorFlow深度学习框架
import tensorflow as tf
# 导入NumPy数值计算库
import numpy as np
# 导入日志记录模块
import logging
# 导入操作系统接口模块
import os
# 导入默认字典集合
from collections import defaultdict
# 导入类型提示相关类型
from typing import List, Text, Dict, Tuple, Union, Optional, Any, TYPE_CHECKING

# 导入Keras工具函数
from keras.utils import tf_utils
# 导入Keras模型基类
from keras import Model

# 导入Rasa共享常量
from rasa.shared.constants import DIAGNOSTIC_DATA
# 导入TensorFlow相关常量
from rasa.utils.tensorflow.constants import (
    LABEL,                    # 标签常量
    IDS,                      # ID常量
    INTENT_CLASSIFICATION,    # 意图分类常量
    SENTENCE,                 # 句子常量
    SEQUENCE_LENGTH,          # 序列长度常量
    RANDOM_SEED,              # 随机种子常量
    EMBEDDING_DIMENSION,      # 嵌入维度常量
    REGULARIZATION_CONSTANT,  # 正则化常数常量
    SIMILARITY_TYPE,          # 相似性类型常量
    CONNECTION_DENSITY,       # 连接密度常量
    NUM_NEG,                  # 负样本数量常量
    LOSS_TYPE,                # 损失类型常量
    MAX_POS_SIM,              # 最大正相似性常量
    MAX_NEG_SIM,              # 最大负相似性常量
    USE_MAX_NEG_SIM,          # 使用最大负相似性常量
    NEGATIVE_MARGIN_SCALE,    # 负边距缩放常量
    SCALE_LOSS,               # 损失缩放常量
    LEARNING_RATE,            # 学习率常量
    CONSTRAIN_SIMILARITIES,   # 约束相似性常量
    MODEL_CONFIDENCE,         # 模型置信度常量
    RUN_EAGERLY,              # 急切执行常量
)
# 导入模型数据相关类
from rasa.utils.tensorflow.model_data import (
    RasaModelData,            # Rasa模型数据类
    FeatureSignature,         # 特征签名类
    FeatureArray,             # 特征数组类
)
# 导入训练工具模块
import rasa.utils.train_utils
# 导入TensorFlow层模块
from rasa.utils.tensorflow import layers
# 导入Rasa自定义层模块
from rasa.utils.tensorflow import rasa_layers
# 导入数据生成器
from rasa.utils.tensorflow.data_generator import (
    RasaDataGenerator,        # Rasa数据生成器
    RasaBatchDataGenerator,   # Rasa批数据生成器
)
# 导入NLU常量
from rasa.shared.nlu.constants import TEXT
# 导入Rasa异常类
from rasa.shared.exceptions import RasaException
# 导入TensorFlow类型定义
from rasa.utils.tensorflow.types import BatchData, MaybeNestedBatchData

# 类型检查时导入TensorFlow通用函数类型
if TYPE_CHECKING:
    from tensorflow.python.types.core import GenericFunction

# 创建日志记录器
logger = logging.getLogger(__name__)

# 定义标签键常量
LABEL_KEY = LABEL
# 定义标签子键常量
LABEL_SUB_KEY = IDS


# 忽略方法重写警告
# noinspection PyMethodOverriding
class RasaModel(Model):
    """抽象自定义Keras模型。

     此模型重写了以下方法：
    - train_step: 训练步骤
    - test_step: 测试步骤
    - predict_step: 预测步骤
    - save: 保存方法
    - load: 加载方法
    不能直接用作tf.keras.Model。
    """

    # 训练状态标志
    _training: Optional[bool]

    def __init__(self, random_seed: Optional[int] = None, **kwargs: Any) -> None:
        """初始化RasaModel。

        Args:
            random_seed: 设置随机种子以获得可重现的结果
        """
        # 确保Keras释放之前训练模型的资源
        tf.keras.backend.clear_session()
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 创建总损失指标
        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        # 设置要记录的指标列表
        self.metrics_to_log = ["t_loss"]

        # 训练阶段应在构建图时定义
        self._training = None

        # 如果没有提供随机种子，使用当前时间戳
        if random_seed is None:
            random_seed = int(time.time())
        self.random_seed = random_seed
        # 设置随机种子
        self._set_random_seed()

        # TensorFlow预测步骤函数
        self._tf_predict_step: Optional["GenericFunction"] = None
        # 预测准备状态标志
        self.prepared_for_prediction = False

        # 创建检查点对象
        self._checkpoint = tf.train.Checkpoint(model=self)

    def _set_random_seed(self) -> None:
        """设置所有随机数生成器的种子以确保可重现性。"""
        # 设置Python标准库随机数种子
        random.seed(self.random_seed)
        # 设置NumPy随机数种子
        np.random.seed(self.random_seed)
        # 设置TensorFlow随机数种子
        tf.random.set_seed(self.random_seed)
        # 设置TensorFlow实验性NumPy随机数种子
        tf.experimental.numpy.random.seed(self.random_seed)
        # 设置Keras随机数种子
        tf.keras.utils.set_random_seed(self.random_seed)
        # 设置Python哈希种子的固定值
        os.environ["PYTHONHASHSEED"] = str(self.random_seed)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """计算给定批次的损失。

        Args:
            batch_in: 输入批次。

        Returns:
            给定批次的损失。
        """
        raise NotImplementedError

    def prepare_for_predict(self) -> None:
        """为预测准备TensorFlow图。

        此方法应包含必要的TensorFlow计算
        并设置`batch_predict`中使用的self变量。
        例如，预计算`self.all_labels_embed`。
        """
        pass

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """预测给定批次的输出。

        Args:
            batch_in: 输入批次。

        Returns:
            要预测的输出。
        """
        raise NotImplementedError

    def train_step(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, float]:
        """使用给定批次执行训练步骤。

        Args:
            batch_in: 批次输入。

        Returns:
            训练指标。
        """
        # 设置训练状态为True
        self._training = True

        # 分别计算监督损失和正则化损失
        with tf.GradientTape(persistent=True) as tape:
            # 计算预测损失
            prediction_loss = self.batch_loss(batch_in)
            # 计算正则化损失
            regularization_loss = tf.math.add_n(self.losses)
            # 计算总损失
            total_loss = prediction_loss + regularization_loss

        # 更新总损失状态
        self.total_loss.update_state(total_loss)

        # 计算来自监督信号的梯度
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)
        # 计算来自正则化的梯度
        regularization_gradients = tape.gradient(
            regularization_loss, self.trainable_variables
        )
        # 手动删除梯度带
        # 因为它使用了`persistent=True`选项创建
        del tape

        # 合并梯度
        gradients = []
        for pred_grad, reg_grad in zip(prediction_gradients, regularization_gradients):
            if pred_grad is not None and reg_grad is not None:
                # 对于没有预测梯度的变量移除正则化梯度
                gradients.append(
                    pred_grad
                    + tf.where(pred_grad > 0, reg_grad, tf.zeros_like(reg_grad))
                )
            else:
                gradients.append(pred_grad)

        # 应用梯度到可训练变量
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 重置训练状态
        self._training = None

        # 返回指标结果
        return self._get_metric_results()

    def test_step(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, float]:
        """使用给定批次测试模型。

        此方法在验证期间使用。

        Args:
            batch_in: 批次输入。

        Returns:
            测试指标。
        """
        # 设置训练状态为False
        self._training = False

        # 计算预测损失
        prediction_loss = self.batch_loss(batch_in)
        # 计算正则化损失
        regularization_loss = tf.math.add_n(self.losses)
        # 计算总损失
        total_loss = prediction_loss + regularization_loss
        # 更新总损失状态
        self.total_loss.update_state(total_loss)

        # 重置训练状态
        self._training = None

        # 返回指标结果
        return self._get_metric_results()

    def predict_step(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, tf.Tensor]:
        """预测给定批次的输出。

        Args:
            batch_in: 要预测的批次。

        Returns:
            预测输出。
        """
        # 设置训练状态为False
        self._training = False

        # 如果模型尚未为预测做好准备
        if not self.prepared_for_prediction:
            # 如果模型在没有加载的情况下用于预测，例如直接在训练后，
            # 我们需要为预测准备模型一次
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        # 返回批次预测结果
        return self.batch_predict(batch_in)

    @staticmethod
    def _dynamic_signature(
        batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> List[List[tf.TensorSpec]]:
        """动态生成TensorFlow函数签名。"""
        element_spec = []
        for tensor in batch_in:
            if len(tensor.shape) > 1:
                # 对于多维张量，除了最后一维外都设为None
                shape: List[Union[None, int]] = [None] * (len(tensor.shape) - 1)
                shape += [tensor.shape[-1]]
            else:
                # 对于一维张量，形状设为None
                shape = [None]
            element_spec.append(tf.TensorSpec(shape, tensor.dtype))
        # batch_in是张量列表，因此我们需要将element_spec包装到列表中
        return [element_spec]

    def _rasa_predict(
        self, batch_in: Tuple[np.ndarray, ...]
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """自定义预测方法，在第一次调用时构建TensorFlow图。

        Args:
            batch_in: 准备好的批次，准备输入到模型的`predict_step`方法。

        Return:
            预测输出，包括诊断数据。
        """
        # 设置训练状态为False
        self._training = False
        # 如果模型尚未为预测做好准备
        if not self.prepared_for_prediction:
            # 如果模型在没有加载的情况下用于预测，例如直接在训练后，
            # 我们需要为预测准备模型一次
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        # 如果使用急切执行模式
        if self._run_eagerly:
            # 一旦我们利用TF的分布式训练，这里就是
            # 调度函数将被强制执行并返回实际值的地方。
            outputs = tf_utils.sync_to_numpy_or_python_type(self.predict_step(batch_in))
            if DIAGNOSTIC_DATA in outputs:
                outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                    outputs[DIAGNOSTIC_DATA]
                )
            return outputs

        # 如果TensorFlow预测步骤函数尚未创建
        if self._tf_predict_step is None:
            self._tf_predict_step = tf.function(
                self.predict_step, input_signature=self._dynamic_signature(batch_in)
            )

        # 一旦我们利用TF的分布式训练，这里就是
        # 调度函数将被强制执行并返回实际值的地方。
        outputs = tf_utils.sync_to_numpy_or_python_type(self._tf_predict_step(batch_in))
        if DIAGNOSTIC_DATA in outputs:
            outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                outputs[DIAGNOSTIC_DATA]
            )
        return outputs

    def run_inference(
        self,
        model_data: RasaModelData,
        batch_size: Union[int, List[int]] = 1,
        output_keys_expected: Optional[List[Text]] = None,
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """通过模型实现批量推理。

        Args:
            model_data: 要输入到模型的数据。
            batch_size: 生成器应创建的批次大小。
            output_keys_expected: 输出中期望的键。
                在将其与所有批次的输出合并之前，
                输出应被过滤为仅包含这些键。

        Returns:
            与输入对应的模型输出。
        """
        # 初始化输出字典
        outputs: Dict[Text, Union[np.ndarray, Dict[Text, Any]]] = {}
        # 创建数据生成器
        (data_generator, _) = rasa.utils.train_utils.create_data_generators(
            model_data=model_data, batch_sizes=batch_size, epochs=1, shuffle=False
        )
        # 创建数据迭代器
        data_iterator = iter(data_generator)
        while True:
            try:
                # data_generator是一个包含2个元素的元组 - 输入和输出。
                # 我们只需要输入，因为输出总是None且不被我们的TF图消耗。
                batch_in = next(data_iterator)[0]
                # 对批次进行预测
                batch_out: Dict[
                    Text, Union[np.ndarray, Dict[Text, Any]]
                ] = self._rasa_predict(batch_in)
                # 如果指定了期望的输出键，则过滤输出
                if output_keys_expected:
                    batch_out = {
                        key: output
                        for key, output in batch_out.items()
                        if key in output_keys_expected
                    }
                # 合并批次输出
                outputs = self._merge_batch_outputs(outputs, batch_out)
            except StopIteration:
                # 生成器用完批次，是时候完成推理
                break
        return outputs

    @staticmethod
    def _merge_batch_outputs(
        all_outputs: Dict[Text, Union[np.ndarray, Dict[Text, Any]]],
        batch_output: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """将批次的输出合并到所有批次的输出中。

        函数假设批次输出的模式保持不变，
        即键及其值类型不会从一个批次的输出
        到另一个批次的输出发生变化。

        Args:
            all_outputs: 所有先前批次的现有输出。
            batch_output: 一个批次的输出。

        Returns:
            合并的输出，当前批次的输出堆叠在
            所有先前批次的输出下方。
        """
        # 如果没有现有输出，直接返回批次输出
        if not all_outputs:
            return batch_output
        # 遍历批次输出的每个键值对
        for key, val in batch_output.items():
            if isinstance(val, np.ndarray):
                # 如果是numpy数组，沿第0轴连接
                all_outputs[key] = np.concatenate(
                    [all_outputs[key], batch_output[key]], axis=0
                )

            elif isinstance(val, dict):
                # 如果是字典，递归合并内部字典
                all_outputs[key] = RasaModel._merge_batch_outputs(all_outputs[key], val)

        return all_outputs

    @staticmethod
    def _empty_lists_to_none_in_dict(input_dict: Dict[Text, Any]) -> Dict[Text, Any]:
        """递归地将字典中的空列表或numpy数组替换为None。"""

        def _recurse(
            x: Union[Dict[Text, Any], List[Any], np.ndarray]
        ) -> Optional[Union[Dict[Text, Any], List[Any], np.ndarray]]:
            # 如果是字典，递归处理每个值
            if isinstance(x, dict):
                return {k: _recurse(v) for k, v in x.items()}
            # 如果是空列表或空numpy数组，返回None
            elif (isinstance(x, list) or isinstance(x, np.ndarray)) and np.size(x) == 0:
                return None
            # 否则返回原值
            return x

        return {k: _recurse(v) for k, v in input_dict.items()}

    def _get_metric_results(self, prefix: Optional[Text] = "") -> Dict[Text, float]:
        """获取指标结果。"""
        return {
            f"{prefix}{metric.name}": metric.result()
            for metric in self.metrics
            if metric.name in self.metrics_to_log
        }

    def save(self, model_file_name: Text, overwrite: bool = True) -> None:
        """将模型保存到给定文件。

        Args:
            model_file_name: 保存模型的文件名。
            overwrite: 如果为'True'，同名的现有模型将被覆盖。
        """
        self.save_weights(model_file_name, overwrite=overwrite, save_format="tf")

    @classmethod
    def load(
        cls,
        model_file_name: Text,
        model_data_example: RasaModelData,
        predict_data_example: Optional[RasaModelData] = None,
        finetune_mode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> "RasaModel":
        """从给定权重加载模型。

        Args:
            model_file_name: 包含模型权重的文件路径。
            model_data_example: 用于构建模型架构的示例数据点。
            predict_data_example: 用于在推理期间加速预测的示例数据点。
            finetune_mode: 指示是否加载模型以进行进一步的微调。
            *args: 任何其他非关键字参数。
            **kwargs: 任何其他关键字参数。

        Returns:
            权重适当设置的已加载模型。
        """
        logger.debug(
            f"从 {model_file_name} 加载模型 "
            f"finetune_mode={finetune_mode}..."
        )
        # 创建空模型
        model = cls(*args, **kwargs)
        # 获取学习率配置
        learning_rate = kwargs.get("config", {}).get(LEARNING_RATE, 0.001)
        # 获取急切执行配置
        run_eagerly = kwargs.get("config", {}).get(RUN_EAGERLY)

        # 需要在1个示例上训练以构建正确大小的权重
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), run_eagerly=run_eagerly
        )
        # 创建数据生成器
        data_generator = RasaBatchDataGenerator(model_data_example, batch_size=1)
        # 训练模型以构建权重
        model.fit(data_generator, verbose=False)
        # 加载训练好的权重
        model.load_weights(model_file_name)

        # 在一个数据示例上进行预测以在推理期间加速预测
        # 第一次预测总是需要更长时间来跟踪tf函数
        if not finetune_mode and predict_data_example:
            model.run_inference(predict_data_example)

        logger.debug("完成模型加载。")
        return model

    @staticmethod
    def batch_to_model_data_format(
        batch: MaybeNestedBatchData,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
    ) -> Dict[Text, Dict[Text, List[tf.Tensor]]]:
        """将输入批次张量转换为批次数据格式。

        批次包含任意数量的批次数据。顺序等于
        会话数据中的键值对。由于稀疏数据之前被转换为（索引，
        数据，形状），此方法将它们转换为稀疏张量。密集
        数据保持不变。
        """
        # 在训练期间，批次是输入和目标数据的元组
        # 由于我们的目标数据在输入数据内部，我们只对
        # 输入数据感兴趣
        unpacked_batch = batch[0] if isinstance(batch[0], Tuple) else batch

        # 创建批次数据字典
        batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

        idx = 0
        # 遍历数据签名
        for key, values in data_signature.items():
            for sub_key, signature in values.items():
                for is_sparse, feature_dimension, number_of_dimensions in signature:
                    # 我们之前将所有4D特征转换为3D特征
                    number_of_dimensions = (
                        number_of_dimensions if number_of_dimensions != 4 else 3
                    )
                    if is_sparse:
                        # 转换稀疏特征
                        tensor, idx = RasaModel._convert_sparse_features(
                            unpacked_batch, feature_dimension, idx, number_of_dimensions
                        )
                    else:
                        # 转换密集特征
                        tensor, idx = RasaModel._convert_dense_features(
                            unpacked_batch, feature_dimension, idx, number_of_dimensions
                        )
                    batch_data[key][sub_key].append(tensor)

        return batch_data

    @staticmethod
    def _convert_dense_features(
        batch: BatchData,
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.Tensor, int]:
        """转换密集特征为TensorFlow张量。"""
        batch_at_idx = batch[idx]
        if isinstance(batch_at_idx, tf.Tensor):
            # 显式地用已知的静态值替换形状中的最后一维
            if number_of_dimensions > 1 and (
                batch_at_idx.shape is None or batch_at_idx.shape[-1] is None
            ):
                shape: List[Optional[int]] = [None] * (number_of_dimensions - 1)
                shape.append(feature_dimension)
                batch_at_idx.set_shape(shape)

            return batch_at_idx, idx + 1

        # 转换为Tensor
        return (
            tf.constant(batch[idx], dtype=tf.float32, shape=batch[idx].shape),
            idx + 1,
        )

    @staticmethod
    def _convert_sparse_features(
        batch: BatchData,
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.SparseTensor, int]:
        """转换稀疏特征为TensorFlow稀疏张量。"""
        # 显式地用已知的静态值替换形状中的最后一维
        shape = [batch[idx + 2][i] for i in range(number_of_dimensions - 1)] + [
            feature_dimension
        ]
        return tf.SparseTensor(batch[idx], batch[idx + 1], shape), idx + 3

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        training: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """在新输入上调用模型。

        Arguments:
            inputs: 张量或张量列表。
            training: 布尔值或布尔标量张量，指示是否在
              训练模式或推理模式下运行`Network`。
            mask: 掩码或掩码列表。掩码可以是
                张量或None（无掩码）。

        Returns:
            如果有单个输出则为张量，或
            如果有多个输出则为张量列表。
        """
        # 此方法需要实现，否则父类会抛出
        # NotImplementedError('当子类化`Model`类时，你应该
        #   实现一个`call`方法。')
        pass


# 忽略方法重写警告
# noinspection PyMethodOverriding
class TransformerRasaModel(RasaModel):
    """基于Transformer的Rasa模型。"""
    def __init__(
        self,
        name: Text,
        config: Dict[Text, Any],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
    ) -> None:
        # 调用父类初始化方法
        super().__init__(name=name, random_seed=config[RANDOM_SEED])

        # 保存配置
        self.config = config
        # 保存数据签名
        self.data_signature = data_signature
        # 获取标签签名
        self.label_signature = label_data.get_signature()
        # 检查数据
        self._check_data()

        # 准备标签批次
        label_batch = RasaDataGenerator.prepare_batch(label_data.data)
        # 转换为模型数据格式
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, self.label_signature
        )

        # 设置TensorFlow层
        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}

    def adjust_for_incremental_training(
        self,
        data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """调整模型以进行增量训练。

        首先我们应该检查是否有任何稀疏特征大小减少了
        如果发生这种情况则抛出异常。
        如果它们都没有减少并且其中任何一个增加了，那么
        函数更新`DenseForSparse`层，编译模型，在其上拟合样本
        数据以激活调整后的层并更新数据签名。

        新旧稀疏特征大小可能如下所示：
        {TEXT: {FEATURE_TYPE_SEQUENCE: [4, 24, 128], FEATURE_TYPE_SENTENCE: [4, 128]}}

        Args:
            data_example: 与ML组件一起存储的数据示例。
            new_sparse_feature_sizes: 当前稀疏特征的大小。
            old_sparse_feature_sizes: 模型之前训练的稀疏特征大小。
        """
        # 检查稀疏特征大小是否减少
        self._check_if_sparse_feature_sizes_decreased(
            new_sparse_feature_sizes=new_sparse_feature_sizes,
            old_sparse_feature_sizes=old_sparse_feature_sizes,
        )
        # 如果稀疏特征大小增加了
        if self._sparse_feature_sizes_have_increased(
            new_sparse_feature_sizes=new_sparse_feature_sizes,
            old_sparse_feature_sizes=old_sparse_feature_sizes,
        ):
            # 更新稀疏层的密集层
            self._update_dense_for_sparse_layers(
                new_sparse_feature_sizes, old_sparse_feature_sizes
            )
            # 编译并拟合数据
            self._compile_and_fit(data_example)

    @staticmethod
    def _check_if_sparse_feature_sizes_decreased(
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """检查稀疏特征大小在微调期间是否减少。

        稀疏特征大小在更改训练数据后可能会减少。
        例如，使用`LexicalSyntacticFeaturizer`时可能发生这种情况。
        我们不支持这种行为，如果发生这种情况我们会抛出异常。

        Args:
            new_sparse_feature_sizes: 当前稀疏特征的大小。
            old_sparse_feature_sizes: 模型之前训练的稀疏特征大小。

        Raises:
            RasaException: 当任何稀疏特征大小从上次运行训练时减少。
        """
        # 遍历新稀疏特征大小
        for attribute, new_feature_sizes in new_sparse_feature_sizes.items():
            old_feature_sizes = old_sparse_feature_sizes[attribute]
            for feature_type, new_sizes in new_feature_sizes.items():
                old_sizes = old_feature_sizes[feature_type]
                for new_size, old_size in zip(new_sizes, old_sizes):
                    if new_size < old_size:
                        raise RasaException(
                            "稀疏特征大小从上次运行训练时减少了。训练数据以某种方式更改，"
                            "导致某些特征不再存在于数据中。如果您在管道中有"
                            "`LexicalSyntacticFeaturizer`，可能会发生这种情况。"
                            "管道在此设置中无法支持增量训练。我们建议您从头重新训练模型。"
                        )

    @staticmethod
    def _sparse_feature_sizes_have_increased(
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> bool:
        """检查稀疏特征大小在微调期间是否增加。

        如果在更改训练数据后有任何稀疏特征大小增加，我们需要查找
        相应的`DenseForSparse`层并调整它。另一方面，如果它们都没有增加，
        我们不需要更改任何东西。此函数帮助我们做出决定。

        注意，函数假设没有任何稀疏特征大小减少。换句话说，
        它应该获得有效参数才能正常工作。

        Args:
            new_sparse_feature_sizes: 当前稀疏特征的大小。
            old_sparse_feature_sizes: 模型之前训练的稀疏特征大小。

        Returns:
            如果任何稀疏特征大小增加则返回`True`，否则返回`False`。
        """
        # 遍历新稀疏特征大小
        for attribute, new_feature_sizes in new_sparse_feature_sizes.items():
            old_feature_sizes = old_sparse_feature_sizes[attribute]
            for feature_type, new_sizes in new_feature_sizes.items():
                old_sizes = old_feature_sizes[feature_type]
                # 如果新大小总和大于旧大小总和
                if sum(new_sizes) > sum(old_sizes):
                    return True
        return False

    def _update_dense_for_sparse_layers(
        self,
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """更新`DenseForSparse`层。

        通过比较当前稀疏特征大小与旧大小来更新`DenseForSparse`层的大小。
        这必须在微调开始之前完成，以考虑可能因添加新数据而发生的
        稀疏特征大小的任何变化。

        Args:
            new_sparse_feature_sizes: 当前稀疏特征的大小。
            old_sparse_feature_sizes: 模型之前训练的稀疏特征大小。
        """
        # 遍历所有TensorFlow层
        for name, layer in self._tf_layers.items():
            # `if`条件是必要的，因为只有`RasaCustomLayer`
            # 默认可以调整稀疏层以进行增量训练。
            if isinstance(layer, rasa_layers.RasaCustomLayer):
                layer.adjust_sparse_layers_for_incremental_training(
                    new_sparse_feature_sizes,
                    old_sparse_feature_sizes,
                    self.config[REGULARIZATION_CONSTANT],
                )

    def _compile_and_fit(
        self, data_example: Dict[Text, Dict[Text, List[FeatureArray]]]
    ) -> None:
        """编译修改后的模型并在其上拟合样本数据。

        Args:
            data_example: 与ML组件一起存储的数据示例。
        """
        # 编译模型
        self.compile(
            optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE]),
            run_eagerly=self.config[RUN_EAGERLY],
        )
        # 设置标签键
        label_key = LABEL_KEY if self.config[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if self.config[INTENT_CLASSIFICATION] else None

        # 创建模型数据
        model_data = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )
        # 更新数据签名
        self._update_data_signatures(model_data)
        # 创建数据生成器
        data_generator = RasaBatchDataGenerator(model_data, batch_size=1)
        # 拟合数据
        self.fit(data_generator, verbose=False)

    def _update_data_signatures(self, model_data: RasaModelData) -> None:
        """更新数据签名。"""
        # 获取数据签名
        self.data_signature = model_data.get_signature()
        # 创建预测数据签名，只包含文本特征
        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in self.data_signature.items()
            if TEXT in feature_name
        }

    def _check_data(self) -> None:
        """检查数据。"""
        raise NotImplementedError

    def _prepare_layers(self) -> None:
        """准备层。"""
        raise NotImplementedError

    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """为最终标签预测步骤准备层和损失。"""
        # 准备预测器属性的嵌入层
        self._prepare_embed_layers(predictor_attribute)
        # 准备标签的嵌入层
        self._prepare_embed_layers(LABEL)
        # 准备点积损失
        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_embed_layers(self, name: Text, prefix: Text = "embed") -> None:
        """准备嵌入层。"""
        self._tf_layers[f"{prefix}.{name}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION], self.config[REGULARIZATION_CONSTANT], name
        )

    def _prepare_ffnn_layer(
        self,
        name: Text,
        layer_sizes: List[int],
        drop_rate: float,
        prefix: Text = "ffnn",
    ) -> None:
        """准备前馈神经网络层。"""
        self._tf_layers[f"{prefix}.{name}"] = layers.Ffnn(
            layer_sizes,
            drop_rate,
            self.config[REGULARIZATION_CONSTANT],
            self.config[CONNECTION_DENSITY],
            layer_name_suffix=name,
        )

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss"
    ) -> None:
        """准备点积损失层。"""
        self._tf_layers[f"{prefix}.{name}"] = self.dot_product_loss_layer(
            self.config[NUM_NEG],
            loss_type=self.config[LOSS_TYPE],
            mu_pos=self.config[MAX_POS_SIM],
            mu_neg=self.config[MAX_NEG_SIM],
            use_max_sim_neg=self.config[USE_MAX_NEG_SIM],
            neg_lambda=self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss=scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
            constrain_similarities=self.config[CONSTRAIN_SIMILARITIES],
            model_confidence=self.config[MODEL_CONFIDENCE],
        )

    @property
    def dot_product_loss_layer(self) -> tf.keras.layers.Layer:
        """返回要使用的点积损失层。

        Returns:
            由`_prepare_dot_product_loss`使用的损失层。
        """
        return layers.SingleLabelDotProductLoss

    def _prepare_entity_recognition_layers(self) -> None:
        """准备实体识别层。"""
        # 遍历实体标签规范
        for tag_spec in self._entity_tag_specs:
            name = tag_spec.tag_name
            num_tags = tag_spec.num_tags
            # 创建logits嵌入层
            self._tf_layers[f"embed.{name}.logits"] = layers.Embed(
                num_tags, self.config[REGULARIZATION_CONSTANT], f"logits.{name}"
            )
            # 创建CRF层
            self._tf_layers[f"crf.{name}"] = layers.CRF(
                num_tags, self.config[REGULARIZATION_CONSTANT], self.config[SCALE_LOSS]
            )
            # 创建标签嵌入层
            self._tf_layers[f"embed.{name}.tags"] = layers.Embed(
                self.config[EMBEDDING_DIMENSION],
                self.config[REGULARIZATION_CONSTANT],
                f"tags.{name}",
            )

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        """获取序列的最后一个标记。"""
        # 计算最后一个序列索引
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        # 创建批次索引
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        # 堆叠索引
        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        # 根据索引获取值
        return tf.gather_nd(x, indices)

    def _get_mask_for(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        key: Text,
        sub_key: Text,
    ) -> Optional[tf.Tensor]:
        """获取指定键和子键的掩码。"""
        # 检查键和子键是否存在
        if key not in tf_batch_data or sub_key not in tf_batch_data[key]:
            return None

        # 获取序列长度并转换为int32
        sequence_lengths = tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32)
        # 计算掩码
        return rasa_layers.compute_mask(sequence_lengths)

    def _get_sequence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text
    ) -> tf.Tensor:
        """获取每个输入示例的真实标记的序列长度。

        示例的真实标记数量与
        该输入示例的序列级（标记级）特征的序列长度相同。
        """
        # 如果存在序列长度信息
        if key in tf_batch_data and SEQUENCE_LENGTH in tf_batch_data[key]:
            return tf.cast(tf_batch_data[key][SEQUENCE_LENGTH][0], dtype=tf.int32)

        # 否则返回零张量
        batch_dim = self._get_batch_dim(tf_batch_data[key])
        return tf.zeros([batch_dim], dtype=tf.int32)

    def _get_sentence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text
    ) -> tf.Tensor:
        """获取每个输入示例的句子级特征的序列长度。

        这是必需的，因为我们将句子级特征视为标记级特征
        每个输入示例有1个标记。因此，如果存在句子级特征，
        此函数返回的序列长度都是1，否则为0。
        """
        # 获取批次维度
        batch_dim = self._get_batch_dim(tf_batch_data[key])

        # 如果存在句子级特征
        if key in tf_batch_data and SENTENCE in tf_batch_data[key]:
            return tf.ones([batch_dim], dtype=tf.int32)

        # 否则返回零张量
        return tf.zeros([batch_dim], dtype=tf.int32)

    @staticmethod
    def _get_batch_dim(attribute_data: Dict[Text, List[tf.Tensor]]) -> int:
        """获取批次维度。"""
        # attribute_data字典中的所有值都应该是张量列表，
        # 每个张量的形状为(batch_dim, ...)。所以我们取第一个非空列表
        # 并从其第一个张量推断批次大小。
        for key, data in attribute_data.items():
            if data:
                return tf.shape(data[0])[0]

        return 0

    def _calculate_entity_loss(
        self,
        inputs: tf.Tensor,
        tag_ids: tf.Tensor,
        mask: tf.Tensor,
        sequence_lengths: tf.Tensor,
        tag_name: Text,
        entity_tags: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        tag_ids = tf.cast(tag_ids[:, :, 0], tf.int32)

        if entity_tags is not None:
            _tags = self._tf_layers[f"embed.{tag_name}.tags"](entity_tags)
            inputs = tf.concat([inputs, _tags], axis=-1)

        logits = self._tf_layers[f"embed.{tag_name}.logits"](inputs)

        # should call first to build weights
        pred_ids, _ = self._tf_layers[f"crf.{tag_name}"](logits, sequence_lengths)
        loss = self._tf_layers[f"crf.{tag_name}"].loss(
            logits, tag_ids, sequence_lengths
        )
        f1 = self._tf_layers[f"crf.{tag_name}"].f1_score(tag_ids, pred_ids, mask)

        return loss, f1, logits

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError
