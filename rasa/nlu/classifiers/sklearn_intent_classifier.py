# 导入未来版本的注解支持，用于类型提示
from __future__ import annotations

# 导入日志记录模块
import logging
# 导入类型检查模块
import typing
# 导入警告模块
import warnings
# 导入类型提示相关的类型
from typing import Any, Dict, List, Optional, Text, Tuple, Type

# 导入numpy数值计算库
import numpy as np

# 导入Rasa共享工具模块
import rasa.shared.utils.io
# 导入图组件基类
from rasa.engine.graph import GraphComponent, ExecutionContext
# 导入默认配方注册装饰器
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# 导入资源管理相关类
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
# 导入标签排序长度常量
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
# 导入意图分类器基类
from rasa.nlu.classifiers.classifier import IntentClassifier
# 导入密集特征提取器
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
# 导入文档URL常量
from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
# 导入Rasa异常类
from rasa.shared.exceptions import RasaException
# 导入文本常量
from rasa.shared.nlu.constants import TEXT
# 导入消息类
from rasa.shared.nlu.training_data.message import Message
# 导入训练数据类
from rasa.shared.nlu.training_data.training_data import TrainingData
# 导入特征提取器常量
from rasa.utils.tensorflow.constants import FEATURIZERS

# 创建日志记录器
logger = logging.getLogger(__name__)

# 类型检查时导入sklearn模块
if typing.TYPE_CHECKING:
    import sklearn


# 注册为默认配方中的意图分类器组件，标记为可训练
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class SklearnIntentClassifier(GraphComponent, IntentClassifier):
    """使用sklearn框架的意图分类器。"""

    @classmethod
    def required_components(cls) -> List[Type]:
        """在此组件之前应该包含在管道中的组件。"""
        return [DenseFeaturizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """组件的默认配置（完整文档字符串请参见父类）。"""
        return {
            # SVM的C参数 - 交叉验证将选择最佳值
            "C": [1, 2, 5, 10, 20, 100],
            # SVM的gamma参数
            "gamma": [0.1],
            # 用于SVM训练的内核 - 交叉验证将决定哪个性能最好
            "kernels": ["linear"],
            # 我们尝试找到在意图训练期间使用的良好交叉折数，这指定了最大折数
            "max_cross_validation_folds": 5,
            # 用于评估超参数的评分函数
            # 这可以是一个名称或函数（更多信息请参见GridSearchCV文档）
            "scoring_function": "f1_weighted",
            # 线程数
            "num_threads": 1,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        clf: Optional["sklearn.model_selection.GridSearchCV"] = None,
        le: Optional["sklearn.preprocessing.LabelEncoder"] = None,
    ) -> None:
        """使用sklearn框架构造新的意图分类器。"""
        # 导入sklearn的标签编码器
        from sklearn.preprocessing import LabelEncoder

        # 保存组件配置
        self.component_config = config
        # 保存模型存储对象
        self._model_storage = model_storage
        # 保存资源对象
        self._resource = resource

        # 如果提供了标签编码器则使用，否则创建新的
        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        # 保存分类器对象
        self.clf = clf

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SklearnIntentClassifier:
        """创建新的未训练组件（完整文档字符串请参见父类）。"""
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """此组件运行所需的额外python依赖。"""
        return ["sklearn"]

    def transform_labels_str2num(self, labels: List[Text]) -> np.ndarray:
        """将字符串标签列表转换为数字标签表示。

        :param labels: 要转换为数字表示的标签列表
        """
        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y: np.ndarray) -> np.ndarray:
        """将数字标签表示转换回字符串标签列表。

        :param y: 要转换回字符串表示的标签列表"""

        return self.le.inverse_transform(y)

    def train(self, training_data: TrainingData) -> Resource:
        """在数据集上训练意图分类器。"""
        # 获取线程数配置
        num_threads = self.component_config["num_threads"]

        # 从训练数据中提取所有意图标签
        labels = [e.get("intent") for e in training_data.intent_examples]

        # 检查是否有足够的意图类别进行训练
        if len(set(labels)) < 2:
            rasa.shared.utils.io.raise_warning(
                "无法训练意图分类器，因为意图数量不足。需要至少2个不同的意图。跳过意图分类器的训练。",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return self._resource

        # 将字符串标签转换为数字标签
        y = self.transform_labels_str2num(labels)
        # 筛选出具有所需特征的训练样本
        training_examples = [
            message
            for message in training_data.intent_examples
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]
        # 提取所有训练样本的句子特征并堆叠成矩阵
        X = np.stack(
            [self._get_sentence_features(example) for example in training_examples]
        )
        # 降维：将特征矩阵重塑为二维数组
        X = np.reshape(X, (len(X), -1))

        # 创建分类器
        self.clf = self._create_classifier(num_threads, y)

        # 捕获并忽略sklearn的警告
        with warnings.catch_warnings():
            # sklearn在意图样本较少时会抛出大量
            # "UndefinedMetricWarning: F-score is ill-defined"
            # 警告，这里需要忽略这些警告
            warnings.simplefilter("ignore")
            # 训练分类器
            self.clf.fit(X, y)

        # 持久化模型
        self.persist()
        return self._resource

    @staticmethod
    def _get_sentence_features(message: Message) -> np.ndarray:
        """从消息中获取句子特征。"""
        # 获取消息的密集特征
        _, sentence_features = message.get_dense_features(TEXT)
        if sentence_features is not None:
            # 返回第一个特征向量
            return sentence_features.features[0]

        # 如果没有句子特征则抛出异常
        raise ValueError(
            "没有句子特征。无法训练sklearn策略。"
        )

    def _num_cv_splits(self, y: np.ndarray) -> int:
        """计算交叉验证的折数。"""
        # 获取最大交叉验证折数配置
        folds = self.component_config["max_cross_validation_folds"]
        # 计算合适的折数：至少2折，最多为配置的最大值，且每折至少有5个样本
        return max(2, min(folds, np.min(np.bincount(y)) // 5))

    def _create_classifier(
        self, num_threads: int, y: np.ndarray
    ) -> "sklearn.model_selection.GridSearchCV":
        """创建网格搜索分类器。"""
        # 导入sklearn的网格搜索和SVM分类器
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        # 获取配置参数
        C = self.component_config["C"]
        kernels = self.component_config["kernels"]
        gamma = self.component_config["gamma"]
        # 修复字符串类型问题，因为sklearn期望字符串而不是basestr实例
        tuned_parameters = [
            {"C": C, "gamma": gamma, "kernel": [str(k) for k in kernels]}
        ]

        # 目标是每折有5个样本

        # 计算交叉验证折数
        cv_splits = self._num_cv_splits(y)

        # 创建并返回网格搜索分类器
        return GridSearchCV(
            SVC(C=1, probability=True, class_weight="balanced"),
            param_grid=tuned_parameters,
            n_jobs=num_threads,
            cv=cv_splits,
            scoring=self.component_config["scoring_function"],
            verbose=1,
        )

    def process(self, messages: List[Message]) -> List[Message]:
        """返回消息最可能的意图及其概率。"""
        for message in messages:
            # 检查分类器是否已训练且消息是否具有所需特征
            if self.clf is None or not message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            ):
                # 组件要么未训练，要么没有接收到足够的训练数据，或者输入没有所需特征
                intent = None
                intent_ranking = []
            else:
                # 获取消息的句子特征并重塑为单行矩阵
                X = self._get_sentence_features(message).reshape(1, -1)

                # 预测意图ID和概率
                intent_ids, probabilities = self.predict(X)
                # 将数字标签转换回字符串标签
                intents = self.transform_labels_num2str(np.ravel(intent_ids))
                # `predict`返回矩阵，因为它应该能够处理多个示例，因此我们需要展平
                probabilities = probabilities.flatten()

                # 如果有有效的意图和概率
                if intents.size > 0 and probabilities.size > 0:
                    # 创建意图排名列表，限制长度
                    ranking = list(zip(list(intents), list(probabilities)))[
                        :LABEL_RANKING_LENGTH
                    ]

                    # 设置最可能的意图
                    intent = {"name": intents[0], "confidence": probabilities[0]}

                    # 创建意图排名列表
                    intent_ranking = [
                        {"name": intent_name, "confidence": score}
                        for intent_name, score in ranking
                    ]
                else:
                    # 如果没有有效结果，设置默认值
                    intent = {"name": None, "confidence": 0.0}
                    intent_ranking = []

            # 将意图和排名设置到消息中
            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)

        return messages

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """给定输入文本的bow向量，预测意图标签。

        返回所有标签的概率。

        :param X: 输入文本的bow向量
        :return: 包含每个标签一个条目的概率向量。
        """
        # 检查分类器是否已初始化
        if self.clf is None:
            raise RasaException(
                "Sklearn意图分类器尚未初始化和训练。"
            )

        # 返回所有标签的概率
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """给定输入文本的bow向量，预测最可能的标签。

        仅返回最可能的标签。

        :param X: 输入文本的bow向量
        :return: 元组，第一个是最可能的标签，第二个是其概率。
        """
        # 获取所有标签的概率
        pred_result = self.predict_prob(X)
        # 对概率进行排序，获取排序后的索引
        # 按降序排列

        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self) -> None:
        """将模型持久化到指定目录。"""
        # 导入skops序列化库
        import skops.io as sio

        # 写入模型存储
        with self._model_storage.write_to(self._resource) as model_dir:
            # 获取类名作为文件名
            file_name = self.__class__.__name__
            # 设置分类器文件名
            classifier_file_name = model_dir / f"{file_name}_classifier.skops"
            # 设置编码器文件名
            encoder_file_name = model_dir / f"{file_name}_encoder.json"

            # 如果分类器和编码器都存在
            if self.clf and self.le:
                # 将self.le.classes_（字符串的numpy数组）转换为列表以便使用json dump
                rasa.shared.utils.io.dump_obj_as_json_to_file(
                    encoder_file_name, list(self.le.classes_)
                )
                # 保存最佳估计器
                sio.dump(self.clf.best_estimator_, classifier_file_name)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> SklearnIntentClassifier:
        """加载已训练的组件（完整文档字符串请参见父类）。"""
        # 导入sklearn的标签编码器和skops序列化库
        from sklearn.preprocessing import LabelEncoder
        import skops.io as sio

        try:
            # 从模型存储中读取
            with model_storage.read_from(resource) as model_dir:
                # 获取类名和分类器文件路径
                file_name = cls.__name__
                classifier_file = model_dir / f"{file_name}_classifier.skops"

                # 如果分类器文件存在
                if classifier_file.exists():
                    # 获取不可信类型
                    unknown_types = sio.get_untrusted_types(file=classifier_file)

                    # 如果有不可信类型则报错
                    if unknown_types:
                        logger.error(
                            f"加载 {classifier_file} 时发现不可信类型 ({unknown_types})！"
                        )
                        raise ValueError()
                    else:
                        # 加载分类器
                        classifier = sio.load(classifier_file, trusted=unknown_types)

                    # 加载编码器文件
                    encoder_file = model_dir / f"{file_name}_encoder.json"
                    classes = rasa.shared.utils.io.read_json_file(encoder_file)

                    # 创建新的编码器
                    encoder = LabelEncoder()
                    # 创建意图分类器实例
                    intent_classifier = cls(
                        config, model_storage, resource, classifier, encoder
                    )
                    # 将字符串列表（类标签）转换回字符串的numpy数组
                    intent_classifier.transform_labels_str2num(classes)
                    return intent_classifier
        except ValueError:
            logger.debug(
                f"从模型存储加载 '{cls.__name__}' 失败。资源 '{resource.name}' 不存在。"
            )
        # 如果加载失败，返回新的未训练实例
        return cls(config, model_storage, resource)
