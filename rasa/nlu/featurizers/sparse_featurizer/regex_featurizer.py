# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入标准库模块
import logging  # 日志记录模块，用于记录程序运行状态和调试信息
import re  # 正则表达式模块，用于模式匹配和文本处理
from typing import Any, Dict, List, Optional, Text, Tuple, Type  # 类型注解模块，提供类型提示功能

# 导入科学计算库
import numpy as np  # 数值计算库，提供多维数组和数学运算功能
import scipy.sparse  # 稀疏矩阵库，用于高效存储和处理稀疏数据

# 导入 Rasa 核心模块
from rasa.nlu.tokenizers.tokenizer import Tokenizer  # 标记化器，用于将文本分割为标记
import rasa.shared.utils.io  # 共享工具模块，提供文件读写和序列化功能
import rasa.utils.io  # 工具模块，提供额外的 I/O 功能
import rasa.nlu.utils.pattern_utils as pattern_utils  # 模式工具模块，用于提取和处理正则表达式模式
from rasa.engine.graph import ExecutionContext, GraphComponent  # 图组件和执行上下文，用于组件管理和执行控制
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.nlu.constants import TOKENS_NAMES  # 标记名称常量，定义各种标记的键名
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer  # 稀疏特征化器基类，提供稀疏特征提取的通用接口
from rasa.shared.nlu.constants import TEXT, RESPONSE, ACTION_TEXT  # NLU 常量，定义消息属性的键名
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True  # 注册为消息特征化器组件类型，标记为可训练组件
)
class RegexFeaturizer(SparseFeaturizer, GraphComponent):
    """基于正则表达式添加消息特征的稀疏特征化器。
    
    该类继承自 SparseFeaturizer 和 GraphComponent，实现了基于正则表达式模式的稀疏特征提取。
    它可以从训练数据中自动提取正则表达式模式，或者使用预定义的模式来生成特征。
    支持查找表和正则表达式两种模式，可以处理大小写敏感和词边界匹配。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """获取此组件运行前必须包含在管道中的组件类型。
        
        该方法定义了组件的依赖关系，确保在特征提取之前文本已经被正确标记化。
        
        Returns:
            必需的组件类型列表，包含标记化器组件
        """
        return [Tokenizer]  # 需要标记化器组件，用于将文本分割为标记

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置参数。
        
        该方法定义了 RegexFeaturizer 的所有可配置参数及其默认值。
        配置参数控制正则表达式匹配的行为和特征生成的方式。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        return {
            **SparseFeaturizer.get_default_config(),  # 继承稀疏特征化器基类的默认配置
            # 文本处理配置
            "case_sensitive": True,  # 是否区分大小写，True表示区分大小写进行模式匹配
            
            # 特征生成配置
            "use_lookup_tables": True,  # 是否使用查找表生成特征，从训练数据中提取词汇模式
            "use_regexes": True,  # 是否使用正则表达式生成特征，从训练数据中提取正则模式
            "use_word_boundaries": True,  # 是否使用词边界匹配查找表，确保完整词匹配
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
    ) -> None:
        """使用正则表达式构造新的特征提取器。

        该构造函数初始化 RegexFeaturizer 实例，设置配置参数，
        加载已知的正则表达式模式，并准备特征提取功能。

        Args:
            config: 组件配置字典
            model_storage: 模型存储接口，图组件用于持久化和加载自身
            resource: 资源定位器，用于从 model_storage 中持久化和加载组件
            execution_context: 当前图运行的信息，包含执行上下文
            known_patterns: 组件应预加载的正则表达式模式列表
        """
        super().__init__(execution_context.node_name, config)  # 调用父类构造函数

        # 存储模型存储和资源管理对象
        self._model_storage = model_storage
        self._resource = resource

        # 初始化已知模式列表和配置参数
        self.known_patterns = known_patterns if known_patterns else []  # 已知的正则表达式模式列表
        self.case_sensitive = config["case_sensitive"]  # 是否区分大小写
        self.finetune_mode = execution_context.is_finetuning  # 是否为微调模式

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexFeaturizer:
        """创建新的未训练组件实例。
        
        这是一个类方法，用于创建新的 RegexFeaturizer 实例。
        通常在训练开始时调用，创建一个全新的、未训练的组件。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            
        Returns:
            新的未训练的 RegexFeaturizer 实例
        """
        return cls(config, model_storage, resource, execution_context)

    def _merge_new_patterns(self, new_patterns: List[Dict[Text, Text]]) -> None:
        """将新提取的模式与已知模式合并。

        该方法用于增量训练场景，将训练数据中提取的新模式与现有的已知模式合并。
        新模式总是添加到现有模式的末尾，不会打乱现有模式的顺序。
        如果模式名称已存在，则更新该模式的表达式。

        Args:
            new_patterns: 从训练数据中提取的模式列表，将与已知模式合并
        """
        # 创建模式名称到索引的映射，用于快速查找现有模式
        pattern_name_index_map = {
            pattern["name"]: index for index, pattern in enumerate(self.known_patterns)
        }
        
        for extra_pattern in new_patterns:  # 遍历每个新提取的模式
            new_pattern_name = extra_pattern["name"]  # 获取新模式名称

            # 某些模式可能只是添加了新的示例，这些不算作额外的模式
            if new_pattern_name in pattern_name_index_map:  # 如果模式名称已存在
                # 更新现有模式的表达式
                self.known_patterns[pattern_name_index_map[new_pattern_name]][
                    "pattern"
                ] = extra_pattern["pattern"]
            else:  # 如果模式名称不存在
                # 添加新模式到列表末尾
                self.known_patterns.append(extra_pattern)

    def train(self, training_data: TrainingData) -> Resource:
        """使用从训练数据中提取的所有模式训练组件。
        
        该方法从训练数据中提取正则表达式模式和查找表模式，
        根据配置决定是否进行增量训练，并持久化训练结果。
        
        Args:
            training_data: 训练数据，包含所有训练样本
            
        Returns:
            资源对象，用于模型持久化
        """
        # 从训练数据中提取模式（正则表达式和查找表）
        patterns_from_data = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self._config["use_lookup_tables"],  # 是否使用查找表
            use_regexes=self._config["use_regexes"],  # 是否使用正则表达式
            use_word_boundaries=self._config["use_word_boundaries"],  # 是否使用词边界
        )
        
        if self.finetune_mode:  # 如果是微调模式
            # 将数据中提取的模式与已知模式合并
            self._merge_new_patterns(patterns_from_data)
        else:  # 如果是全新训练模式
            # 直接使用从数据中提取的模式
            self.known_patterns = patterns_from_data

        self._persist()  # 持久化训练结果
        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """处理训练样本并生成特征。
        
        该方法遍历所有训练样本，为每个样本的文本、响应和动作文本属性生成正则表达式特征。
        
        Args:
            training_data: 训练数据
            
        Returns:
            处理后的训练数据
        """
        for example in training_data.training_examples:  # 遍历每个训练样本
            for attribute in [TEXT, RESPONSE, ACTION_TEXT]:  # 处理文本、响应和动作文本属性
                self._text_features_with_regex(example, attribute)  # 生成正则表达式特征

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """对给定的消息列表进行特征化处理。

        该方法对输入的消息列表进行就地修改，为每个消息生成正则表达式特征。
        主要用于预测时的特征提取。

        Args:
            messages: 要处理的消息列表

        Returns:
            已修改的消息列表（就地修改）
        """
        for message in messages:  # 遍历每个消息
            self._text_features_with_regex(message, TEXT)  # 为文本属性生成正则表达式特征

        return messages

    def _text_features_with_regex(self, message: Message, attribute: Text) -> None:
        """提取特征并适当设置到消息中的辅助方法。

        该方法检查消息的指定属性是否匹配已知的正则表达式模式，
        并生成相应的序列特征和句子特征。

        Args:
            message: 要进行特征化的消息
            attribute: 要进行特征化的消息属性
        """
        if self.known_patterns:  # 如果存在已知模式
            # 为模式生成特征
            sequence_features, sentence_features = self._features_for_patterns(
                message, attribute
            )

            # 将特征添加到消息中
            self.add_features_to_message(
                sequence_features, sentence_features, attribute, message
            )

    def _features_for_patterns(
        self, message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.coo_matrix], Optional[scipy.sparse.coo_matrix]]:
        """检查哪些已知模式匹配消息。

        给定一个句子，返回一个 {1,0} 值的向量，指示哪些正则表达式匹配。
        此外，如果消息被标记化，函数将在所有标记上标记一个字典，
        将正则表达式的名称与是否匹配相关联。

        Args:
            message: 要进行特征化的消息
            attribute: 要进行特征化的消息属性

        Returns:
           消息属性的标记级别和句子级别特征
        """
        # 属性未设置（例如响应不存在）
        if not message.get(attribute):
            return None, None

        tokens = message.get(TOKENS_NAMES[attribute], [])  # 获取标记

        if not tokens:
            # 没有内容进行特征化
            return None, None

        flags = 0  # 默认标志
        if not self.case_sensitive:  # 如果不区分大小写
            flags = re.IGNORECASE  # 设置忽略大小写标志

        sequence_length = len(tokens)  # 序列长度
        num_patterns = len(self.known_patterns)  # 模式数量

        # 初始化特征矩阵
        sequence_features = np.zeros([sequence_length, num_patterns])  # 序列特征矩阵
        sentence_features = np.zeros([1, num_patterns])  # 句子特征矩阵

        for pattern_index, pattern in enumerate(self.known_patterns):  # 遍历每个模式
            # 查找所有匹配项
            matches = list(
                re.finditer(pattern["pattern"], message.get(attribute), flags=flags)
            )

            for token_index, t in enumerate(tokens):  # 遍历每个标记
                patterns = t.get("pattern", default={})  # 获取标记的模式信息
                patterns[pattern["name"]] = False  # 初始化为未匹配

                for match in matches:  # 检查每个匹配项
                    # 检查标记是否与匹配项重叠
                    if t.start < match.end() and t.end > match.start():
                        patterns[pattern["name"]] = True  # 标记为匹配
                        sequence_features[token_index][pattern_index] = 1.0  # 设置序列特征
                        if attribute in [RESPONSE, TEXT, ACTION_TEXT]:  # 如果是文本相关属性
                            # 句子向量应包含所有模式
                            sentence_features[0][pattern_index] = 1.0

                t.set("pattern", patterns)  # 设置标记的模式信息

        return (
            scipy.sparse.coo_matrix(sequence_features),  # 返回稀疏序列特征矩阵
            scipy.sparse.coo_matrix(sentence_features),  # 返回稀疏句子特征矩阵
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> RegexFeaturizer:
        """加载已训练的组件。
        
        该方法从模型存储中加载已训练的正则表达式模式，
        并创建相应的 RegexFeaturizer 实例。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            **kwargs: 其他关键字参数
            
        Returns:
            加载的 RegexFeaturizer 实例
        """
        known_patterns = None  # 初始化已知模式为 None

        try:
            with model_storage.read_from(resource) as model_dir:  # 从模型存储中读取
                patterns_file_name = model_dir / "patterns.json"  # 模式文件路径
                known_patterns = rasa.shared.utils.io.read_json_file(patterns_file_name)  # 读取模式文件
        except (ValueError, FileNotFoundError):  # 捕获文件不存在或格式错误异常
            logger.warning(
                f"Failed to load `{cls.__class__.__name__}` from model storage. "
                f"Resource '{resource.name}' doesn't exist."
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            known_patterns=known_patterns,  # 传入加载的模式
        )

    def _persist(self) -> None:
        """持久化组件到模型存储。
        
        该方法将训练好的正则表达式模式保存到模型存储中，
        以便后续加载和使用。
        """
        with self._model_storage.write_to(self._resource) as model_dir:  # 写入模型存储
            regex_file = model_dir / "patterns.json"  # 模式文件路径
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                regex_file, self.known_patterns  # 保存已知模式到 JSON 文件
            )

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """验证组件配置是否正确。
        
        该方法检查配置参数的有效性，确保组件能够正常运行。
        当前实现为空，表示所有配置都被认为是有效的。
        
        Args:
            config: 要验证的配置字典
        """
        pass  # 当前没有配置验证逻辑
