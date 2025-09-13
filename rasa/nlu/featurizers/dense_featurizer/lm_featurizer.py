# 导入未来版本注解支持，允许使用字符串形式的类型注解
from __future__ import annotations

# 导入科学计算库
import numpy as np  # 数值计算库，提供多维数组和数学运算功能
import logging  # 日志记录模块，用于记录程序运行状态和调试信息

# 导入类型注解模块
from typing import Any, Text, List, Dict, Tuple, Type  # 类型注解模块，提供类型提示功能

# 导入深度学习框架
import tensorflow as tf  # TensorFlow 深度学习框架，用于模型推理

# 导入 Rasa 核心模块
from rasa.engine.graph import ExecutionContext, GraphComponent  # 图组件和执行上下文，用于组件管理和执行控制
from rasa.engine.recipes.default_recipe import DefaultV1Recipe  # 默认配方，用于组件注册和配置
from rasa.engine.storage.resource import Resource  # 资源管理，用于模型资源的存储和访问
from rasa.engine.storage.storage import ModelStorage  # 模型存储，提供模型持久化功能
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer  # 密集特征化器基类，提供密集特征提取的通用接口
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer  # 标记化器，用于将文本分割为标记
from rasa.shared.nlu.training_data.training_data import TrainingData  # 训练数据类，包含所有训练样本
from rasa.shared.nlu.training_data.message import Message  # 消息类，表示训练数据中的单条消息
from rasa.nlu.constants import (  # NLU 常量导入
    DENSE_FEATURIZABLE_ATTRIBUTES,  # 可密集特征化的属性列表
    SEQUENCE_FEATURES,  # 序列特征常量
    SENTENCE_FEATURES,  # 句子特征常量
    NO_LENGTH_RESTRICTION,  # 无长度限制常量
    NUMBER_OF_SUB_TOKENS,  # 子标记数量常量
    TOKENS_NAMES,  # 标记名称常量，定义各种标记的键名
)
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT  # NLU 常量，定义消息属性的键名
from rasa.utils import train_utils  # 训练工具模块，提供训练相关功能
from rasa.utils.tensorflow.model_data import ragged_array_to_ndarray  # TensorFlow 模型数据工具，用于不规则数组转换

# 初始化日志记录器，用于记录组件的运行状态和调试信息
logger = logging.getLogger(__name__)

# 各种语言模型的最大序列长度限制字典
MAX_SEQUENCE_LENGTHS = {
    "bert": 512,  # BERT 模型最大序列长度为 512
    "gpt": 512,  # GPT 模型最大序列长度为 512
    "gpt2": 512,  # GPT-2 模型最大序列长度为 512
    "xlnet": NO_LENGTH_RESTRICTION,  # XLNet 模型无序列长度限制
    "distilbert": 512,  # DistilBERT 模型最大序列长度为 512
    "roberta": 512,  # RoBERTa 模型最大序列长度为 512
    "camembert": 512,  # CamemBERT 模型最大序列长度为 512
}


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False  # 注册为消息特征化器组件类型，标记为不可训练组件
)
class LanguageModelFeaturizer(DenseFeaturizer, GraphComponent):
    """基于 Transformer 语言模型的特征化器。

    该类继承自 DenseFeaturizer 和 GraphComponent，实现了基于预训练语言模型的密集特征提取。
    该组件从 Transformers 库（https://github.com/huggingface/transformers）加载预训练的语言模型，
    包括 BERT、GPT、GPT-2、XLNet、DistilBERT、RoBERTa 和 CamemBERT。
    它还对每个消息的可密集特征化属性进行标记化和特征化处理。
    
    该特征化器能够生成高质量的上下文感知的文本嵌入表示，
    适用于各种自然语言理解任务。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """获取此组件运行前必须包含在管道中的组件类型。
        
        该方法定义了组件的依赖关系，确保在特征提取之前文本已经被正确标记化。
        
        Returns:
            必需的组件类型列表，包含标记化器组件
        """
        return [Tokenizer]  # 需要标记化器组件，用于将文本分割为标记

    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        """使用配置中的模型初始化特征化器。
        
        该构造函数初始化 LanguageModelFeaturizer 实例，加载模型元数据，
        并实例化预训练的语言模型和标记化器。
        
        Args:
            config: 组件配置字典
            execution_context: 执行上下文，包含节点名称和执行模式信息
        """
        super(LanguageModelFeaturizer, self).__init__(
            execution_context.node_name, config  # 调用父类构造函数
        )
        self._load_model_metadata()  # 加载模型元数据
        self._load_model_instance()  # 加载模型实例

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回 LanguageModelFeaturizer 的默认配置参数。
        
        该方法定义了 LanguageModelFeaturizer 的所有可配置参数及其默认值。
        配置参数主要控制语言模型的选择和加载方式。
        
        Returns:
            包含所有配置参数及其默认值的字典
        """
        return {
            **DenseFeaturizer.get_default_config(),  # 继承密集特征化器基类的默认配置
            # 语言模型配置
            "model_name": "bert",  # 要加载的语言模型名称，默认为 BERT
            "model_weights": None,  # 预训练权重，None 表示使用默认权重
            "cache_dir": None,  # 可选的缓存目录路径，用于下载和缓存预训练模型权重
        }

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """验证组件配置是否正确。
        
        该方法检查配置参数的有效性，确保组件能够正常运行。
        当前实现为空，表示所有配置都被认为是有效的。
        
        Args:
            config: 要验证的配置字典
        """
        pass  # 当前没有配置验证逻辑

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LanguageModelFeaturizer:
        """创建 LanguageModelFeaturizer 实例。

        该方法创建新的 LanguageModelFeaturizer 实例，并加载配置中指定的模型。
        
        Args:
            config: 组件配置字典
            model_storage: 模型存储接口
            resource: 资源管理对象
            execution_context: 执行上下文
            
        Returns:
            新的 LanguageModelFeaturizer 实例
        """
        return cls(config, execution_context)

    @staticmethod
    def required_packages() -> List[Text]:
        """获取此组件运行所需的额外 Python 依赖项。
        
        该方法定义了组件运行所需的外部包，确保在组件初始化前已安装必要的依赖。
        
        Returns:
            依赖包名称列表，包含 transformers 包
        """
        return ["transformers"]  # 需要 Hugging Face Transformers 库

    def _load_model_metadata(self) -> None:
        """加载指定模型的元数据并设置为属性。

        该方法加载模型名称、模型权重、缓存目录和模型能处理的最大序列长度等元数据。
        这些元数据将用于后续的模型加载和特征提取过程。
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,  # 模型类字典
            model_weights_defaults,  # 模型权重默认值字典
        )

        # 获取模型名称
        self.model_name = self._config["model_name"]

        # 验证模型名称是否有效
        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create"
                f"a new class inheriting from this class to support your model."
            )

        # 获取模型权重和缓存目录
        self.model_weights = self._config["model_weights"]
        self.cache_dir = self._config["cache_dir"]

        # 如果没有指定模型权重，使用默认权重
        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model "
                f"weights: {model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

        # 设置模型的最大序列长度
        self.max_model_sequence_length = MAX_SEQUENCE_LENGTHS[self.model_name]

    def _load_model_instance(self) -> None:
        """尝试加载模型实例。

        该方法加载预训练的语言模型和标记化器实例。
        在单元测试中应跳过模型加载，参见单元测试示例。

        Note:
            模型加载过程包括：
            1. 加载标记化器
            2. 加载预训练模型
            3. 设置填充标记 ID
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,  # 模型类字典
            model_tokenizer_dict,  # 标记化器类字典
        )

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        # 加载标记化器
        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )
        # 加载预训练模型
        self.model = model_class_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )

        # 使用通用填充标记，因为所有 Transformer 架构都没有一致的填充标记
        # 使用 unk_token_id 而不是 pad_token_id，因为并非所有架构都设置了 pad_token_id
        # 我们不能添加新标记，因为 TF 类还不支持词汇表调整大小
        # 这不会影响模型预测，因为我们在输入时使用注意力掩码
        self.pad_token_id = self.tokenizer.unk_token_id

    def _lm_tokenize(self, text: Text) -> Tuple[List[int], List[Text]]:
        """Passes the text through the tokenizer of the language model.

        Args:
            text: Text to be tokenized.

        Returns: List of token ids and token strings.
        """
        split_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        split_token_strings = self.tokenizer.convert_ids_to_tokens(split_token_ids)

        return split_token_ids, split_token_strings

    def _add_lm_specific_special_tokens(
        self, token_ids: List[List[int]]
    ) -> List[List[int]]:
        """Adds the language and model-specific tokens used during training.

        Args:
            token_ids: List of token ids for each example in the batch.

        Returns: Augmented list of token ids for each example in the batch.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_special_tokens_pre_processors,
        )

        augmented_tokens = [
            model_special_tokens_pre_processors[self.model_name](example_token_ids)
            for example_token_ids in token_ids
        ]
        return augmented_tokens

    def _lm_specific_token_cleanup(
        self, split_token_ids: List[int], token_strings: List[Text]
    ) -> Tuple[List[int], List[Text]]:
        """Cleans up special chars added by tokenizers of language models.

        Many language models add a special char in front/back of (some) words. We clean
        up those chars as they are not
        needed once the features are already computed.

        Args:
            split_token_ids: List of token ids received as output from the language
            model specific tokenizer.
            token_strings: List of token strings received as output from the language
            model specific tokenizer.

        Returns: Cleaned up token ids and token strings.
        """
        from rasa.nlu.utils.hugging_face.registry import model_tokens_cleaners

        return model_tokens_cleaners[self.model_name](split_token_ids, token_strings)

    def _post_process_sequence_embeddings(
        self, sequence_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes sentence and sequence level representations for relevant tokens.

        Args:
            sequence_embeddings: Sequence level dense features received as output from
            language model.

        Returns: Sentence and sequence level representations.
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_embeddings_post_processors,
        )

        sentence_embeddings = []
        post_processed_sequence_embeddings = []

        for example_embedding in sequence_embeddings:
            (
                example_sentence_embedding,
                example_post_processed_embedding,
            ) = model_embeddings_post_processors[self.model_name](example_embedding)

            sentence_embeddings.append(example_sentence_embedding)
            post_processed_sequence_embeddings.append(example_post_processed_embedding)

        return (
            np.array(sentence_embeddings),
            ragged_array_to_ndarray(post_processed_sequence_embeddings),
        )

    def _tokenize_example(
        self, message: Message, attribute: Text
    ) -> Tuple[List[Token], List[int]]:
        """Tokenizes a single message example.

        Many language models add a special char in front of (some) words and split
        words into sub-words. To ensure the entity start and end values matches the
        token values, use the tokens produced by the Tokenizer component. If
        individual tokens are split up into multiple tokens, we add this information
        to the respected token.

        Args:
            message: Single message object to be processed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns: List of token strings and token ids for the corresponding
                attribute of the message.
        """
        tokens_in = message.get(TOKENS_NAMES[attribute])
        tokens_out = []

        token_ids_out = []

        for token in tokens_in:
            # use lm specific tokenizer to further tokenize the text
            split_token_ids, split_token_strings = self._lm_tokenize(token.text)

            if not split_token_ids:
                # fix the situation that `token.text` only contains whitespace or other
                # special characters, which cause `split_token_ids` and
                # `split_token_strings` be empty, finally cause
                # `self._lm_specific_token_cleanup()` to raise an exception
                continue

            (split_token_ids, split_token_strings) = self._lm_specific_token_cleanup(
                split_token_ids, split_token_strings
            )

            token_ids_out += split_token_ids

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out, token_ids_out

    def _get_token_ids_for_batch(
        self, batch_examples: List[Message], attribute: Text
    ) -> Tuple[List[List[Token]], List[List[int]]]:
        """Computes token ids and token strings for each example in batch.

        A token id is the id of that token in the vocabulary of the language model.

        Args:
            batch_examples: Batch of message objects for which tokens need to be
            computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.

        Returns: List of token strings and token ids for each example in the batch.
        """
        batch_token_ids = []
        batch_tokens = []
        for example in batch_examples:

            example_tokens, example_token_ids = self._tokenize_example(
                example, attribute
            )
            batch_tokens.append(example_tokens)
            batch_token_ids.append(example_token_ids)

        return batch_tokens, batch_token_ids

    @staticmethod
    def _compute_attention_mask(
        actual_sequence_lengths: List[int], max_input_sequence_length: int
    ) -> np.ndarray:
        """Computes a mask for padding tokens.

        This mask will be used by the language model so that it does not attend to
        padding tokens.

        Args:
            actual_sequence_lengths: List of length of each example without any
            padding.
            max_input_sequence_length: Maximum length of a sequence that will be
            present in the input batch. This is
            after taking into consideration the maximum input sequence the model
            can handle. Hence it can never be
            greater than self.max_model_sequence_length in case the model
            applies length restriction.

        Returns: Computed attention mask, 0 for padding and 1 for non-padding
        tokens.
        """
        attention_mask = []

        for actual_sequence_length in actual_sequence_lengths:
            # add 1s for present tokens, fill up the remaining space up to max
            # sequence length with 0s (non-existing tokens)
            padded_sequence = [1] * min(
                actual_sequence_length, max_input_sequence_length
            ) + [0] * (
                max_input_sequence_length
                - min(actual_sequence_length, max_input_sequence_length)
            )
            attention_mask.append(padded_sequence)

        return np.array(attention_mask).astype(np.float32)

    def _extract_sequence_lengths(
        self, batch_token_ids: List[List[int]]
    ) -> Tuple[List[int], int]:
        """Extracts the sequence length for each example and maximum sequence length.

        Args:
            batch_token_ids: List of token ids for each example in the batch.

        Returns:
            Tuple consisting of: the actual sequence lengths for each example,
            and the maximum input sequence length (taking into account the
            maximum sequence length that the model can handle.
        """
        # Compute max length across examples
        max_input_sequence_length = 0
        actual_sequence_lengths = []

        for example_token_ids in batch_token_ids:
            sequence_length = len(example_token_ids)
            actual_sequence_lengths.append(sequence_length)
            max_input_sequence_length = max(
                max_input_sequence_length, len(example_token_ids)
            )

        # Take into account the maximum sequence length the model can handle
        max_input_sequence_length = (
            max_input_sequence_length
            if self.max_model_sequence_length == NO_LENGTH_RESTRICTION
            else min(max_input_sequence_length, self.max_model_sequence_length)
        )

        return actual_sequence_lengths, max_input_sequence_length

    def _add_padding_to_batch(
        self, batch_token_ids: List[List[int]], max_sequence_length_model: int
    ) -> List[List[int]]:
        """Adds padding so that all examples in the batch are of the same length.

        Args:
            batch_token_ids: Batch of examples where each example is a non-padded list
            of token ids.
            max_sequence_length_model: Maximum length of any input sequence in the batch
            to be fed to the model.

        Returns:
            Padded batch with all examples of the same length.
        """
        padded_token_ids = []

        # Add padding according to max_sequence_length
        # Some models don't contain pad token, we use unknown token as padding token.
        # This doesn't affect the computation since we compute an attention mask
        # anyways.
        for example_token_ids in batch_token_ids:

            # Truncate any longer sequences so that they can be fed to the model
            if len(example_token_ids) > max_sequence_length_model:
                example_token_ids = example_token_ids[:max_sequence_length_model]

            padded_token_ids.append(
                example_token_ids
                + [self.pad_token_id]
                * (max_sequence_length_model - len(example_token_ids))
            )
        return padded_token_ids

    @staticmethod
    def _extract_nonpadded_embeddings(
        embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Extracts embeddings for actual tokens.

        Use pre-computed non-padded lengths of each example to extract embeddings
        for non-padding tokens.

        Args:
            embeddings: sequence level representations for each example of the batch.
            actual_sequence_lengths: non-padded lengths of each example of the batch.

        Returns:
            Sequence level embeddings for only non-padding tokens of the batch.
        """
        nonpadded_sequence_embeddings = []
        for index, embedding in enumerate(embeddings):
            unmasked_embedding = embedding[: actual_sequence_lengths[index]]
            nonpadded_sequence_embeddings.append(unmasked_embedding)

        return ragged_array_to_ndarray(nonpadded_sequence_embeddings)

    def _compute_batch_sequence_features(
        self, batch_attention_mask: np.ndarray, padded_token_ids: List[List[int]]
    ) -> np.ndarray:
        """Feeds the padded batch to the language model.

        Args:
            batch_attention_mask: Mask of 0s and 1s which indicate whether the token
            is a padding token or not.
            padded_token_ids: Batch of token ids for each example. The batch is padded
            and hence can be fed at once.

        Returns:
            Sequence level representations from the language model.
        """
        model_outputs = self.model(
            tf.convert_to_tensor(padded_token_ids),
            attention_mask=tf.convert_to_tensor(batch_attention_mask),
        )

        # sequence hidden states is always the first output from all models
        sequence_hidden_states = model_outputs[0]

        sequence_hidden_states = sequence_hidden_states.numpy()
        return sequence_hidden_states

    def _validate_sequence_lengths(
        self,
        actual_sequence_lengths: List[int],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> None:
        """Validates sequence length.

        Checks if sequence lengths of inputs are less than
        the max sequence length the model can handle.

        This method should throw an error during training, and log a debug
        message during inference if any of the input examples have a length
        greater than maximum sequence length allowed.

        Args:
            actual_sequence_lengths: original sequence length of all inputs
            batch_examples: all message instances in the batch
            attribute: attribute of message object to be processed
            inference_mode: whether this is during training or inference
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # There is no restriction on sequence length from the model
            return

        for sequence_length, example in zip(actual_sequence_lengths, batch_examples):
            if sequence_length > self.max_model_sequence_length:
                if not inference_mode:
                    raise RuntimeError(
                        f"The sequence length of '{example.get(attribute)[:20]}...' "
                        f"is too long({sequence_length} tokens) for the "
                        f"model chosen {self.model_name} which has a maximum "
                        f"sequence length of {self.max_model_sequence_length} tokens. "
                        f"Either shorten the message or use a model which has no "
                        f"restriction on input sequence length like XLNet."
                    )
                logger.debug(
                    f"The sequence length of '{example.get(attribute)[:20]}...' "
                    f"is too long({sequence_length} tokens) for the "
                    f"model chosen {self.model_name} which has a maximum "
                    f"sequence length of {self.max_model_sequence_length} tokens. "
                    f"Downstream model predictions may be affected because of this."
                )

    def _add_extra_padding(
        self, sequence_embeddings: np.ndarray, actual_sequence_lengths: List[int]
    ) -> np.ndarray:
        """Adds extra zero padding to match the original sequence length.

        This is only done if the input was truncated during the batch
        preparation of input for the model.
        Args:
            sequence_embeddings: Embeddings returned from the model
            actual_sequence_lengths: original sequence length of all inputs

        Returns:
            Modified sequence embeddings with padding if necessary
        """
        if self.max_model_sequence_length == NO_LENGTH_RESTRICTION:
            # No extra padding needed because there wouldn't have been any
            # truncation in the first place
            return sequence_embeddings

        reshaped_sequence_embeddings = []
        for index, embedding in enumerate(sequence_embeddings):
            embedding_size = embedding.shape[-1]
            if actual_sequence_lengths[index] > self.max_model_sequence_length:
                embedding = np.concatenate(
                    [
                        embedding,
                        np.zeros(
                            (
                                actual_sequence_lengths[index]
                                - self.max_model_sequence_length,
                                embedding_size,
                            ),
                            dtype=np.float32,
                        ),
                    ]
                )
            reshaped_sequence_embeddings.append(embedding)
        return ragged_array_to_ndarray(reshaped_sequence_embeddings)

    def _get_model_features_for_batch(
        self,
        batch_token_ids: List[List[int]],
        batch_tokens: List[List[Token]],
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes dense features of each example in the batch.

        We first add the special tokens corresponding to each language model. Next, we
        add appropriate padding and compute a mask for that padding so that it doesn't
        affect the feature computation. The padded batch is next fed to the language
        model and token level embeddings are computed. Using the pre-computed mask,
        embeddings for non-padding tokens are extracted and subsequently sentence
        level embeddings are computed.

        Args:
            batch_token_ids: List of token ids of each example in the batch.
            batch_tokens: List of token objects for each example in the batch.
            batch_examples: List of examples in the batch.
            attribute: attribute of the Message object to be processed.
            inference_mode: Whether the call is during training or during inference.

        Returns:
            Sentence and token level dense representations.
        """
        # Let's first add tokenizer specific special tokens to all examples
        batch_token_ids_augmented = self._add_lm_specific_special_tokens(
            batch_token_ids
        )

        # Compute sequence lengths for all examples
        (
            actual_sequence_lengths,
            max_input_sequence_length,
        ) = self._extract_sequence_lengths(batch_token_ids_augmented)

        # Validate that all sequences can be processed based on their sequence
        # lengths and the maximum sequence length the model can handle
        self._validate_sequence_lengths(
            actual_sequence_lengths, batch_examples, attribute, inference_mode
        )

        # Add padding so that whole batch can be fed to the model
        padded_token_ids = self._add_padding_to_batch(
            batch_token_ids_augmented, max_input_sequence_length
        )

        # Compute attention mask based on actual_sequence_length
        batch_attention_mask = self._compute_attention_mask(
            actual_sequence_lengths, max_input_sequence_length
        )

        # Get token level features from the model
        sequence_hidden_states = self._compute_batch_sequence_features(
            batch_attention_mask, padded_token_ids
        )

        # Extract features for only non-padding tokens
        sequence_nonpadded_embeddings = self._extract_nonpadded_embeddings(
            sequence_hidden_states, actual_sequence_lengths
        )

        # Extract sentence level and post-processed features
        (
            sentence_embeddings,
            sequence_embeddings,
        ) = self._post_process_sequence_embeddings(sequence_nonpadded_embeddings)

        # Pad zeros for examples which were truncated in inference mode.
        # This is intentionally done after sentence embeddings have been
        # extracted so that they are not affected
        sequence_embeddings = self._add_extra_padding(
            sequence_embeddings, actual_sequence_lengths
        )

        # shape of matrix for all sequence embeddings
        batch_dim = len(sequence_embeddings)
        seq_dim = max(e.shape[0] for e in sequence_embeddings)
        feature_dim = sequence_embeddings[0].shape[1]
        shape = (batch_dim, seq_dim, feature_dim)

        # align features with tokens so that we have just one vector per token
        # (don't include sub-tokens)
        sequence_embeddings = train_utils.align_token_features(
            batch_tokens, sequence_embeddings, shape
        )

        # sequence_embeddings is a padded numpy array
        # remove the padding, keep just the non-zero vectors
        sequence_final_embeddings = []
        for embeddings, tokens in zip(sequence_embeddings, batch_tokens):
            sequence_final_embeddings.append(embeddings[: len(tokens)])

        return sentence_embeddings, ragged_array_to_ndarray(sequence_final_embeddings)

    def _get_docs_for_batch(
        self,
        batch_examples: List[Message],
        attribute: Text,
        inference_mode: bool = False,
    ) -> List[Dict[Text, Any]]:
        """Computes language model docs for all examples in the batch.

        Args:
            batch_examples: Batch of message objects for which language model docs
            need to be computed.
            attribute: Property of message to be processed, one of ``TEXT`` or
            ``RESPONSE``.
            inference_mode: Whether the call is during inference or during training.


        Returns:
            List of language model docs for each message in batch.
        """
        batch_tokens, batch_token_ids = self._get_token_ids_for_batch(
            batch_examples, attribute
        )

        (
            batch_sentence_features,
            batch_sequence_features,
        ) = self._get_model_features_for_batch(
            batch_token_ids, batch_tokens, batch_examples, attribute, inference_mode
        )

        # A doc consists of
        # {'sequence_features': ..., 'sentence_features': ...}
        batch_docs = []
        for index in range(len(batch_examples)):
            doc = {
                SEQUENCE_FEATURES: batch_sequence_features[index],
                SENTENCE_FEATURES: np.reshape(batch_sentence_features[index], (1, -1)),
            }
            batch_docs.append(doc)

        return batch_docs

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Computes tokens and dense features for each message in training data.

        Args:
            training_data: NLU training data to be tokenized and featurized
            config: NLU pipeline config consisting of all components.
        """
        batch_size = 64

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            batch_start_index = 0

            while batch_start_index < len(non_empty_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )
                # Collect batch examples
                batch_messages = non_empty_examples[batch_start_index:batch_end_index]

                # Construct a doc with relevant features
                # extracted(tokens, dense_features)
                batch_docs = self._get_docs_for_batch(batch_messages, attribute)

                for index, ex in enumerate(batch_messages):
                    self._set_lm_features(batch_docs[index], ex, attribute)
                batch_start_index += batch_size

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes messages by computing tokens and dense features."""
        for message in messages:
            self._process_message(message)
        return messages

    def _process_message(self, message: Message) -> Message:
        """Processes a message by computing tokens and dense features."""
        # processing featurizers operates only on TEXT and ACTION_TEXT attributes,
        # because all other attributes are labels which are featurized during
        # training and their features are stored by the model itself.
        for attribute in {TEXT, ACTION_TEXT}:
            if message.get(attribute):
                self._set_lm_features(
                    self._get_docs_for_batch(
                        [message], attribute=attribute, inference_mode=True
                    )[0],
                    message,
                    attribute,
                )
        return message

    def _set_lm_features(
        self, doc: Dict[Text, Any], message: Message, attribute: Text = TEXT
    ) -> None:
        """Adds the precomputed word vectors to the messages features."""
        sequence_features = doc[SEQUENCE_FEATURES]
        sentence_features = doc[SENTENCE_FEATURES]

        self.add_features_to_message(
            sequence=sequence_features,
            sentence=sentence_features,
            attribute=attribute,
            message=message,
        )
