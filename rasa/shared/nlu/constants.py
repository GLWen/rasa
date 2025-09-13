# =============================================================================
# NLU 模块常量定义 - 定义自然语言理解相关的常量
# =============================================================================

# =============================================================================
# 消息属性常量 - 定义消息对象中的各种属性键名
# =============================================================================
TEXT = "text"  # 文本内容
TEXT_TOKENS = "text_tokens"  # 文本分词结果
INTENT = "intent"  # 意图
NOT_INTENT = "not_intent"  # 非意图（用于排除特定意图）
RESPONSE = "response"  # 响应
RESPONSE_SELECTOR = "response_selector"  # 响应选择器
INTENT_RESPONSE_KEY = "intent_response_key"  # 意图响应键
ACTION_TEXT = "action_text"  # 动作文本
ACTION_NAME = "action_name"  # 动作名称
INTENT_NAME_KEY = "name"  # 意图名称键
FULL_RETRIEVAL_INTENT_NAME_KEY = "full_retrieval_intent_name"  # 完整检索意图名称键
METADATA = "metadata"  # 元数据
METADATA_INTENT = "intent"  # 意图元数据
METADATA_EXAMPLE = "example"  # 示例元数据
METADATA_MODEL_ID = "model_id"  # 模型ID元数据
INTENT_RANKING_KEY = "intent_ranking"  # 意图排名键
PREDICTED_CONFIDENCE_KEY = "confidence"  # 预测置信度键

# =============================================================================
# 响应标识符分隔符
# =============================================================================
RESPONSE_IDENTIFIER_DELIMITER = "/"  # 响应标识符分隔符，用于分隔意图和响应键

# =============================================================================
# 特征类型常量 - 定义不同类型的特征
# =============================================================================
FEATURE_TYPE_SENTENCE = "sentence"  # 句子级特征
FEATURE_TYPE_SEQUENCE = "sequence"  # 序列级特征
VALID_FEATURE_TYPES = [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]  # 有效特征类型列表

# =============================================================================
# 实体提取器常量 - 定义不同类型的实体提取器
# =============================================================================
EXTRACTOR = "extractor"  # 提取器标识符
PRETRAINED_EXTRACTORS = {"DucklingEntityExtractor", "SpacyEntityExtractor"}  # 预训练提取器
TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}  # 可训练提取器

# =============================================================================
# 实体相关常量 - 定义实体相关的各种属性
# =============================================================================
ENTITIES = "entities"  # 实体列表
ENTITY_TAGS = "entity_tags"  # 实体标签
ENTITY_ATTRIBUTE_TYPE = "entity"  # 实体类型属性
ENTITY_ATTRIBUTE_GROUP = "group"  # 实体组属性
ENTITY_ATTRIBUTE_ROLE = "role"  # 实体角色属性
ENTITY_ATTRIBUTE_VALUE = "value"  # 实体值属性
ENTITY_ATTRIBUTE_START = "start"  # 实体开始位置属性
ENTITY_ATTRIBUTE_END = "end"  # 实体结束位置属性
ENTITY_ATTRIBUTE_TEXT = "text"  # 实体文本属性
ENTITY_ATTRIBUTE_CONFIDENCE = "confidence"  # 实体置信度属性
NO_ENTITY_TAG = "O"  # 非实体标签（BIO标注中的O标签）
SPLIT_ENTITIES_BY_COMMA = "split_entities_by_comma"  # 按逗号分割实体配置
SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE = True  # 按逗号分割实体的默认值
SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET = {".", ",", " ", ";"}  # 单个实体允许的交错字符集
