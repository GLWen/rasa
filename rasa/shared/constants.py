from typing import List, Text

# =============================================================================
# 文档链接常量 - 定义 Rasa 官方文档的 URL 地址
# =============================================================================

# 基础文档 URL
DOCS_BASE_URL = "https://rasa.com/docs/rasa"  # Rasa 开源版本文档基础 URL
LEGACY_DOCS_BASE_URL = "https://legacy-docs-v1.rasa.com"  # 旧版本文档 URL

# 训练数据相关文档链接
DOCS_URL_TRAINING_DATA = DOCS_BASE_URL + "/training-data-format"  # 训练数据格式文档
DOCS_URL_TRAINING_DATA_NLU = DOCS_URL_TRAINING_DATA + "#nlu-training-data"  # NLU 训练数据文档

# 领域（Domain）相关文档链接
DOCS_URL_DOMAINS = DOCS_BASE_URL + "/domain"  # 领域定义文档
DOCS_URL_SLOTS = DOCS_URL_DOMAINS + "#slots"  # 槽位定义文档
DOCS_URL_INTENTS = DOCS_URL_DOMAINS + "#intents"  # 意图定义文档
DOCS_URL_ENTITIES = DOCS_URL_DOMAINS + "#entities"  # 实体定义文档

# 响应和对话相关文档链接
DOCS_URL_RESPONSES = DOCS_BASE_URL + "/responses"  # 响应定义文档
DOCS_URL_STORIES = DOCS_BASE_URL + "/stories"  # 故事定义文档
DOCS_URL_RULES = DOCS_BASE_URL + "/rules"  # 规则定义文档
DOCS_URL_FORMS = DOCS_BASE_URL + "/forms"  # 表单定义文档

# 模型和策略相关文档链接
DOCS_URL_PIPELINE = DOCS_BASE_URL + "/tuning-your-model"  # 模型调优文档
DOCS_URL_POLICIES = DOCS_BASE_URL + "/policies"  # 策略配置文档
DOCS_URL_TEST_STORIES = DOCS_BASE_URL + "/testing-your-assistant"  # 测试故事文档
DOCS_URL_MARKERS = DOCS_BASE_URL + "/markers"  # 标记器文档

# 动作相关文档链接
DOCS_URL_ACTIONS = DOCS_BASE_URL + "/actions"  # 自定义动作文档
DOCS_URL_DEFAULT_ACTIONS = DOCS_BASE_URL + "/default-actions"  # 默认动作文档

# 连接器和集成相关文档链接
DOCS_URL_CONNECTORS = DOCS_BASE_URL + "/connectors/"  # 连接器文档
DOCS_URL_CONNECTORS_SLACK = DOCS_URL_CONNECTORS + "/slack"  # Slack 连接器文档
DOCS_URL_EVENT_BROKERS = DOCS_BASE_URL + "/event-brokers"  # 事件代理文档
DOCS_URL_PIKA_EVENT_BROKER = DOCS_URL_EVENT_BROKERS + "#pika-event-broker"  # Pika 事件代理文档
DOCS_URL_TRACKER_STORES = DOCS_BASE_URL + "/tracker-stores"  # 跟踪器存储文档

# 组件和图形相关文档链接
DOCS_URL_COMPONENTS = DOCS_BASE_URL + "/components"  # 组件文档
DOCS_URL_GRAPH_COMPONENTS = DOCS_BASE_URL + "/custom-graph-components"  # 自定义图形组件文档
DOCS_URL_GRAPH_RECIPE = DOCS_BASE_URL + "/graph-recipe"  # 图形配方文档

# 迁移和遥测相关文档链接
DOCS_URL_MIGRATION_GUIDE = DOCS_BASE_URL + "/migration-guide"  # 迁移指南文档
DOCS_URL_MIGRATION_GUIDE_MD_DEPRECATION = (
    f"{DOCS_URL_MIGRATION_GUIDE}#rasa-21-to-rasa-22"  # Rasa 2.1 到 2.2 迁移文档
)
DOCS_URL_TELEMETRY = DOCS_BASE_URL + "/telemetry/telemetry"  # 遥测文档

# 企业版和动作服务器文档链接
DOCS_BASE_URL_RASA_X = "https://rasa.com/docs/rasa-enterprise"  # Rasa Enterprise 文档
DOCS_BASE_URL_ACTION_SERVER = "https://rasa.com/docs/action-server"  # 动作服务器文档

# =============================================================================
# 基础标识符常量
# =============================================================================

INTENT_MESSAGE_PREFIX = "/"  # 意图消息的前缀标识符

PACKAGE_NAME = "rasa"  # Rasa 包名称
NEXT_MAJOR_VERSION_FOR_DEPRECATIONS = "4.0.0"  # 下一个主要版本号，用于弃用功能标记

# =============================================================================
# Schema 文件路径常量 - 定义各种配置文件的 Schema 验证文件路径
# =============================================================================

MODEL_CONFIG_SCHEMA_FILE = "shared/utils/schemas/model_config.yml"  # 模型配置 Schema 文件路径
CONFIG_SCHEMA_FILE = "shared/utils/schemas/config.yml"  # 主配置文件 Schema 文件路径
RESPONSES_SCHEMA_FILE = "shared/nlu/training_data/schemas/responses.yml"  # 响应定义 Schema 文件路径
SCHEMA_EXTENSIONS_FILE = "shared/utils/pykwalify_extensions.py"  # Schema 扩展文件路径
LATEST_TRAINING_DATA_FORMAT_VERSION = "3.1"  # 最新训练数据格式版本号

DOMAIN_SCHEMA_FILE = "shared/utils/schemas/domain.yml"  # 领域定义 Schema 文件路径

# =============================================================================
# 会话和默认值常量
# =============================================================================

DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES = 60  # 默认会话过期时间（分钟）
DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION = True  # 是否将槽位值传递到新会话

DEFAULT_NLU_FALLBACK_INTENT_NAME = "nlu_fallback"  # 默认 NLU 回退意图名称

# =============================================================================
# 测试和日志相关常量
# =============================================================================

DEFAULT_E2E_TESTS_PATH = "."  # 默认端到端测试路径
TEST_STORIES_FILE_PREFIX = "test_"  # 测试故事文件前缀

DEFAULT_LOG_LEVEL = "INFO"  # 默认日志级别
ENV_LOG_LEVEL = "LOG_LEVEL"  # 环境变量中的日志级别键名
TCP_PROTOCOL = "TCP"  # TCP 协议标识符

# =============================================================================
# 发送者和响应相关常量
# =============================================================================

DEFAULT_SENDER_ID = "default"  # 默认发送者 ID
UTTER_PREFIX = "utter_"  # 响应动作名称前缀

# =============================================================================
# 助手和配置管理常量
# =============================================================================

ASSISTANT_ID_KEY = "assistant_id"  # 助手 ID 配置键名
ASSISTANT_ID_DEFAULT_VALUE = "placeholder_default"  # 助手 ID 默认值

# =============================================================================
# 配置键定义常量 - 定义各种配置键的名称和分类
# =============================================================================

CONFIG_MANDATORY_COMMON_KEYS = [ASSISTANT_ID_KEY]  # 所有配置都必须包含的通用键
CONFIG_AUTOCONFIGURABLE_KEYS_CORE = ["policies"]  # Core 模块可自动配置的键
CONFIG_AUTOCONFIGURABLE_KEYS_NLU = ["pipeline"]  # NLU 模块可自动配置的键
CONFIG_AUTOCONFIGURABLE_KEYS = (
    CONFIG_AUTOCONFIGURABLE_KEYS_CORE + CONFIG_AUTOCONFIGURABLE_KEYS_NLU  # 所有可自动配置的键
)

# 核心和 NLU 配置键定义
CONFIG_KEYS_CORE = ["policies"] + CONFIG_MANDATORY_COMMON_KEYS  # Core 模块配置键
CONFIG_KEYS_NLU = ["language", "pipeline"] + CONFIG_MANDATORY_COMMON_KEYS  # NLU 模块配置键
CONFIG_KEYS = CONFIG_KEYS_CORE + CONFIG_KEYS_NLU  # 所有配置键

# 强制配置键定义
CONFIG_MANDATORY_KEYS_CORE: List[Text] = [] + CONFIG_MANDATORY_COMMON_KEYS  # Core 模块强制配置键
CONFIG_MANDATORY_KEYS_NLU = ["language"] + CONFIG_MANDATORY_COMMON_KEYS  # NLU 模块强制配置键
CONFIG_MANDATORY_KEYS = CONFIG_MANDATORY_KEYS_CORE + CONFIG_MANDATORY_KEYS_NLU  # 所有强制配置键

# =============================================================================
# 表单相关常量（在领域定义中使用）
# =============================================================================

REQUIRED_SLOTS_KEY = "required_slots"  # 必需槽位键名
IGNORED_INTENTS = "ignored_intents"  # 忽略的意图键名

# =============================================================================
# 默认 Rasa 开源项目布局常量 - 定义标准项目目录结构
# =============================================================================

# 配置文件路径
DEFAULT_ENDPOINTS_PATH = "endpoints.yml"  # 端点配置文件路径
DEFAULT_CREDENTIALS_PATH = "credentials.yml"  # 凭据配置文件路径
DEFAULT_CONFIG_PATH = "config.yml"  # 主配置文件路径
DEFAULT_DOMAIN_PATH = "domain.yml"  # 领域定义文件路径

# 目录路径
DEFAULT_ACTIONS_PATH = "actions"  # 自定义动作目录路径
DEFAULT_MODELS_PATH = "models"  # 模型文件目录路径
DEFAULT_CONVERTED_DATA_PATH = "converted_data"  # 转换后数据目录路径
DEFAULT_DATA_PATH = "data"  # 训练数据目录路径
DEFAULT_RESULTS_PATH = "results"  # 结果输出目录路径
DEFAULT_NLU_RESULTS_PATH = "nlu_comparison_results"  # NLU 比较结果目录路径

# 子目录名称
DEFAULT_CORE_SUBDIRECTORY_NAME = "core"  # Core 子目录名称
DEFAULT_NLU_SUBDIRECTORY_NAME = "nlu"  # NLU 子目录名称
DEFAULT_CONVERSATION_TEST_PATH = "tests"  # 对话测试目录路径

# 标记器相关路径
DEFAULT_MARKERS_PATH = "markers"  # 标记器目录路径
DEFAULT_MARKERS_CONFIG_PATH = "markers/config"  # 标记器配置目录路径
DEFAULT_MARKERS_OUTPUT_PATH = "markers/output"  # 标记器输出目录路径
DEFAULT_MARKERS_STATS_PATH = "markers/stats"  # 标记器统计目录路径

# =============================================================================
# 其他常量
# =============================================================================

DIAGNOSTIC_DATA = "diagnostic_data"  # 诊断数据标识符

RESPONSE_CONDITION = "condition"  # 响应条件键名
CHANNEL = "channel"  # 通道键名
