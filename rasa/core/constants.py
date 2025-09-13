# =============================================================================
# Rasa Core 常量定义文件
# 本文件定义了 Rasa 核心模块中使用的所有常量配置
# =============================================================================

# 服务器配置常量
# =============================================================================

# 默认服务器端口号，Rasa 服务器默认监听 5005 端口
DEFAULT_SERVER_PORT = 5005

# 默认服务器接口地址，0.0.0.0 表示监听所有网络接口
DEFAULT_SERVER_INTERFACE = "0.0.0.0"

# 默认服务器 URL 格式模板，用于构建完整的服务器地址
DEFAULT_SERVER_FORMAT = "{}://localhost:{}"

# 默认服务器完整 URL，使用 HTTP 协议和默认端口
DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format("http", DEFAULT_SERVER_PORT)

# 默认交互式服务器 URL 格式模板，用于交互式学习模式
DEFAULT_INTERACTIVE_SERVER_URL = "{}://localhost:{}"

# NLU (自然语言理解) 配置常量
# =============================================================================

# NLU 回退阈值，当意图预测置信度低于此值时触发回退处理
DEFAULT_NLU_FALLBACK_THRESHOLD = 0.3

# NLU 回退歧义阈值，用于处理多个意图预测结果过于接近的情况
DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD = 0.1

# Core 回退阈值，当动作预测置信度低于此值时触发回退处理
DEFAULT_CORE_FALLBACK_THRESHOLD = 0.3

# 默认最大历史记录长度，None 表示核心策略历史记录默认无限制
DEFAULT_MAX_HISTORY = None  # 核心策略历史记录默认无限制

# 超时配置常量
# =============================================================================

# 默认响应超时时间，设置为 1 小时（3600 秒）
DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 小时

# 默认请求超时时间，设置为 5 分钟（300 秒）
DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 分钟

# 默认流读取超时时间，设置为 10 秒
DEFAULT_STREAM_READING_TIMEOUT = 10  # 秒

# 默认锁生命周期，设置为 60 秒
DEFAULT_LOCK_LIFETIME = 60  # 秒

# 默认保持连接超时时间，设置为 120 秒
DEFAULT_KEEP_ALIVE_TIMEOUT = 120  # 秒

# 认证配置常量
# =============================================================================

# Bearer Token 前缀，用于 HTTP 认证头
BEARER_TOKEN_PREFIX = "Bearer "

# 策略优先级配置常量
# =============================================================================

# 默认策略优先级，最低优先级，主要用于机器学习策略
DEFAULT_POLICY_PRIORITY = 1

# 意图预测策略优先级
# 此优先级应低于所有基于规则的策略，但高于基于机器学习的策略
# 这启用了集成内部的循环，如果基于规则的策略都没有预测动作，
# 而意图预测策略预测了一个，则集成选择其预测，然后再次运行
# 基于机器学习的策略以获得实际动作的预测。为了防止无限循环，
# 意图预测策略只有在跟踪器中的最后一个事件是 `UserUttered` 类型时
# 才预测动作。因此，它们在每个对话轮次中最多进行一次动作预测。
# 这允许其他策略预测获胜的动作预测。
UNLIKELY_INTENT_POLICY_PRIORITY = DEFAULT_POLICY_PRIORITY + 1

# 记忆化策略优先级
# 此优先级高于默认优先级，以优先处理训练故事
MEMOIZATION_POLICY_PRIORITY = UNLIKELY_INTENT_POLICY_PRIORITY + 1

# 规则策略优先级
# 规则策略的优先级高于所有其他策略，因为规则执行优先于训练故事或预测动作
RULE_POLICY_PRIORITY = MEMOIZATION_POLICY_PRIORITY + 1

# 对话相关常量
# =============================================================================

# 对话标识符，用于标识对话类型
DIALOGUE = "dialogue"

# 消息队列配置常量
# =============================================================================

# RabbitMQ 消息属性头名称，用于使用 `rasa export` 发布的事件
RASA_EXPORT_PROCESS_ID_HEADER_NAME = "rasa-export-process-id"

# 数据库配置常量
# =============================================================================

# PostgreSQL 模式环境变量名称，定义要访问的 PostgreSQL 模式
# 详见 https://www.postgresql.org/docs/9.1/ddl-schemas.html
POSTGRESQL_SCHEMA = "POSTGRESQL_SCHEMA"

# PostgreSQL 连接池大小环境变量名称
POSTGRESQL_POOL_SIZE = "SQL_POOL_SIZE"

# PostgreSQL 最大溢出连接数环境变量名称
POSTGRESQL_MAX_OVERFLOW = "SQL_MAX_OVERFLOW"

# 测试文件配置常量
# =============================================================================

# 混淆矩阵故事文件名称，用于存储故事测试的混淆矩阵图片
CONFUSION_MATRIX_STORIES_FILE = "story_confusion_matrix.png"

# 报告故事文件名称，用于存储故事测试报告
REPORT_STORIES_FILE = "story_report.json"

# 失败故事文件名称，用于存储测试失败的故事
FAILED_STORIES_FILE = "failed_test_stories.yml"

# 成功故事文件名称，用于存储测试成功的故事
SUCCESSFUL_STORIES_FILE = "successful_test_stories.yml"

# 警告故事文件名称，用于存储有警告的故事
STORIES_WITH_WARNINGS_FILE = "stories_with_warnings.yml"

# 策略配置键名常量
# =============================================================================

# 策略优先级配置键名
POLICY_PRIORITY = "priority"

# 策略特征化器配置键名
POLICY_FEATURIZER = "featurizer"

# 策略最大历史记录配置键名
POLICY_MAX_HISTORY = "max_history"

# 系统日志配置常量
# =============================================================================

# 默认协议类型，使用 UDP 协议
DEFAULT_PROTOCOL = "UDP"

# 默认系统日志主机地址
DEFAULT_SYSLOG_HOST = "localhost"

# 默认系统日志端口号
DEFAULT_SYSLOG_PORT = 514

# 动作服务器配置常量
# =============================================================================

# 压缩动作服务器请求环境变量名称
COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME = "COMPRESS_ACTION_SERVER_REQUEST"

# 默认是否压缩动作服务器请求，设置为 False（不压缩）
DEFAULT_COMPRESS_ACTION_SERVER_REQUEST = False
