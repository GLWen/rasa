# 导入日志记录模块
import logging
# 导入类型提示相关类型
from typing import List, Optional, Union, Text, Any, Dict

# 导入Rasa共享工具模块
import rasa.shared.utils.common
import rasa.shared.utils.io
# 导入Rasa共享常量
from rasa.shared.constants import CHANNEL, RESPONSE_CONDITION
# 导入Rasa核心领域模块
from rasa.shared.core.domain import Domain
# 导入端点配置类
from rasa.utils.endpoints import EndpointConfig
# 导入对话状态跟踪器
from rasa.shared.core.trackers import DialogueStateTracker

# 创建日志记录器
logger = logging.getLogger(__name__)


class NaturalLanguageGenerator:
    """基于对话状态生成机器人话语的自然语言生成器基类。"""

    async def generate(
        self,
        utter_action: Text,
        tracker: "DialogueStateTracker",
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """为请求的utter动作生成响应。

        有很多不同的方法来实现此功能，例如，
        生成可以基于响应，或者通过将对话状态输入到机器学习NLG模型中来完全基于ML。

        Args:
            utter_action: utter动作名称
            tracker: 对话状态跟踪器
            output_channel: 输出通道名称
            **kwargs: 其他关键字参数

        Returns:
            生成的响应字典，如果无法生成则返回None
        """
        raise NotImplementedError

    @staticmethod
    def create(
        obj: Union["NaturalLanguageGenerator", EndpointConfig, None],
        domain: Optional[Domain],
    ) -> "NaturalLanguageGenerator":
        """创建生成器的工厂方法。
        
        Args:
            obj: 自然语言生成器实例或端点配置
            domain: 领域对象
            
        Returns:
            自然语言生成器实例
        """
        # 如果已经是生成器实例，直接返回
        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        else:
            # 从端点配置创建生成器
            return _create_from_endpoint_config(obj, domain)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None, domain: Optional[Domain] = None
) -> "NaturalLanguageGenerator":
    """根据端点配置创建适当的NLG对象。
    
    Args:
        endpoint_config: 端点配置对象
        domain: 领域对象
        
    Returns:
        自然语言生成器实例
    """
    # 如果没有提供领域对象，则创建空领域
    domain = domain or Domain.empty()

    if endpoint_config is None:
        # 导入模板自然语言生成器
        from rasa.core.nlg import TemplatedNaturalLanguageGenerator

        # 如果没有设置端点配置，这是默认类型
        nlg: "NaturalLanguageGenerator" = TemplatedNaturalLanguageGenerator(
            domain.responses
        )
    elif endpoint_config.type is None or endpoint_config.type.lower() == "callback":
        # 导入回调自然语言生成器
        from rasa.core.nlg import CallbackNaturalLanguageGenerator

        # 如果没有设置nlg类型，这是默认类型
        nlg = CallbackNaturalLanguageGenerator(endpoint_config=endpoint_config)
    elif endpoint_config.type.lower() == "response":
        # 导入模板自然语言生成器
        from rasa.core.nlg import TemplatedNaturalLanguageGenerator

        nlg = TemplatedNaturalLanguageGenerator(domain.responses)
    else:
        # 从模块名称加载自定义生成器
        nlg = _load_from_module_name_in_endpoint_config(endpoint_config, domain)

    # 记录实例化的NLG类型
    logger.debug(f"实例化NLG为 '{nlg.__class__.__name__}'.")
    return nlg


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig, domain: Domain
) -> "NaturalLanguageGenerator":
    """初始化自定义自然语言生成器。

    Args:
        domain: 定义助手运行的领域
        endpoint_config: 特定的自然语言生成器配置
        
    Returns:
        自定义自然语言生成器实例
        
    Raises:
        Exception: 当无法找到或导入指定类时抛出异常
    """
    try:
        # 从模块路径获取类
        nlg_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        # 创建生成器实例
        return nlg_class(endpoint_config=endpoint_config, domain=domain)
    except (AttributeError, ImportError) as e:
        # 如果无法找到类，抛出异常
        raise Exception(
            f"无法基于模块路径找到类 "
            f"'{endpoint_config.type}'. 创建 "
            f"`NaturalLanguageGenerator` 实例失败。错误: {e}"
        )


class ResponseVariationFilter:
    """基于通道、动作和条件过滤响应变体的过滤器。"""

    def __init__(self, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        """初始化响应变体过滤器。
        
        Args:
            responses: 响应字典，包含utter动作和对应的响应变体列表
        """
        # 存储响应字典
        self.responses = responses

    @staticmethod
    def _matches_filled_slots(
        filled_slots: Dict[Text, Any], response: Dict[Text, Any]
    ) -> bool:
        """检查条件响应变体是否与已填充的槽位匹配。
        
        Args:
            filled_slots: 已填充的槽位字典
            response: 响应字典
            
        Returns:
            如果匹配则返回True，否则返回False
        """
        # 获取响应条件约束
        constraints = response.get(RESPONSE_CONDITION, [])
        # 遍历每个约束条件
        for constraint in constraints:
            name = constraint["name"]
            value = constraint["value"]
            filled_slots_value = filled_slots.get(name)
            # 如果槽位值和约束值都是字符串，进行大小写不敏感比较
            if isinstance(filled_slots_value, str) and isinstance(value, str):
                if filled_slots_value.casefold() != value.casefold():
                    return False
            # 槽位值可以是不同的数据类型
            # 如int、float、bool等，因此当槽位值不是字符串时执行此检查
            elif filled_slots_value != value:
                return False

        return True

    def responses_for_utter_action(
        self,
        utter_action: Text,
        output_channel: Text,
        filled_slots: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """返回适合通道、动作和条件的响应数组。
        
        Args:
            utter_action: utter动作名称
            output_channel: 输出通道名称
            filled_slots: 已填充的槽位字典
            
        Returns:
            适合的响应列表
        """
        # 过滤没有条件的响应
        default_responses = list(
            filter(
                lambda x: (x.get(RESPONSE_CONDITION) is None),
                self.responses[utter_action],
            )
        )
        # 过滤有条件且与已填充槽位匹配的响应
        conditional_responses = list(
            filter(
                lambda x: (
                    x.get(RESPONSE_CONDITION)
                    and self._matches_filled_slots(
                        filled_slots=filled_slots, response=x
                    )
                ),
                self.responses[utter_action],
            )
        )

        # 过滤匹配通道的条件响应
        conditional_channel = list(
            filter(lambda x: (x.get(CHANNEL) == output_channel), conditional_responses)
        )
        # 过滤不匹配通道的条件响应
        conditional_no_channel = list(
            filter(lambda x: (x.get(CHANNEL) is None), conditional_responses)
        )
        # 过滤匹配通道的默认响应
        default_channel = list(
            filter(lambda x: (x.get(CHANNEL) == output_channel), default_responses)
        )
        # 过滤不匹配通道的默认响应
        default_no_channel = list(
            filter(lambda x: (x.get(CHANNEL) is None), default_responses)
        )

        # 按优先级返回响应
        if conditional_channel:
            return conditional_channel

        if default_channel:
            return default_channel

        if conditional_no_channel:
            return conditional_no_channel

        return default_no_channel

    def get_response_variation_id(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
    ) -> Optional[Text]:
        """返回第一个匹配的响应变体ID。

        此ID对应于适合通道、动作和条件的响应变体。
        
        Args:
            utter_action: utter动作名称
            tracker: 对话状态跟踪器
            output_channel: 输出通道名称
            
        Returns:
            响应变体ID，如果没有找到则返回None
        """
        # 获取当前槽位值
        filled_slots = tracker.current_slot_values()
        # 检查utter动作是否存在于响应字典中
        if utter_action in self.responses:
            # 获取符合条件的响应变体
            eligible_variations = self.responses_for_utter_action(
                utter_action, output_channel, filled_slots
            )
            # 验证响应ID是否有效
            response_ids_are_valid = self._validate_response_ids(eligible_variations)

            # 如果有符合条件的变体且ID有效，返回第一个变体的ID
            if eligible_variations and response_ids_are_valid:
                return eligible_variations[0].get("id")

        return None

    @staticmethod
    def _validate_response_ids(response_variations: List[Dict[Text, Any]]) -> bool:
        """检查特定utter_action的响应ID是否唯一。

        Args:
            response_variations: 要验证的响应变体列表

        Returns:
            如果响应ID唯一则返回True，否则返回False
        """
        # 创建响应ID集合用于检查重复
        response_ids = set()
        # 遍历每个响应变体
        for response_variation in response_variations:
            response_variation_id = response_variation.get("id")
            # 如果ID已存在，发出警告并返回False
            if response_variation_id and response_variation_id in response_ids:
                rasa.shared.utils.io.raise_warning(
                    f"在领域中发现重复的响应ID '{response_variation_id}' "
                    f"定义。"
                )
                return False

            # 将ID添加到集合中
            response_ids.add(response_variation_id)

        return True
