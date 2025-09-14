# 导入深拷贝模块
import copy
# 导入日志记录模块
import logging

# 导入对话状态跟踪器
from rasa.shared.core.trackers import DialogueStateTracker
# 导入类型提示相关类型
from typing import Text, Any, Dict, Optional, List

# 导入Rasa NLG插值器模块
from rasa.core.nlg import interpolator
# 导入自然语言生成器和响应变体过滤器
from rasa.core.nlg.generator import NaturalLanguageGenerator, ResponseVariationFilter
# 导入响应条件常量
from rasa.shared.constants import RESPONSE_CONDITION

# 创建日志记录器
logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    """基于响应的自然语言生成器。

    响应可以使用变量来根据对话状态自定义话语。
    """

    def __init__(self, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        """创建模板自然语言生成器。

        Args:
            responses: 用于生成消息的响应字典
        """
        # 存储响应字典
        self.responses = responses

    # 忽略未使用局部变量警告
    # noinspection PyUnusedLocal
    def _random_response_for(
        self, utter_action: Text, output_channel: Text, filled_slots: Dict[Text, Any]
    ) -> Optional[Dict[Text, Any]]:
        """从可用响应中为utter动作选择随机响应。

        如果为当前输出通道提供了特定于通道的响应，
        则只从特定于通道的响应中选择。
        
        Args:
            utter_action: utter动作名称
            output_channel: 输出通道名称
            filled_slots: 已填充的槽位字典
            
        Returns:
            选中的响应字典，如果没有合适的响应则返回None
        """
        # 导入numpy用于随机选择
        import numpy as np

        # 检查utter动作是否存在于响应字典中
        if utter_action in self.responses:
            # 创建响应变体过滤器
            response_filter = ResponseVariationFilter(self.responses)
            # 获取适合的响应列表
            suitable_responses = response_filter.responses_for_utter_action(
                utter_action, output_channel, filled_slots
            )

            if suitable_responses:
                # 从适合的响应中随机选择一个
                selected_response = np.random.choice(suitable_responses)
                # 获取响应条件
                condition = selected_response.get(RESPONSE_CONDITION)
                if condition:
                    # 格式化响应条件用于日志记录
                    formatted_response_conditions = self._format_response_conditions(
                        condition
                    )
                    logger.debug(
                        "选择具有条件的响应变体:"
                        f"{formatted_response_conditions}"
                    )
                return selected_response
            else:
                return None
        else:
            return None

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """为请求的utter动作生成响应。
        
        Args:
            utter_action: utter动作名称
            tracker: 对话状态跟踪器
            output_channel: 输出通道名称
            **kwargs: 其他关键字参数
            
        Returns:
            生成的响应字典，如果无法生成则返回None
        """
        # 获取当前槽位值
        filled_slots = tracker.current_slot_values()
        # 基于槽位值生成响应
        return self.generate_from_slots(
            utter_action, filled_slots, output_channel, **kwargs
        )

    def generate_from_slots(
        self,
        utter_action: Text,
        filled_slots: Dict[Text, Any],
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """为请求的utter动作生成响应。
        
        Args:
            utter_action: utter动作名称
            filled_slots: 已填充的槽位字典
            output_channel: 输出通道名称
            **kwargs: 其他关键字参数
            
        Returns:
            生成的响应字典，如果无法生成则返回None
        """
        # 为传递的utter动作获取随机响应
        r = copy.deepcopy(
            self._random_response_for(utter_action, output_channel, filled_slots)
        )
        # 用占位符填充响应中的槽位并返回响应
        if r is not None:
            return self._fill_response(r, filled_slots, **kwargs)
        else:
            return None

    def _fill_response(
        self,
        response: Dict[Text, Any],
        filled_slots: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """结合槽位值和关键字参数来填充响应。
        
        Args:
            response: 要填充的响应字典
            filled_slots: 已填充的槽位字典
            **kwargs: 其他关键字参数
            
        Returns:
            填充后的响应字典
        """
        # 获取响应变量中的槽位值
        response_vars = self._response_variables(filled_slots, kwargs)

        # 需要插值的键列表
        keys_to_interpolate = [
            "text",           # 文本内容
            "image",          # 图片
            "custom",         # 自定义内容
            "buttons",        # 按钮
            "attachment",     # 附件
            "quick_replies",  # 快速回复
        ]
        # 如果有响应变量，则对指定键进行插值
        if response_vars:
            for key in keys_to_interpolate:
                if key in response:
                    response[key] = interpolator.interpolate(
                        response[key], response_vars
                    )
        return response

    @staticmethod
    def _response_variables(
        filled_slots: Dict[Text, Any], kwargs: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        """结合槽位值和关键字参数来填充响应。
        
        Args:
            filled_slots: 已填充的槽位字典
            kwargs: 关键字参数字典
            
        Returns:
            合并后的响应变量字典
        """
        # 如果槽位字典为None，则初始化为空字典
        if filled_slots is None:
            filled_slots = {}

        # 将已填充的槽位复制到响应变量中
        response_vars = filled_slots.copy()
        # 更新关键字参数
        response_vars.update(kwargs)
        return response_vars

    @staticmethod
    def _format_response_conditions(response_conditions: List[Dict[Text, Any]]) -> Text:
        """格式化响应条件用于日志记录。
        
        Args:
            response_conditions: 响应条件列表
            
        Returns:
            格式化后的条件字符串
        """
        # 初始化格式化条件列表
        formatted_response_conditions = [""]
        # 遍历每个条件
        for index, condition in enumerate(response_conditions):
            # 构建约束条件列表
            constraints = []
            constraints.append(f"type: {str(condition['type'])}")
            constraints.append(f"name: {str(condition['name'])}")
            constraints.append(f"value: {str(condition['value'])}")

            # 用分隔符连接约束条件
            condition_message = " | ".join(constraints)
            # 格式化条件消息
            formatted_condition = f"[condition {str(index + 1)}] {condition_message}"
            formatted_response_conditions.append(formatted_condition)

        # 用换行符连接所有格式化条件
        return "\n".join(formatted_response_conditions)
