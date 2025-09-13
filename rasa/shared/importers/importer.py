from abc import ABC, abstractmethod
from functools import reduce
from typing import Text, Optional, List, Dict, Set, Any, Tuple, Type, Union, cast
import logging

import rasa.shared.constants
import rasa.shared.utils.common
import rasa.shared.core.constants
import rasa.shared.utils.io
from rasa.shared.core.domain import (
    Domain,
    KEY_E2E_ACTIONS,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_ACTIONS,
)
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import ENTITIES, ACTION_NAME
from rasa.shared.core.domain import IS_RETRIEVAL_INTENT_KEY

# =============================================================================
# 训练数据导入器模块 - 提供多种训练数据导入机制的统一接口
# =============================================================================

logger = logging.getLogger(__name__)


class TrainingDataImporter(ABC):
    """训练数据导入器的通用接口。
    
    定义了不同机制加载训练数据的统一接口，
    包括域、故事、配置和 NLU 数据的加载方法。
    """

    @abstractmethod
    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化导入器。
        
        Args:
            config_file: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表或单个路径
            **kwargs: 其他关键字参数
        """
        ...

    @abstractmethod
    def get_domain(self) -> Domain:
        """检索机器人的域。

        Returns:
            加载的 `Domain` 对象
        """
        ...

    @abstractmethod
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """检索用于训练的故事。

        Args:
            exclusion_percentage: 应排除的训练数据百分比

        Returns:
            包含所有加载故事的 `StoryGraph` 对象
        """
        ...

    def get_conversation_tests(self) -> StoryGraph:
        """检索用于测试的端到端对话故事。

        Returns:
            包含所有加载故事的 `StoryGraph` 对象
        """
        return self.get_stories()

    @abstractmethod
    def get_config(self) -> Dict:
        """检索用于训练的配置。

        Returns:
            配置字典
        """
        ...

    @abstractmethod
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """返回自动配置的配置文件路径（仅当存在单个配置文件时）。
        
        Returns:
            配置文件路径，如果不存在或存在多个则返回 None
        """
        ...

    @abstractmethod
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """检索用于训练的 NLU 训练数据。

        Args:
            language: 可用于仅加载特定语言的训练数据

        Returns:
            加载的 NLU `TrainingData` 对象
        """
        ...

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """从配置文件加载 `TrainingDataImporter` 实例。
        
        Args:
            config_path: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表
            args: 额外参数
            
        Returns:
            配置的 `TrainingDataImporter` 实例
        """
        config = rasa.shared.utils.io.read_config_file(config_path)
        return TrainingDataImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths, args
        )

    @staticmethod
    def load_core_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """加载核心 `TrainingDataImporter` 实例。

        从配置文件加载的实例将仅读取 Core 训练数据。
        
        Args:
            config_path: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表
            args: 额外参数
            
        Returns:
            配置的 `TrainingDataImporter` 实例
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )
        return importer

    @staticmethod
    def load_nlu_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """加载 NLU `TrainingDataImporter` 实例。

        从配置文件加载的实例将仅读取 NLU 训练数据。
        
        Args:
            config_path: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表
            args: 额外参数
            
        Returns:
            配置的 `TrainingDataImporter` 实例
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )

        if isinstance(importer, E2EImporter):
            # 当我们仅训练 NLU 时，不需要用 Core 训练数据中的 E2E 数据来丰富数据
            importer = importer.importer

        return NluDataImporter(importer)

    @staticmethod
    def load_from_dict(
        config: Optional[Dict] = None,
        config_path: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """从字典加载 `TrainingDataImporter` 实例。
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表
            args: 额外参数
            
        Returns:
            配置的 `TrainingDataImporter` 实例
        """
        from rasa.shared.importers.rasa import RasaFileImporter

        config = config or {}
        importers = config.get("importers", [])
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths, args
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]
        if not importers:
            importers = [
                RasaFileImporter(config_path, domain_path, training_data_paths)
            ]

        return E2EImporter(ResponsesSyncImporter(CombinedDataImporter(importers)))

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> Optional["TrainingDataImporter"]:
        """从字典配置创建导入器实例。
        
        Args:
            importer_config: 导入器配置字典
            config_path: 配置文件路径
            domain_path: 域文件路径
            training_data_paths: 训练数据路径列表
            args: 额外参数
            
        Returns:
            配置的导入器实例，如果创建失败则返回 None
        """
        from rasa.shared.importers.multi_project import MultiProjectImporter
        from rasa.shared.importers.rasa import RasaFileImporter

        module_path = importer_config.pop("name", None)
        if module_path == RasaFileImporter.__name__:
            importer_class: Type[TrainingDataImporter] = RasaFileImporter
        elif module_path == MultiProjectImporter.__name__:
            importer_class = MultiProjectImporter
        else:
            try:
                importer_class = rasa.shared.utils.common.class_from_module_path(
                    module_path
                )
            except (AttributeError, ImportError):
                logging.warning(f"导入器 '{module_path}' 未找到。")
                return None

        constructor_arguments = rasa.shared.utils.common.minimal_kwargs(
            {**importer_config, **(args or {})}, importer_class
        )

        return importer_class(
            config_path,
            domain_path,
            training_data_paths,
            **constructor_arguments,
        )

    def fingerprint(self) -> Text:
        """返回随机指纹，因为数据不应被缓存。
        
        Returns:
            25 字符的随机字符串
        """
        return rasa.shared.utils.io.random_string(25)

    def __repr__(self) -> Text:
        """返回对象的文本表示。
        
        Returns:
            类名
        """
        return self.__class__.__name__


class NluDataImporter(TrainingDataImporter):
    """跳过任何 Core 相关文件读取的导入器。
    
    此类专门用于 NLU 训练，不加载 Core 相关的数据。
    """

    def __init__(self, actual_importer: TrainingDataImporter):
        """初始化 NLUDataImporter。
        
        Args:
            actual_importer: 实际的导入器实例
        """
        self._importer = actual_importer

    def get_domain(self) -> Domain:
        """检索模型域（参见父类完整文档字符串）。
        
        Returns:
            空域对象
        """
        return Domain.empty()

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """检索训练故事/规则（参见父类完整文档字符串）。
        
        Returns:
            空故事图
        """
        return StoryGraph([])

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return StoryGraph([])

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        return self._importer.get_nlu_data(language)

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()


class CombinedDataImporter(TrainingDataImporter):
    """组合多个导入器的 `TrainingDataImporter`。

    使用多个 `TrainingDataImporter` 实例来加载数据，
    就像它们是单个实例一样。
    """

    def __init__(self, importers: List[TrainingDataImporter]):
        """初始化组合数据导入器。
        
        Args:
            importers: 导入器列表
        """
        self._importers = importers

    @rasa.shared.utils.common.cached_method
    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        configs = [importer.get_config() for importer in self._importers]

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        domains = [importer.get_domain() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other),
            domains,
            Domain.empty(),
        )

    @rasa.shared.utils.common.cached_method
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        stories = [
            importer.get_stories(exclusion_percentage) for importer in self._importers
        ]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @rasa.shared.utils.common.cached_method
    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        stories = [importer.get_conversation_tests() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        nlu_data = [importer.get_nlu_data(language) for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), nlu_data, TrainingData()
        )

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        if len(self._importers) != 1:
            rasa.shared.utils.io.raise_warning(
                "Auto-config for multiple importers is not supported; "
                "using config as is."
            )
            return None
        return self._importers[0].get_config_file_for_auto_config()


class ResponsesSyncImporter(TrainingDataImporter):
    """在域和 NLU 训练数据之间同步 `responses` 的导入器。

    在域和 NLU 之间同步响应，并将 NLU 训练数据中的
    检索意图属性添加回域。
    """

    def __init__(self, importer: TrainingDataImporter):
        """初始化响应同步导入器。
        
        Args:
            importer: 底层导入器
        """
        self._importer = importer

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of retrieval intents in NLU data."""
        existing_domain = self._importer.get_domain()
        existing_nlu_data = self._importer.get_nlu_data()

        # Merge responses from NLU data with responses in the domain.
        # If NLU data has any retrieval intents, then add corresponding
        # retrieval actions with `utter_` prefix automatically to the
        # final domain, update the properties of existing retrieval intents.
        domain_with_retrieval_intents = self._get_domain_with_retrieval_intents(
            existing_nlu_data.retrieval_intents,
            existing_nlu_data.responses,
            existing_domain,
        )

        existing_domain = existing_domain.merge(
            domain_with_retrieval_intents, override=True
        )
        existing_domain.check_missing_responses()

        return existing_domain

    @staticmethod
    def _construct_retrieval_action_names(retrieval_intents: Set[Text]) -> List[Text]:
        """Lists names of all retrieval actions related to passed retrieval intents.

        Args:
            retrieval_intents: List of retrieval intents defined in the NLU training
                data.

        Returns: Names of corresponding retrieval actions
        """
        return [
            f"{rasa.shared.constants.UTTER_PREFIX}{intent}"
            for intent in retrieval_intents
        ]

    @staticmethod
    def _get_domain_with_retrieval_intents(
        retrieval_intents: Set[Text],
        responses: Dict[Text, List[Dict[Text, Any]]],
        existing_domain: Domain,
    ) -> Domain:
        """Construct a domain consisting of retrieval intents.

         The result domain will have retrieval intents that are listed
         in the NLU training data.

        Args:
            retrieval_intents: Set of retrieval intents defined in NLU training data.
            responses: Responses defined in NLU training data.
            existing_domain: Domain which is already loaded from the domain file.

        Returns: Domain with retrieval actions added to action names and properties
          for retrieval intents updated.
        """
        # Get all the properties already defined
        # for each retrieval intent in other domains
        # and add the retrieval intent property to them
        retrieval_intent_properties = []
        for intent in retrieval_intents:
            intent_properties = (
                existing_domain.intent_properties[intent]
                if intent in existing_domain.intent_properties
                else {}
            )
            intent_properties[IS_RETRIEVAL_INTENT_KEY] = True
            retrieval_intent_properties.append({intent: intent_properties})

        action_names = ResponsesSyncImporter._construct_retrieval_action_names(
            retrieval_intents
        )

        return Domain.from_dict(
            {
                KEY_INTENTS: retrieval_intent_properties,
                KEY_RESPONSES: responses,
                KEY_ACTIONS: action_names,
            }
        )

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return self._importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return self._importer.get_conversation_tests()

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Updates NLU data with responses for retrieval intents from domain."""
        existing_nlu_data = self._importer.get_nlu_data(language)
        existing_domain = self._importer.get_domain()

        return existing_nlu_data.merge(
            self._get_nlu_data_with_responses(
                existing_domain.retrieval_intent_responses
            )
        )

    @staticmethod
    def _get_nlu_data_with_responses(
        responses: Dict[Text, List[Dict[Text, Any]]]
    ) -> TrainingData:
        """Construct training data object with only the responses supplied.

        Args:
            responses: Responses the NLU data should
            be initialized with.

        Returns: TrainingData object with responses.

        """
        return TrainingData(responses=responses)


class E2EImporter(TrainingDataImporter):
    """具有以下功能的导入器：

    - 用故事中的动作/用户消息增强 NLU 训练数据
    - 将故事中潜在的端到端机器人消息作为动作添加到域中
    """

    def __init__(self, importer: TrainingDataImporter) -> None:
        """初始化 E2E 导入器。
        
        Args:
            importer: 底层导入器
        """
        self.importer = importer

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        original = self.importer.get_domain()
        e2e_domain = self._get_domain_with_e2e_actions()

        return original.merge(e2e_domain)

    def _get_domain_with_e2e_actions(self) -> Domain:

        stories = self.get_stories()

        additional_e2e_action_names = set()
        for story_step in stories.story_steps:
            additional_e2e_action_names.update(
                {
                    event.action_text
                    for event in story_step.events
                    if isinstance(event, ActionExecuted) and event.action_text
                }
            )

        return Domain.from_dict({KEY_E2E_ACTIONS: list(additional_e2e_action_names)})

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves the stories that should be used for training.

        See parent class for details.
        """
        return self.importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return self.importer.get_conversation_tests()

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self.importer.get_config()

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self.importer.get_config_file_for_auto_config()

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        training_datasets = [
            _additional_training_data_from_default_actions(),
            self.importer.get_nlu_data(language),
            self._additional_training_data_from_stories(),
        ]

        return reduce(
            lambda merged, other: merged.merge(other), training_datasets, TrainingData()
        )

    def _additional_training_data_from_stories(self) -> TrainingData:
        stories = self.get_stories()

        utterances, actions = _unique_events_from_stories(stories)

        # Sort events to guarantee deterministic behavior and to avoid that the NLU
        # model has to be retrained due to changes in the event order within
        # the stories.
        sorted_utterances = sorted(
            utterances, key=lambda user: user.intent_name or user.text or ""
        )
        sorted_actions = sorted(
            actions, key=lambda action: action.action_name or action.action_text or ""
        )

        additional_messages_from_stories = [
            _messages_from_action(action) for action in sorted_actions
        ] + [_messages_from_user_utterance(user) for user in sorted_utterances]

        logger.debug(
            f"Added {len(additional_messages_from_stories)} training data examples "
            f"from the story training data."
        )
        return TrainingData(additional_messages_from_stories)


def _unique_events_from_stories(
    stories: StoryGraph,
) -> Tuple[Set[UserUttered], Set[ActionExecuted]]:
    action_events = set()
    user_events = set()

    for story_step in stories.story_steps:
        for event in story_step.events:
            if isinstance(event, ActionExecuted):
                action_events.add(event)
            elif isinstance(event, UserUttered):
                user_events.add(event)

    return user_events, action_events


def _messages_from_user_utterance(event: UserUttered) -> Message:
    # sub state correctly encodes intent vs text
    data = cast(Dict[Text, Any], event.as_sub_state())
    # sub state stores entities differently
    if data.get(ENTITIES) and event.entities:
        data[ENTITIES] = event.entities

    return Message(data=data)


def _messages_from_action(event: ActionExecuted) -> Message:
    # sub state correctly encodes action_name vs action_text
    return Message(data=event.as_sub_state())


def _additional_training_data_from_default_actions() -> TrainingData:
    additional_messages_from_default_actions = [
        Message(data={ACTION_NAME: action_name})
        for action_name in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
    ]

    return TrainingData(additional_messages_from_default_actions)
