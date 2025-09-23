# 启用类型注解的前向引用功能，允许在类型注解中使用字符串形式的类型
from __future__ import annotations

# 导入数据类相关功能
import dataclasses
# 导入抽象基类和抽象方法
from abc import ABC, abstractmethod
# 导入数据类装饰器和字段函数
from dataclasses import dataclass, field
# 导入日志记录模块
import logging
# 导入类型注解相关的类型
from typing import Any, Callable, Dict, List, Optional, Text, Type, Tuple, Union

# 导入图组件相关的异常类
from rasa.engine.exceptions import (
    GraphComponentException,  # 图组件异常
    GraphRunError,           # 图运行错误
    GraphSchemaException,    # 图模式异常
)
# 导入共享工具模块
import rasa.shared.utils.common
# 导入工具模块
import rasa.utils.common
# 导入资源类
from rasa.engine.storage.resource import Resource

# 导入模型存储接口
from rasa.engine.storage.storage import ModelStorage
# 导入Rasa异常类
from rasa.shared.exceptions import InvalidConfigException, RasaException
# 导入训练类型枚举
from rasa.shared.data import TrainingType

# 创建日志记录器
logger = logging.getLogger(__name__)


# 使用数据类装饰器定义模式节点类
@dataclass
class SchemaNode:
    """表示图模式中的一个节点。

    图模式节点定义了图中单个组件的配置和依赖关系。每个节点代表一个可执行的组件，
    它知道如何实例化自己、需要哪些输入、以及如何执行。

    Args:
        needs: 描述`fn`（或`constructor_name`如果`eager==False`）中的哪些参数
            由哪些父节点填充。这是一个从参数名到提供该参数的父节点名的映射。
        uses: 建模此特定图节点行为的类。这是实现GraphComponent接口的类。
        constructor_name: 应该用于实例化组件的构造函数的名称。如果`eager==False`，
            则`constructor`也可以指定由父节点填充的参数。例如，当父节点返回一个
            `Resource`而此节点希望直接从该资源加载自己时很有用。
        fn: 在图执行时应该在实例化的组件上调用的函数名称。来自`needs`的参数
            由父节点填充。
        config: 此图节点的用户配置。此配置不需要指定所有可能的参数；
            稍后将填充缺失参数的默认值。
        eager: 如果为`eager`，则在图运行之前实例化组件。否则在图运行时实例化（延迟）。
            通常在训练期间总是延迟实例化，在推理期间急切实例化（以避免第一次预测
            花费更长时间）。
        is_target: 如果为`True`，则此节点在指纹识别期间不能被修剪（尽管可能被
            缓存值替换）。例如，用于所有训练的组件，因为其结果总是需要添加到
            模型存档中，以便在推理期间数据可用。
        is_input: 具有`is_input`的节点总是运行（也在指纹运行期间）。这确保我们
            例如检测文件内容的变化。
        resource: 如果给定，则从现有资源加载图节点，而不是从头实例化。例如，
            用于加载训练好的组件进行预测。
    """

    # 定义节点需要的输入参数映射（参数名 -> 提供该参数的父节点名）
    needs: Dict[Text, Text]
    # 定义此节点使用的组件类（必须实现GraphComponent接口）
    uses: Type[GraphComponent]
    # 定义用于实例化组件的构造函数名称
    constructor_name: Text
    # 定义在图执行时要调用的函数名称
    fn: Text
    # 定义此图节点的用户配置
    config: Dict[Text, Any]
    # 是否急切实例化（默认False，即延迟实例化）
    eager: bool = False
    # 是否为目标节点（默认False）
    is_target: bool = False
    # 是否为输入节点（默认False）
    is_input: bool = False
    # 可选的资源对象，用于从现有资源加载组件
    resource: Optional[Resource] = None


# 使用数据类装饰器定义图模式类
@dataclass
class GraphSchema:
    """表示用于训练模型或进行预测的图。

    图模式定义了整个计算图的拓扑结构和配置。它包含所有节点及其依赖关系，
    可以用于序列化和反序列化，以及图的优化和修剪。
    """

    # 定义图中所有节点的映射（节点名 -> 节点对象）
    nodes: Dict[Text, SchemaNode]

    def as_dict(self) -> Dict[Text, Any]:
        """以可序列化格式返回图模式。

        将图模式转换为可以序列化为JSON或其他格式的字典格式。
        类对象不能直接序列化，所以需要将类转换为字符串路径。

        Returns:
            可以转储为JSON或其他格式的图模式格式。
        """
        # 创建可序列化的图模式字典
        serializable_graph_schema: Dict[Text, Dict[Text, Any]] = {"nodes": {}}
        # 遍历所有节点并序列化
        for node_name, node in self.nodes.items():
            # 将节点转换为字典
            serializable = dataclasses.asdict(node)

            # 类不能JSON序列化，需要转换为模块路径字符串
            serializable["uses"] = f"{node.uses.__module__}.{node.uses.__name__}"

            # 将序列化的节点添加到结果中
            serializable_graph_schema["nodes"][node_name] = serializable

        return serializable_graph_schema

    @classmethod
    def from_dict(cls, serialized_graph_schema: Dict[Text, Any]) -> GraphSchema:
        """Loads a graph schema which has been serialized using `schema.as_dict()`.

        Args:
            serialized_graph_schema: A serialized graph schema.

        Returns:
            A properly loaded schema.

        Raises:
            GraphSchemaException: In case the component class for a node couldn't be
                found.
        """
        nodes = {}
        for node_name, serialized_node in serialized_graph_schema["nodes"].items():
            try:
                serialized_node[
                    "uses"
                ] = rasa.shared.utils.common.class_from_module_path(
                    serialized_node["uses"]
                )

                resource = serialized_node["resource"]
                if resource:
                    serialized_node["resource"] = Resource(**resource)

            except ImportError as e:
                raise GraphSchemaException(
                    "Error deserializing graph schema. Can't "
                    "find class for graph component type "
                    f"'{serialized_node['uses']}'."
                ) from e

            nodes[node_name] = SchemaNode(**serialized_node)

        return GraphSchema(nodes)

    @property
    def target_names(self) -> List[Text]:
        """Returns the names of all target nodes."""
        return [node_name for node_name, node in self.nodes.items() if node.is_target]

    def minimal_graph_schema(self, targets: Optional[List[Text]] = None) -> GraphSchema:
        """Returns a new schema where all nodes are a descendant of a target."""
        dependencies = self._all_dependencies_schema(
            targets if targets else self.target_names
        )

        return GraphSchema(
            {
                node_name: node
                for node_name, node in self.nodes.items()
                if node_name in dependencies
            }
        )

    def _all_dependencies_schema(self, targets: List[Text]) -> List[Text]:
        required = []
        for target in targets:
            required.append(target)
            try:
                target_dependencies = self.nodes[target].needs.values()
            except KeyError:  # This can happen if the target is an input placeholder.
                continue
            for dependency in target_dependencies:
                required += self._all_dependencies_schema([dependency])

        return required


# 定义图组件的抽象基类
class GraphComponent(ABC):
    """将在图中运行的任何组件的接口。

    所有图组件都必须实现此接口。它定义了组件如何创建、加载、配置和运行。
    组件可以是训练器、预测器、特征提取器、分类器等。
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """应该在此组件之前包含在管道中的组件。

        返回此组件运行前需要的其他组件类型列表。
        用于依赖关系管理和管道构建。
        """
        return []

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """创建一个新的 `GraphComponent`。

        这是组件的工厂方法，用于创建新的组件实例。每个组件都必须实现此方法。

        Args:
            config: 此配置会覆盖 `default_config`。包含组件的所有配置参数。
            model_storage: 图组件可以用来持久化和加载自己的存储接口。
            resource: 此组件的资源定位器，可用于从 `model_storage` 持久化和加载自己。
            execution_context: 关于当前图运行的信息，包含图模式、模型ID等。

        Returns: 一个实例化的 `GraphComponent`。
        """
        ...

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """使用持久化版本创建组件。

        此方法用于从已保存的资源中加载组件。如果未重写，此方法仅调用 `create`。
        主要用于推理阶段，从训练好的模型中加载组件。

        Args:
            config: 此图组件的配置。这是组件的默认配置与用户指定配置的合并结果。
            model_storage: 图组件可以用来持久化和加载自己的存储接口。
            resource: 此组件的资源定位器，可用于从 `model_storage` 持久化和加载自己。
            execution_context: 关于当前图运行的信息。
            kwargs: 来自前一个节点的输出值可能作为 `kwargs` 传入。

        Returns:
            一个实例化的、已加载的 `GraphComponent`。
        """
        return cls.create(config, model_storage, resource, execution_context)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """返回组件的默认配置。

        默认配置和用户配置由 `GraphNode` 在将配置传递给组件的 `create` 和 `load` 方法之前合并。
        子类可以重写此方法以提供特定的默认配置。

        Returns:
            组件的默认配置字典。
        """
        return {}

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """确定此组件可以处理哪些语言。

        返回支持的语言列表，用于语言兼容性检查。返回 `None` 表示支持所有语言。

        Returns: 支持的语言列表，或 `None` 表示支持所有语言。
        """
        return None

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """确定此组件不能处理哪些语言。

        返回不支持的语言列表，用于语言兼容性检查。返回 `None` 表示支持所有语言。

        Returns: 不支持的语言列表，或 `None` 表示支持所有语言。
        """
        return None

    @staticmethod
    def required_packages() -> List[Text]:
        """此组件运行所需的额外Python依赖包。

        返回组件运行所需的Python包名称列表。用于依赖检查和安装验证。

        Returns: 必需的Python包名称列表。
        """
        return []

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """为指纹计算添加额外数据。

        如果组件使用图中未提供的外部数据，此方法很有用。返回的字符串将添加到
        组件的指纹计算中，用于缓存失效检测。

        Args:
            config: 组件的配置字典。

        Returns: 要添加到指纹中的额外数据字符串，或 `None` 表示无额外数据。
        """
        return None


# 定义图节点钩子的抽象基类
class GraphNodeHook(ABC):
    """保存要在 `GraphNode` 执行前后运行的功能。

    钩子机制允许在节点执行前后插入自定义逻辑，用于监控、调试、日志记录等目的。
    每个钩子都会在节点执行前调用 `on_before_node`，在节点执行后调用 `on_after_node`。
    """

    @abstractmethod
    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """在 `GraphNode` 执行之前运行。

        此方法在节点实际执行前被调用，可以用于：
        - 输入验证和预处理
        - 性能监控开始
        - 日志记录
        - 资源准备

        Args:
            node_name: 正在运行的节点名称。
            execution_context: 当前图运行的执行上下文。
            config: 节点的配置。
            received_inputs: 从参数名到输入值的映射。

        Returns:
            传递给 `on_after_node` 的数据字典。
        """
        ...

    @abstractmethod
    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """在 `GraphNode` 执行之后运行。

        此方法在节点执行完成后被调用，可以用于：
        - 输出验证和后处理
        - 性能监控结束
        - 结果日志记录
        - 资源清理

        Args:
            node_name: 已运行的节点名称。
            execution_context: 当前图运行的执行上下文。
            config: 节点的配置。
            output: 节点的输出结果。
            input_hook_data: 从 `on_before_node` 返回的数据。
        """
        ...


# 使用数据类装饰器定义执行上下文类
@dataclass
class ExecutionContext:
    """保存单次图运行的信息。

    执行上下文包含图运行期间需要的所有元数据，包括图模式、模型ID、
    诊断数据标志等。这些信息在整个图执行过程中传递给所有组件。
    """

    # 图模式对象（不在repr中显示，避免输出过长）
    graph_schema: GraphSchema = field(repr=False)
    # 可选的模型ID，用于标识特定的模型版本
    model_id: Optional[Text] = None
    # 是否应该添加诊断数据，用于调试和性能分析
    should_add_diagnostic_data: bool = False
    # 是否为微调模式，影响组件的训练行为
    is_finetuning: bool = False
    # 当前节点名称（由 `GraphNode` 在传递给 `GraphComponent` 之前设置）
    node_name: Optional[Text] = None


# 定义图节点类
class GraphNode:
    """在图中实例化和运行 `GraphComponent`。

    `GraphNode` 是 `GraphComponent` 的包装器，允许它在图的上下文中执行。
    它负责在正确的时间实例化组件、从父节点收集输入、运行组件的执行函数
    并将输出传递给后续节点。

    主要职责：
    1. 组件生命周期管理（创建、加载、销毁）
    2. 输入参数收集和验证
    3. 执行上下文管理
    4. 钩子机制支持
    5. 异常处理和错误传播
    """

    def __init__(
        self,
        node_name: Text,
        component_class: Type[GraphComponent],
        constructor_name: Text,
        component_config: Dict[Text, Any],
        fn_name: Text,
        inputs: Dict[Text, Text],
        eager: bool,
        model_storage: ModelStorage,
        resource: Optional[Resource],
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """初始化 `GraphNode`。

        创建图节点实例，设置所有必要的属性和配置。如果 `eager=True`，
        则立即实例化组件；否则延迟到实际执行时再实例化。

        Args:
            node_name: 模式中节点的名称，用于标识和日志记录。
            component_class: 要实例化和运行的组件类（必须实现GraphComponent接口）。
            constructor_name: 用于实例化组件的方法名称（通常是 'create' 或 'load'）。
            component_config: 传递给组件的配置字典。
            fn_name: 在节点执行时要在实例化的 `GraphComponent` 上调用的函数名称。
            inputs: 从输入参数名到提供该参数的父节点名的映射。
            eager: 确定节点是立即实例化，还是在即将运行时才实例化。
            model_storage: 图组件可以用来持久化和加载自己的存储接口。
            resource: 如果给定，`GraphComponent` 将使用给定资源从 `model_storage` 加载。
            execution_context: 关于当前图运行的信息。
            hooks: 在执行前后调用的钩子列表。
        """
        # 存储节点名称
        self._node_name: Text = node_name
        # 存储组件类
        self._component_class: Type[GraphComponent] = component_class
        # 存储构造函数名称
        self._constructor_name: Text = constructor_name
        # 获取构造函数方法引用
        self._constructor_fn: Callable = getattr(
            self._component_class, self._constructor_name
        )
        # 合并默认配置和用户配置
        self._component_config: Dict[Text, Any] = rasa.utils.common.override_defaults(
            self._component_class.get_default_config(), component_config
        )
        # 存储执行函数名称
        self._fn_name: Text = fn_name
        # 获取执行函数方法引用
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        # 存储输入参数映射
        self._inputs: Dict[Text, Text] = inputs
        # 存储是否急切实例化标志
        self._eager: bool = eager

        # 存储模型存储接口
        self._model_storage = model_storage
        # 存储现有资源（用于加载已训练的组件）
        self._existing_resource = resource

        # 创建包含当前节点名称的执行上下文副本
        self._execution_context: ExecutionContext = dataclasses.replace(
            execution_context, node_name=self._node_name
        )

        # 存储钩子列表（如果未提供则使用空列表）
        self._hooks: List[GraphNodeHook] = hooks if hooks else []

        # 初始化组件实例为None（延迟实例化时使用）
        self._component: Optional[GraphComponent] = None
        # 如果设置为急切实例化，则立即加载组件
        if self._eager:
            self._load_component()

    def _load_component(self, **kwargs: Any) -> None:
        """加载组件实例。

        根据构造函数名称和提供的参数创建组件实例。支持创建新组件或从资源加载已训练的组件。

        Args:
            **kwargs: 传递给构造函数的额外参数，通常来自父节点的输出。
        """
        # 记录组件加载的调试信息
        logger.debug(
            f"Node '{self._node_name}' loading "
            f"'{self._component_class.__name__}.{self._constructor_name}' "
            f"and kwargs: '{kwargs}'."
        )

        # 获取构造函数方法
        constructor = getattr(self._component_class, self._constructor_name)
        try:
            # 调用构造函数创建组件实例
            self._component: GraphComponent = constructor(  # type: ignore[no-redef]
                config=self._component_config,
                model_storage=self._model_storage,
                resource=self._get_resource(kwargs),
                execution_context=self._execution_context,
                **kwargs,
            )
        except InvalidConfigException:
            # 传递预期的配置异常，允许更细粒度的异常处理
            raise
        except Exception as e:
            # 处理其他异常
            if not isinstance(e, RasaException):
                # 将非Rasa异常包装为GraphComponentException
                raise GraphComponentException(
                    f"Error initializing graph component for node {self._node_name}."
                ) from e
            else:
                # 记录Rasa异常并重新抛出
                logger.error(
                    f"Error initializing graph component for node {self._node_name}."
                )
                raise

    def _get_resource(self, kwargs: Dict[Text, Any]) -> Resource:
        """获取组件使用的资源。

        确定组件应该使用哪个资源进行实例化。优先级如下：
        1. 父节点提供的资源（训练时）
        2. 现有的持久化资源（推理时）
        3. 新创建的资源（新组件）

        Args:
            kwargs: 来自父节点的参数，可能包含资源信息。

        Returns:
            组件应该使用的资源对象。
        """
        if "resource" in kwargs:
            # 父节点在训练期间提供资源。此 `GraphNode` 包装的组件
            # 将从此资源加载自己。
            return kwargs.pop("resource")

        if self._existing_resource:
            # 组件应该在推理期间从训练好的资源加载。
            # 例如，分类器可能在训练期间训练并持久化自己，
            # 然后在推理期间从此资源加载自己。
            return self._existing_resource

        # 组件获得持久化自己的机会（创建新资源）
        return Resource(self._node_name)

    def __call__(
        self, *inputs_from_previous_nodes: Union[Tuple[Text, Any], Text]
    ) -> Tuple[Text, Any]:
        """当节点在图中执行时调用 `GraphComponent` 的运行方法。

        这是图节点的核心执行方法，负责：
        1. 收集和验证父节点输出
        2. 运行前置钩子
        3. 实例化组件（如果未急切实例化）
        4. 执行组件函数
        5. 运行后置钩子
        6. 返回结果

        Args:
            *inputs_from_previous_nodes: 所有父节点的输出。每个都是一个元组，
                包含节点名称和其输出。如果节点无法解析且没有输出，
                则只提供节点名称而不是元组。

        Returns:
            包含节点名称和其输出的元组。
        """
        # 过滤掉dask无法查找的参数
        received_inputs: Dict[Text, Any] = {}
        for i in inputs_from_previous_nodes:
            if isinstance(i, tuple):
                # 解析父节点输出（节点名，输出值）
                node_name, node_output = i
                received_inputs[node_name] = node_output
            else:
                # 记录未解析节点的警告
                logger.warning(
                    f"Node '{i}' was not resolved, there is no putput. "
                    f"Another component should have provided this as an "
                    f"output."
                )

        # 根据输入映射收集必需的参数
        kwargs = {}
        for input_name, input_provider_node_name in self._inputs.items():
            if input_provider_node_name not in received_inputs:
                # 如果缺少必需的输入，抛出错误
                raise GraphRunError(
                    f"Missing input to run node '{self._node_name}'. "
                    f"Expected input '{input_provider_node_name}' to "
                    f"provide parameter '{input_name}'."
                )
            kwargs[input_name] = received_inputs[input_provider_node_name]

        # 运行前置钩子
        input_hook_outputs = self._run_before_hooks(kwargs)

        # 根据是否急切实例化决定组件加载和参数分离策略
        if not self._eager:
            # 延迟实例化：分离构造函数参数和执行参数
            constructor_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            # 使用构造函数参数加载组件
            self._load_component(**constructor_kwargs)
            # 剩余参数用于执行函数
            run_kwargs = {
                k: v for k, v in kwargs.items() if k not in constructor_kwargs
            }
        else:
            # 急切实例化：所有参数都用于执行函数
            run_kwargs = kwargs

        # 记录组件执行的调试信息
        logger.debug(
            f"Node '{self._node_name}' running "
            f"'{self._component_class.__name__}.{self._fn_name}'."
        )

        try:
            # 执行组件的运行函数
            output = self._fn(self._component, **run_kwargs)
        except InvalidConfigException:
            # 传递预期的配置异常，允许更细粒度的异常处理
            raise
        except Exception as e:
            # 处理其他异常
            if not isinstance(e, RasaException):
                # 将非Rasa异常包装为GraphComponentException
                raise GraphComponentException(
                    f"Error running graph component for node {self._node_name}."
                ) from e
            else:
                # 记录Rasa异常并重新抛出
                logger.error(
                    f"Error running graph component for node {self._node_name}."
                )
                raise

        # 运行后置钩子
        self._run_after_hooks(input_hook_outputs, output)

        # 返回节点名称和输出结果
        return self._node_name, output

    def _run_after_hooks(self, input_hook_outputs: List[Dict], output: Any) -> None:
        """运行后置钩子。

        在组件执行完成后运行所有后置钩子，用于输出处理、清理等操作。

        Args:
            input_hook_outputs: 前置钩子的输出数据列表。
            output: 组件的执行输出。
        """
        for hook, hook_data in zip(self._hooks, input_hook_outputs):
            try:
                # 记录后置钩子执行的调试信息
                logger.debug(
                    f"Hook '{hook.__class__.__name__}.on_after_node' "
                    f"running for node '{self._node_name}'."
                )
                # 调用后置钩子
                hook.on_after_node(
                    node_name=self._node_name,
                    execution_context=self._execution_context,
                    config=self._component_config,
                    output=output,
                    input_hook_data=hook_data,
                )
            except Exception as e:
                # 钩子执行异常时抛出GraphComponentException
                raise GraphComponentException(
                    f"Error running after hook for node '{self._node_name}'."
                ) from e

    def _run_before_hooks(self, received_inputs: Dict[Text, Any]) -> List[Dict]:
        """运行前置钩子。

        在组件执行前运行所有前置钩子，用于输入验证、预处理等操作。

        Args:
            received_inputs: 接收到的输入参数字典。

        Returns:
            前置钩子的输出数据列表，将传递给后置钩子。
        """
        input_hook_outputs = []
        for hook in self._hooks:
            try:
                # 记录前置钩子执行的调试信息
                logger.debug(
                    f"Hook '{hook.__class__.__name__}.on_before_node' "
                    f"running for node '{self._node_name}'."
                )
                # 调用前置钩子并收集输出
                hook_output = hook.on_before_node(
                    node_name=self._node_name,
                    execution_context=self._execution_context,
                    config=self._component_config,
                    received_inputs=received_inputs,
                )
                input_hook_outputs.append(hook_output)
            except Exception as e:
                # 钩子执行异常时抛出GraphComponentException
                raise GraphComponentException(
                    f"Error running before hook for node '{self._node_name}'."
                ) from e
        return input_hook_outputs

    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> GraphNode:
        """从 `SchemaNode` 创建 `GraphNode`。

        这是一个工厂方法，用于从模式节点创建图节点实例。简化了图节点的创建过程。

        Args:
            node_name: 节点的名称。
            schema_node: 模式节点对象，包含所有必要的配置信息。
            model_storage: 模型存储接口。
            execution_context: 执行上下文。
            hooks: 可选的钩子列表。

        Returns:
            创建的 `GraphNode` 实例。
        """
        return cls(
            node_name=node_name,
            component_class=schema_node.uses,
            constructor_name=schema_node.constructor_name,
            component_config=schema_node.config,
            fn_name=schema_node.fn,
            inputs=schema_node.needs,
            eager=schema_node.eager,
            model_storage=model_storage,
            execution_context=execution_context,
            resource=schema_node.resource,
            hooks=hooks,
        )


# 使用数据类装饰器定义图模型配置类
@dataclass()
class GraphModelConfiguration:
    """在训练和预测期间作为图运行的模型配置。

    此配置类包含运行图所需的所有信息，包括训练和预测模式、
    训练类型、助手ID、语言设置等。
    """

    # 训练时使用的图模式
    train_schema: GraphSchema
    # 预测时使用的图模式
    predict_schema: GraphSchema
    # 训练类型（如端到端、NLU、对话策略等）
    training_type: TrainingType
    # 可选的助手ID，用于多助手场景
    assistant_id: Optional[Text]
    # 可选的语言设置
    language: Optional[Text]
    # 可选的Core目标节点名称
    core_target: Optional[Text]
    # 可选的NLU目标节点名称
    nlu_target: Optional[Text]
    # 可选的命名空间映射
    spaces: Optional[Dict[Text, Text]] = None
