from typing import Iterable, Text, Optional, List

from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData

# =============================================================================
# 导入器工具模块 - 提供导入器使用的通用工具函数
# =============================================================================


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    """从路径列表加载并合并训练数据。
    
    Args:
        paths: NLU 文件路径列表
        language: 语言代码
        
    Returns:
        合并后的训练数据对象
    """
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


def story_graph_from_paths(
    files: List[Text], domain: Domain, exclusion_percentage: Optional[int] = None
) -> StoryGraph:
    """从路径列表返回 `StoryGraph`。
    
    Args:
        files: 故事文件路径列表
        domain: 域对象
        exclusion_percentage: 排除的百分比
        
    Returns:
        故事图对象
    """
    from rasa.shared.core.training_data import loading

    story_steps = loading.load_data_from_files(files, domain, exclusion_percentage)
    return StoryGraph(story_steps)
