from colossalai.gemini.chunk import ChunkManager
from colossalai.utils import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity

from .memstats_collector import MemStatsCollector


class ChunkMemStatsCollector(MemStatsCollector):

    def __init__(self, chunk_manager: ChunkManager) -> None:
        super().__init__()
        self._chunk_manager = chunk_manager

    # override
    def sample_model_data(self) -> None:
        """Sampling model data statistics.
        """
        if self._start_flag:
            cuda_mem = self._chunk_manager.total_mem['cuda']
            cpu_mem = self._chunk_manager.total_mem['cpu']
            self._memstats.append_model_data('cuda', cuda_mem)
            self._memstats.append_model_data('cpu', cpu_mem)

    @property
    def cuda_margin_mem(self) -> float:
        return colo_device_memory_capacity(get_current_device()) - self._memstats.max_overall_cuda('cuda')