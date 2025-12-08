"""추론 결과 관리 서비스"""
from __future__ import annotations

from ..logging_config import get_logger
from ..models.responses import OutputInfo
from ..repositories.output_repository import OutputRepository

logger = get_logger(__name__)


class OutputService:
    """추론 결과 관리 서비스"""
    
    def __init__(self, repository: OutputRepository | None = None):
        self.repository = repository or OutputRepository()
        self._logger = logger
    
    def list_outputs(self) -> list[OutputInfo]:
        """추론 결과 리스트 조회"""
        return self.repository.list_outputs()
    
    def delete_output(self, output_id: str) -> None:
        """추론 결과 삭제"""
        self.repository.delete_output(output_id)

