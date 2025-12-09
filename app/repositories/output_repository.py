"""추론 결과 리포지토리"""
from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Optional

from app.constants import RVC_ROOT
from app.logging_config import get_logger
from app.models.responses import OutputInfo

logger = get_logger(__name__)

OUTPUT_ROOT = RVC_ROOT / "outputs"


class OutputRepository:
    """추론 결과 관리 리포지토리"""
    
    def __init__(self, output_root: Path | None = None):
        self.output_root = output_root or OUTPUT_ROOT
        self._logger = logger
    
    def list_outputs(self) -> list[OutputInfo]:
        """추론 결과 리스트 조회"""
        if not self.output_root.exists():
            return []
        
        outputs = []
        
        # 직접 파일들 조회
        for file_path in self.output_root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                output_id = file_path.name
                file_size = file_path.stat().st_size
                created_at = self._get_created_at(file_path)
                relative_path = str(file_path.relative_to(RVC_ROOT))
                
                outputs.append(
                    OutputInfo(
                        output_id=output_id,
                        file_path=relative_path,
                        file_size=file_size,
                        created_at=created_at,
                    )
                )
        
        # 하위 디렉토리의 파일들도 조회 (temp_inference_* 폴더 제외)
        for subdir in self.output_root.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("temp_inference_"):
                for file_path in subdir.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                        output_id = file_path.name
                        file_size = file_path.stat().st_size
                        created_at = self._get_created_at(file_path)
                        relative_path = str(file_path.relative_to(RVC_ROOT))
                        
                        outputs.append(
                            OutputInfo(
                                output_id=output_id,
                                file_path=relative_path,
                                file_size=file_size,
                                created_at=created_at,
                            )
                        )
        
        return sorted(outputs, key=lambda x: x.created_at or "", reverse=True)
    
    def get_output_path(self, output_id: str) -> Path:
        """출력 파일 경로 반환 (파일명으로 검색)"""
        # 직접 파일 검색
        direct_path = self.output_root / output_id
        if direct_path.exists() and direct_path.is_file():
            return direct_path
        
        # 하위 디렉토리에서 검색
        for file_path in self.output_root.rglob(output_id):
            if file_path.is_file():
                return file_path
        
        raise FileNotFoundError(f"출력 파일을 찾을 수 없습니다: {output_id}")
    
    def delete_output(self, output_id: str) -> None:
        """추론 결과 삭제"""
        try:
            output_path = self.get_output_path(output_id)
            
            # 파일 삭제
            if output_path.is_file():
                output_path.unlink()
                self._logger.info(f"출력 파일 삭제 완료: {output_id}")
                
                # 빈 디렉토리 정리
                parent = output_path.parent
                if parent != self.output_root and not any(parent.iterdir()):
                    parent.rmdir()
                    self._logger.info(f"빈 디렉토리 삭제: {parent}")
            else:
                raise ValueError(f"파일이 아닙니다: {output_id}")
        except FileNotFoundError:
            raise
        except Exception as e:
            self._logger.exception(f"출력 파일 삭제 실패: {output_id}")
            raise RuntimeError(f"출력 파일 삭제 중 오류 발생: {str(e)}") from e
    
    @staticmethod
    def _get_created_at(file_path: Path) -> Optional[str]:
        """파일 생성 시간 조회"""
        try:
            return datetime.datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat()
        except Exception:
            return None

