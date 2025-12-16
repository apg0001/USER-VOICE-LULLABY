# 로깅 시스템 정리

## 개요

이 프로젝트는 Python의 `logging` 모듈을 사용하여 모든 로그를 파일과 콘솔에 동시에 출력합니다.

## 로그 파일 설정

### 1. app.log (모든 로그)

- **경로**: `logs/app.log` (프로젝트 루트 기준)
- **저장 내용**: INFO, WARNING, ERROR 레벨의 모든 로그
- **최대 크기**: 100MB
- **백업 파일 수**: 최근 10개 파일 유지
- **인코딩**: UTF-8 (BOM 포함, Windows 메모장 호환)
- **파일명 패턴**:
  - `app.log` (현재 파일)
  - `app.log.1` (가장 최근 백업)
  - `app.log.2` (두 번째로 최근 백업)
  - ...
  - `app.log.10` (가장 오래된 백업)

### 2. error.log (에러 로그만)

- **경로**: `logs/error.log` (프로젝트 루트 기준)
- **저장 내용**: ERROR 레벨 로그만
- **최대 크기**: 20MB
- **백업 파일 수**: 최근 5개 파일 유지
- **인코딩**: UTF-8 (BOM 포함)
- **파일명 패턴**:
  - `error.log` (현재 파일)
  - `error.log.1` ~ `error.log.5` (백업 파일)

## 로그 저장 위치

### Docker 환경

- **컨테이너 내부 경로**: `/app/logs/`
- **호스트 경로**: `./logs/` (프로젝트 루트)
- **볼륨 마운트**: `./logs:/app/logs` (docker-compose.yml)

### 로컬 환경

- **경로**: `logs/` (프로젝트 루트 기준)
- **절대 경로**: `{PROJECT_ROOT}/logs/`

## 로그 초기화 시점

1. **모듈 임포트 시**: `app/main.py`가 임포트될 때 `configure_logging()` 호출
2. **애플리케이션 시작 전**: uvicorn이 시작되기 전에 로깅 설정 완료
3. **한 번만 실행**: `configure_logging._configured` 플래그로 중복 실행 방지

## 로그 출력 대상

### 1. 콘솔 (stdout/stderr)

- **핸들러**: `logging.StreamHandler`
- **출력 시점**: 실시간
- **대상**: Docker 로그 (`docker-compose logs -f`), 터미널

### 2. app.log 파일

- **핸들러**: `RotatingFileHandler`
- **출력 시점**: 실시간 (버퍼링 최소화)
- **대상**: `logs/app.log`

### 3. error.log 파일

- **핸들러**: `RotatingFileHandler`
- **출력 시점**: 실시간
- **대상**: `logs/error.log`

## 로그 포맷

모든 로그는 다음 형식으로 저장됩니다:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**예시**:
```
2025-01-16 10:30:45,123 - app.routers.inference - INFO - 추론 요청 수신 | model_path=logs/model.pth
2025-01-16 10:30:46,456 - app.services - ERROR - 모델 로드 실패 | error=FileNotFoundError
```

## 로그 레벨

### 사용되는 로그 레벨

- **INFO**: 일반 정보 (요청 수신, 작업 시작/완료 등)
- **WARNING**: 경고 (메모리 증가, 재시도 등)
- **ERROR**: 에러 (예외 발생, 작업 실패 등)

### 로그 레벨 필터링

- **루트 로거**: INFO 이상만 출력
- **app.log**: INFO 이상 저장
- **error.log**: ERROR만 저장

## 로그 소스

### 1. 애플리케이션 로그

- **모듈**: `app/` 내 모든 모듈
- **로거 생성**: `get_logger(__name__)` 사용
- **예시**:
  - `app.routers.inference`
  - `app.services`
  - `app.task_queue`

### 2. uvicorn 로그

- **로거 이름**: `uvicorn`, `uvicorn.access`
- **설정**: 루트 로거로 전파 (`propagate=True`)
- **기본 핸들러**: 제거하고 루트 로거 핸들러만 사용

### 3. FastAPI 로그

- **로거 이름**: `fastapi`
- **설정**: 루트 로거로 전파

## 로그 로테이션 동작

### RotatingFileHandler 동작 방식

1. **현재 파일 크기 확인**: `app.log`가 100MB에 도달하면
2. **백업 파일 생성**: 
   - 기존 `app.log.1` → `app.log.2`
   - 기존 `app.log.2` → `app.log.3`
   - ...
   - 기존 `app.log.10` → 삭제
   - 현재 `app.log` → `app.log.1`
3. **새 파일 생성**: 새로운 `app.log` 파일 생성
4. **자동 정리**: 오래된 백업 파일 자동 삭제

## 로그 확인 방법

### Docker 환경

```bash
# 실시간 로그 확인 (콘솔)
docker-compose logs -f fastapi-app

# 파일에서 확인
cat logs/app.log
tail -f logs/app.log

# 에러 로그만 확인
cat logs/error.log
tail -f logs/error.log
```

### 로컬 환경

```bash
# 실시간 로그 확인
tail -f logs/app.log

# 최근 100줄 확인
tail -n 100 logs/app.log

# 에러 로그 확인
cat logs/error.log
```

## 로그 파일 관리

### 자동 관리

- **로테이션**: 파일 크기 도달 시 자동 로테이션
- **백업 파일 삭제**: 오래된 백업 파일 자동 삭제
- **디렉토리 생성**: `logs/` 디렉토리 자동 생성

### 수동 관리

필요한 경우 수동으로 로그 파일을 삭제하거나 백업할 수 있습니다:

```bash
# 로그 파일 삭제 (주의: 모든 로그가 삭제됨)
rm -rf logs/*.log*

# 로그 파일 백업
cp -r logs/ logs_backup_$(date +%Y%m%d)/
```

## 주의사항

1. **로그 파일 크기**: app.log는 최대 100MB × 10개 = 약 1GB까지 저장될 수 있습니다.
2. **디스크 공간**: 로그 파일이 계속 증가하므로 주기적으로 확인이 필요합니다.
3. **성능 영향**: 로그 파일이 매우 클 경우 I/O 성능에 영향을 줄 수 있습니다.
4. **Docker 볼륨**: `./logs:/app/logs` 볼륨 마운트가 설정되어 있어야 호스트에서 로그 파일에 접근할 수 있습니다.

## 설정 변경

로그 설정을 변경하려면 `app/logging_config.py`의 다음 부분을 수정하세요:

```python
# app.log 설정
app_handler = _build_rotating_handler(
    LOG_FILE_PATH, 
    level=logging.INFO,
    max_bytes=100 * 1024 * 1024,  # 100MB
    backups=10  # 최근 10개 파일 유지
)

# error.log 설정
error_handler = _build_rotating_handler(
    ERROR_FILE_PATH, 
    level=logging.ERROR,
    max_bytes=20 * 1024 * 1024,  # 20MB
    backups=5  # 최근 5개 파일 유지
)
```

