## 개요

이 프로젝트는 Applio RVC 코어를 감싸는 **FastAPI 백엔드**입니다.  
최상위 디렉터리에는 두 가지 주요 서브 폴더가 있습니다.

- `applio/` – RVC 모델, 데이터, 학습/추론 스크립트 등 핵심 자원
- `app/` – FastAPI 서버, 서비스 로직, 정적 HTML 콘솔

모든 로그는 기본적으로 `logs/app.log`, `logs/error.log`에 UTF-8 인코딩으로 기록되며, 콘솔에도 동시에 출력됩니다.

## 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

이 명령은 `app/requirements.txt`와 `applio/requirements.txt`의 의존성을 한 번에 설치합니다.

## 실행 방법

### Windows

```powershell
start.bat
```

### Linux / macOS / WSL

```bash
chmod +x start.sh
./start.sh
```

서버가 정상적으로 실행되면 `logs/app.log`에 uvicorn/FastAPI 로그가 쌓이고, 콘솔에도 동일한 내용이 출력됩니다.

## 헬스체크 (`GET /`)

헬스체크 엔드포인트는 **파라미터 기본값 정보 없이**, 다음과 같은 런타임 상태만 반환합니다.

- **status**: `"ok"` 고정
- **cpu_percent**: 현재 CPU 사용률
- **memory_percent**: 현재 메모리 사용률
- **disk_percent**: `PROJECT_ROOT` 디스크 사용률
- **queues**: 작업 큐 상태
  - `train.running` / `train.pending`
  - `inference.running` / `inference.pending`

예시 응답:

```json
{
  "status": "ok",
  "cpu_percent": 12.3,
  "memory_percent": 48.7,
  "disk_percent": 35.1,
  "queues": {
    "train":    { "name": "train",    "running": 1, "pending": 2 },
    "inference":{ "name": "inference","running": 0, "pending": 0 }
  }
}
```

## 작업 큐 구조 (대기열 방식)

### 1) AsyncJobQueue

- `app/task_queue.py`에 정의된 **단일 워커 FIFO 큐**입니다.
- 내부적으로 `asyncio.Queue`를 사용하며, 하나의 워커 코루틴이 큐에서 작업을 꺼내 순차 실행합니다.
- 각 작업은 `(코루틴 함수, args, kwargs, future)` 형태로 저장되고, 완료 시 해당 `future`에 결과가 설정됩니다.

핵심 사용 예:

```python
queue = AsyncJobQueue("train")
await queue.start()           # 워커 시작
result = await queue.enqueue( # 작업 등록 + 완료까지 대기
    train_model,
    model_name="foo",
    dataset_path="...",
)
```

### 2) FastAPI와의 연동

- `app/main.py`에서 다음 두 개의 큐를 사용합니다.
  - `train_queue = AsyncJobQueue("train")`
  - `inference_queue = AsyncJobQueue("inference")`
- 애플리케이션 시작 시(`startup` 이벤트) 두 큐의 워커를 시작하고, 종료 시(`shutdown`) 워커를 정리합니다.
- `/train`, `/train-files` 요청은 `train_queue.enqueue(train_model, ...)`로 들어가고,
  `/inference`, `/inference-files` 요청은 `inference_queue.enqueue(run_inference, ...)`로 들어갑니다.
- 각 HTTP 요청 핸들러는 **큐 안에서 작업이 끝날 때까지 `await`** 한 뒤, 그 결과를 응답으로 반환합니다.

이 방식의 장점:

- **동시 요청이 들어와도 실제 학습/추론은 한 번에 하나만 실행**되므로, GPU/CPU 사용량을 예측하기 쉽고 안정적입니다.
- 헬스체크에서 각 큐의 `running` / `pending` 값을 확인해 **현재 몇 개의 작업이 처리/대기 중인지 바로 파악**할 수 있습니다.

## API 요약

- `GET /` – 헬스체크 및 시스템/큐 상태 조회
- `POST /train` – 전처리 → 특징 추출 → 학습 파이프라인 실행
- `POST /inference` – 학습된 모델(.pth)과 선택적 인덱스로 음성 변환
- `POST /train-files` – 파일 업로드 기반 학습 요청
- `POST /inference-files` – 파일 업로드 기반 추론 요청
- `GET /ui` – `/train`, `/inference`를 바로 호출할 수 있는 정적 HTML 콘솔

모든 경로 인자는 기본적으로 **프로젝트 루트** 기준 상대 경로를 허용하며, 서버 내부에서 절대 경로로 안전하게 변환됩니다.

## 기타

- Windows: `start.bat`, Linux/macOS/WSL: `start.sh`로 서버를 빠르게 실행할 수 있습니다.
- FastAPI 설정, 서비스 로직 등은 `app/` 내부에 존재합니다.
- RVC 관련 자원(모델, 데이터, 학습 설정 등)은 `applio/` 폴더 안에서 관리합니다.


