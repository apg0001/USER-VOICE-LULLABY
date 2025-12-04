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

### 수동 실행

```bash
python -m app --host 0.0.0.0 --port 8000
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

## 작동 방식 (요청 → 처리 → 결과물)

### 공통 구조
1. 클라이언트가 HTTP 요청을 보내면 FastAPI 엔드포인트가 입력을 검증한 뒤 **AsyncJobQueue**(`train` 또는 `inference`)에 작업을 넣는다.
2. 큐는 단일 워커로 순차 실행되므로 GPU/CPU 사용량을 예측하기 쉽고, 헬스체크(`/`) 응답에서 `running/pending` 상태를 모니터링할 수 있다.
3. 실제 연산은 `app/services.py`에 정의된 `train_model`, `run_inference`가 담당한다.

### 학습 흐름 (`/train`, `/train-files`)
1. **파일 업로드가 있는 경우**  
   - 업로드된 오디오는 `applio/datasets/<모델명>/audio_XXX.ext`로 저장된다.  
   - 이미 데이터셋 폴더가 있는 경우에는 해당 경로를 직접 지정할 수 있다.
2. 큐에서 `train_model`이 실행되면 다음 순서로 RVC 스크립트를 호출한다.  
   - `run_prerequisites_script` → 환경 검사  
   - `run_preprocess_script` → 전처리 결과가 `applio/logs/<모델명>/preprocess` 등에 생성  
   - `run_extract_script` → 특징 추출 (f0, 임베딩 등)  
   - `run_train_script` → 본격 학습, 체크포인트 `.pth`가 `applio/logs/<모델명>`에 저장
3. 학습 완료 후에는  
   - 요청에 사용된 데이터셋 폴더를 삭제(`_remove_dataset`)  
   - `applio/logs/<모델명>` 안에서 `.pth`와 `.index`를 제외한 중간 산출물을 비우고(`_remove_preprocess`), 최종 모델 파일과 인덱스 파일만 남긴다.
4. API 응답에는 모델명, 로그 디렉터리, 학습 파라미터 요약이 포함된다. 최종 모델(.pth)은 `applio/logs/<모델명>`에 존재한다.

### 추론 흐름 (`/inference`, `/inference-files`)
1. 입력 오디오는 `applio/datasets/target_audio/temp_inference_<UUID>.wav`로 저장된다.
2. `run_inference`는 다음 단계를 순차 실행한다.  
   - **보컬/반주 분리**: spleeter로 `applio/output/temp_inference_<UUID>/{vocals,accompaniment}.wav` 생성  
   - **보컬 변환**: RVC `run_infer_script`로 변환된 보컬을 `<UUID>_vocal_infer.wav`로 저장  
   - **합성**: 변환된 보컬 + 원본 반주를 합쳐 `<UUID>_final.wav` 생성 (`applio/output/` 하위)
3. 최종 응답에는 `output_path`(예: `applio/output/xxxxxxxx_final.wav`)가 포함되고,  
   - 정리 단계에서 임시 파일들(`vocals.wav`, `accompaniment.wav`, `*_vocal_infer.wav`)은 삭제되어 **최종 결과물만 남는다.**
4. 생성된 파일은 `/download?path=<상대경로>`로 내려받을 수 있으며, 허용 경로는 `applio/output` 내부로 제한된다.

## 기타

- Windows: `start.bat`, Linux/macOS/WSL: `start.sh`로 서버를 빠르게 실행할 수 있습니다.
- FastAPI 설정, 서비스 로직 등은 `app/` 내부에 존재합니다.
- RVC 관련 자원(모델, 데이터, 학습 설정 등)은 `applio/` 폴더 안에서 관리합니다.


