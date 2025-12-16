## 개요

이 프로젝트는 Applio RVC 코어를 감싸는 **FastAPI 백엔드**입니다.  
최상위 디렉터리에는 두 가지 주요 서브 폴더가 있습니다.

- `applio/` – RVC 모델, 데이터, 학습/추론 스크립트 등 핵심 자원
- `app/` – FastAPI 서버, 서비스 로직, 정적 HTML 콘솔

모든 로그는 기본적으로 `logs/app.log`, `logs/error.log`에 UTF-8 인코딩으로 기록되며, 콘솔에도 동시에 출력됩니다.

**로그 파일 관리**:
- `app.log`: 모든 로그 (INFO 이상), 최대 100MB, 최근 10개 파일 유지
- `error.log`: 에러 로그만, 최대 20MB, 최근 5개 파일 유지
- 자동 로테이션: 파일 크기 도달 시 자동으로 백업 파일 생성 및 오래된 파일 삭제

자세한 로깅 설정은 [LOGGING.md](LOGGING.md)를 참조하세요.

## 설치

Docker를 사용하므로 별도의 Python 환경 설정이 필요하지 않습니다. Dockerfile에서 모든 의존성을 자동으로 설치합니다.

의존성은 다음 파일에서 관리됩니다:
- `app/requirements.txt` - FastAPI 및 애플리케이션 의존성
- `applio/requirements.txt` - RVC 및 오디오 처리 의존성

## 실행 방법

### 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA GPU 및 NVIDIA Container Toolkit 설치 (GPU 사용 시)
- CUDA 11.8 지원 GPU 드라이버

### Docker로 빌드 및 실행

#### 1. 이미지 빌드

```bash
docker-compose build
```

#### 2. 컨테이너 실행

```bash
docker-compose up -d
```

#### 3. 로그 확인

```bash
docker-compose logs -f
```

#### 4. 컨테이너 중지

```bash
docker-compose down
```

#### 5. 컨테이너 재시작

```bash
docker-compose restart
```

### Docker Compose 설정

`docker-compose.yml` 파일에서 다음 설정을 확인할 수 있습니다:

- **포트**: `8000:8000` (호스트:컨테이너)
- **볼륨 마운트**:
  - `./app:/app/app` - 애플리케이션 코드
  - `./logs:/app/logs` - 애플리케이션 로그 파일 (app.log, error.log)
  - `./applio/logs:/app/applio/logs` - RVC 모델 로그 파일
  - `./applio/outputs:/app/applio/outputs` - 출력 파일
- **GPU 지원**: NVIDIA GPU 런타임 사용
- **공유 메모리**: 2GB 설정

서버가 정상적으로 실행되면 `http://localhost:8000`에서 접근할 수 있으며, `logs/app.log`에 uvicorn/FastAPI 로그가 쌓이고, Docker 로그에도 동일한 내용이 출력됩니다.

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

- `app/task_queue.py`에 정의된 **다중 워커 FIFO 큐**입니다.
- 내부적으로 `asyncio.Queue`를 사용하며, 리소스 상태에 따라 동적으로 결정된 여러 워커가 큐에서 작업을 꺼내 동시 실행합니다.
- 각 작업은 `(코루틴 함수, args, kwargs, future)` 형태로 저장되고, 완료 시 해당 `future`에 결과가 설정됩니다.
- **모든 요청은 큐에 추가되며 거부되지 않습니다.** 리소스 상태에 따라 동시 실행 가능한 작업 수가 결정됩니다.

핵심 사용 예:

```python
queue = AsyncJobQueue("train", resource_monitor=monitor, max_workers=4)
await queue.start()           # 워커 시작 (리소스 상태에 따라 워커 수 결정)
job_id = queue.enqueue_async( # 작업 등록 (즉시 job_id 반환)
    train_model,
    model_name="foo",
    dataset_path="...",
)
```

### 2) FastAPI와의 연동

- `app/main.py`에서 다음 두 개의 큐를 사용합니다.
  - `train_queue = AsyncJobQueue("train", resource_monitor=monitor, max_workers=4)`
  - `inference_queue = AsyncJobQueue("inference", resource_monitor=monitor, max_workers=4)`
- 애플리케이션 시작 시(`startup` 이벤트) 두 큐의 워커를 시작하고, 종료 시(`shutdown`) 워커를 정리합니다.
- `/train` 요청은 `train_queue.enqueue_async(...)`로 들어가고,
  `/inference` 요청은 `inference_queue.enqueue_async(...)`로 들어갑니다.
- 각 HTTP 요청 핸들러는 **작업을 큐에 등록하고 즉시 `job_id`를 반환**합니다. 실제 작업은 백그라운드에서 실행됩니다.

이 방식의 장점:

- **리소스 상태에 따라 동시에 여러 작업을 실행**할 수 있어 효율적입니다.
- **모든 요청이 큐에 추가되므로 거부되지 않습니다.** 리소스가 부족하면 대기 후 실행됩니다.
- 헬스체크에서 각 큐의 `running` / `pending` 값을 확인해 **현재 몇 개의 작업이 처리/대기 중인지 바로 파악**할 수 있습니다.
- 작업 실패 시 자동 재시도(최대 3회)가 수행됩니다.

## API 요약

### 주요 엔드포인트
- `GET /` – 헬스체크 및 시스템/큐 상태 조회
- `POST /train` – 파일 업로드 기반 학습 요청 (모든 파라미터 지원)
- `POST /inference` – 파일 업로드 기반 추론 요청 (모든 파라미터 지원)
- `GET /jobs/{queue_name}` – 큐에 있는 모든 작업 리스트 조회
- `GET /jobs/{queue_name}/{job_id}` – 특정 작업 상태 조회 (진행률 포함)
- `DELETE /jobs/{queue_name}/{job_id}` – 작업 취소
- `GET /models` – 학습된 모델 리스트 조회
- `DELETE /models/{model_id}` – 모델 삭제
- `GET /outputs` – 추론 결과 리스트 조회
- `GET /outputs/{output_id}/download` – 추론 결과 파일 다운로드
- `DELETE /outputs/{output_id}` – 추론 결과 파일 삭제
- `GET /ui` – 웹 기반 콘솔 UI

### 학습 파라미터 (`POST /train`)
- `sample_rate`: 샘플레이트 (기본: 40000)
- `total_epoch`: 총 epoch 수 (기본: 200)
- `batch_size`: 배치 크기 (기본: 4)
- `embedder_model`: 임베더 모델 (기본: contentvec)
- `vocoder`: 보코더 (기본: HiFi-GAN)
- `overtraining_detector`: 과적합 감지 활성화 여부 (기본: true)
- `custom_pretrained`: 커스텀 사전 학습 모델 사용 여부 (기본: false)
- `g_pretrained_path`: G 모델 사전 학습 경로 (Custom Pretrained 활성화 시)
- `d_pretrained_path`: D 모델 사전 학습 경로 (Custom Pretrained 활성화 시)

### 추론 파라미터 (`POST /inference`)
- `model_path`: 모델 파일 경로 (.pth) - 필수
- `index_path`: 인덱스 파일 경로 (.index) - 선택
- `output_dir`: 출력 디렉토리 (기본: outputs)
- `pitch`: 피치 조절 값 (반음 단위, -24 ~ 24, 기본: 0)
  - 0이 아닌 경우 보컬과 배경 음원 모두 동일하게 피치 조절됩니다 (속도는 유지).
- `volume_envelope`: 볼륨 엔벨로프 (기본: 1.0)
- `protect`: 보호 계수 (기본: 0.5)
- `f0_autotune`: F0 오토튠 활성화 여부 (기본: false)
- `f0_autotune_strength`: F0 오토튠 강도 (기본: 1.0)
- `index_rate`: 인덱스 사용 비율 (0.0 ~ 1.0, 기본: 0.75)
- `embedder_model`: 임베더 모델 (기본: contentvec)

모든 경로 인자는 기본적으로 **프로젝트 루트** 기준 상대 경로를 허용하며, 서버 내부에서 절대 경로로 안전하게 변환됩니다.

## 작동 방식 (요청 → 처리 → 결과물)

### 공통 구조
1. 클라이언트가 HTTP 요청을 보내면 FastAPI 엔드포인트가 입력을 검증한 뒤 **AsyncJobQueue**(`train` 또는 `inference`)에 작업을 넣는다.
2. 큐는 단일 워커로 순차 실행되므로 GPU/CPU 사용량을 예측하기 쉽고, 헬스체크(`/`) 응답에서 `running/pending` 상태를 모니터링할 수 있다.
3. 실제 연산은 `app/services.py`에 정의된 `train_model`, `run_inference`가 담당한다.

### 학습 흐름 (`POST /train`)
1. **파일 업로드가 있는 경우**  
   - 업로드된 오디오는 `applio/datasets/<모델명>/audio_XXX.ext`로 저장된다.  
   - 이미 데이터셋 폴더가 있는 경우에는 해당 경로를 직접 지정할 수 있다.
2. **보컬 분리** (추론과 동일한 방식)  
   - 학습 시작 전 모든 오디오 파일에 대해 Spleeter를 별도 프로세스에서 실행하여 보컬/반주 분리를 수행한다.  
   - 분리된 보컬 파일로 원본 파일을 교체하여 보컬만으로 학습이 진행된다.  
   - 메모리 누수 방지를 위해 별도 프로세스에서 실행되며, 프로세스 종료 시 모든 TensorFlow 메모리가 자동으로 해제됩니다.
3. 큐에서 `train_model`이 실행되면 다음 순서로 RVC 스크립트를 호출한다.  
   - `run_prerequisites_script` → 환경 검사  
   - `run_preprocess_script` → 전처리 결과가 `applio/logs/<모델명>/preprocess` 등에 생성  
   - `run_extract_script` → 특징 추출 (f0, 임베딩 등)  
   - `run_train_script` → 본격 학습, 체크포인트 `.pth`가 `applio/logs/<모델명>`에 저장
4. 학습 완료 후에는  
   - 요청에 사용된 데이터셋 폴더를 삭제(`_remove_dataset`)  
   - `applio/logs/<모델명>` 안에서 `.pth`와 `.index`를 제외한 중간 산출물을 비우고(`_remove_preprocess`), 최종 모델 파일과 인덱스 파일만 남긴다.  
   - 모델 정보(파라미터, 파일 경로 등)가 `model_info.json`으로 저장된다.
5. API 응답에는 모델명, 로그 디렉터리, 학습 파라미터 요약이 포함된다. 최종 모델(.pth)은 `applio/logs/<모델명>`에 존재한다.

### 추론 흐름 (`POST /inference`)
1. 입력 오디오는 `applio/datasets/target_audio/temp_inference_<UUID>.wav`로 저장된다.
2. `run_inference`는 다음 단계를 순차 실행한다.  
   - **보컬/반주 분리**: Spleeter를 별도 프로세스(`app/services/spleeter_worker.py`)에서 실행하여 `applio/output/temp_inference_<UUID>/{vocals,accompaniment}.wav` 생성  
     - 메모리 누수 방지를 위해 별도 프로세스에서 실행되며, 프로세스 종료 시 모든 TensorFlow 메모리가 자동으로 해제됩니다.
   - **보컬 변환**: RVC를 별도 프로세스(`app/services/rvc_worker.py`)에서 실행하여 변환된 보컬을 `<UUID>_vocal_infer.wav`로 저장 (index_rate 파라미터 사용)  
     - 메모리 누수 방지를 위해 별도 프로세스에서 실행되며, 프로세스 종료 시 모든 PyTorch 모델 메모리가 자동으로 해제됩니다.
   - **합성**: 변환된 보컬 + 원본 반주를 합쳐 `<UUID>_final.wav` 생성 (`applio/output/` 하위)
     - 피치 조절이 0이 아닌 경우 배경 음원도 동일하게 피치 조절됩니다 (속도는 유지).
3. 최종 응답에는 `output_path`(예: `applio/output/xxxxxxxx_final.wav`)가 포함되고,  
   - 정리 단계에서 임시 파일들(`vocals.wav`, `accompaniment.wav`, `*_vocal_infer.wav`)은 삭제되어 **최종 결과물만 남는다.**
4. 생성된 파일은 `/outputs/{output_id}/download` 또는 `/download?path=<상대경로>`로 내려받을 수 있으며, 허용 경로는 `applio/output` 내부로 제한된다.

## 주요 기능

### 메모리 관리
- **별도 프로세스 실행**: RVC와 Spleeter는 별도 프로세스에서 실행되어 메모리 누수를 완전히 방지합니다.
  - `app/services/rvc_worker.py`: RVC inference를 별도 프로세스에서 실행
  - `app/services/spleeter_worker.py`: Spleeter 보컬 분리를 별도 프로세스에서 실행
  - 프로세스 종료 시 모든 메모리(CPU, GPU, TensorFlow, PyTorch)가 자동으로 해제됩니다.
- **메모리 추적**: 추론 작업 전후 메모리 사용량을 로깅하여 메모리 누수를 모니터링합니다.
- **자동 정리**: 작업 완료 후 임시 파일과 메모리를 자동으로 정리합니다.

### 작업 큐 및 재시도
- 모든 작업은 비동기 큐에서 실행됩니다.
- 리소스 상태(CPU, 메모리, GPU)에 따라 동시에 여러 작업을 실행할 수 있습니다 (최대 4개).
- 모든 요청은 큐에 추가되며 거부되지 않습니다. 리소스가 부족하면 대기 후 실행됩니다.
- 작업 실패 시 자동으로 최대 3회 재시도합니다 (지수 백오프 적용).
- 재시도 실패 시 클라이언트에 오류를 반환합니다.

### 진행률 추적
- 학습 작업의 경우 현재 epoch와 진행률을 실시간으로 조회할 수 있습니다.
- `GET /jobs/train` 엔드포인트로 모든 학습 작업의 진행률을 확인할 수 있습니다.

### 모델 관리
- 학습 완료 시 모델 정보가 `model_info.json`으로 저장됩니다.
- 모델 리스트 조회 시 모델 이름, 파라미터, 파일 경로 등 상세 정보를 확인할 수 있습니다.
- UI에서 모델을 선택하여 추론에 바로 사용할 수 있습니다.

### 작업 취소
- `DELETE /jobs/{queue_name}/{job_id}` 엔드포인트로 작업을 취소할 수 있습니다.
- PENDING 상태인 작업은 즉시 취소됩니다.
- RUNNING 상태인 작업은 취소 플래그가 설정되며, 작업 완료 시 취소 상태로 표시됩니다.
- UI에서도 작업 리스트의 취소 버튼을 통해 작업을 취소할 수 있습니다.

### UI 기능
- 웹 기반 콘솔(`GET /ui`)에서 모든 기능을 사용할 수 있습니다.
- 학습 및 추론의 모든 파라미터를 UI에서 입력할 수 있습니다.
- 사전 학습 모델 경로를 드롭다운으로 선택할 수 있습니다 (절대 경로 사용).
- 작업 리스트를 표로 확인하고 자동 새로고침이 가능합니다.
- 작업 취소 버튼을 통해 실행 중인 작업을 취소할 수 있습니다.
- 모델 및 출력 파일은 테이블에서 직접 삭제할 수 있습니다.

## 로깅

- 모든 로그는 통일된 포맷으로 기록됩니다: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- 로그 파일:
  - `logs/app.log`: 일반 로그 (INFO 레벨 이상)
  - `logs/error.log`: 에러 로그 (ERROR 레벨만)
- 콘솔에도 동시에 출력되며, UTF-8 인코딩으로 기록됩니다.
- 에러 발생 시 스택 트레이스가 자동으로 포함됩니다.

## 기타

- **Docker 기반 실행**: 모든 기능은 Docker 컨테이너에서 실행됩니다.
- **볼륨 마운트**: 로그와 출력 파일은 호스트 시스템에 마운트되어 컨테이너 재시작 후에도 유지됩니다.
- FastAPI 설정, 서비스 로직 등은 `app/` 내부에 존재합니다.
- RVC 관련 자원(모델, 데이터, 학습 설정 등)은 `applio/` 폴더 안에서 관리합니다.
- CSS는 `app/public/styles.css`에서 별도로 관리됩니다.
- 코드 품질: 모든 함수에 docstring이 포함되어 있으며, 타입 힌트가 추가되었습니다.


