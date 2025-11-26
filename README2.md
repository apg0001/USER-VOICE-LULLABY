# Applio FastAPI Backend

최상위 디렉터리에는 두 가지 주요 서브 폴더가 있습니다.

- `rvc/` – 모델, 데이터, 학습/추론 스크립트 등 Applio 핵심 자원
- `app/` – FastAPI 서버 및 정적 HTML 콘솔

## 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

이 명령은 `rvc/requirements.txt`와 `app/requirements.txt`의 모든 의존성을 한 번에 설치합니다.

## 실행 방법

### Windows

```powershell
.\run-applio.bat --host 0.0.0.0 --port 8000
```

### Linux / macOS

```bash
chmod +x run-applio.sh
./run-applio.sh --host 0.0.0.0 --port 8000
```

### 수동 실행

```bash
python -m app --host 0.0.0.0 --port 8000
```

## API 요약

- `GET /` – 헬스체크 및 기본 설정 조회
- `POST /train` – 전처리 → 특징 추출 → 학습 파이프라인 실행
- `POST /inference` – 학습된 모델(.pth)과 선택적 인덱스로 음성 변환
- `GET /ui` – `/train`, `/inference`를 바로 호출할 수 있는 정적 HTML 콘솔

모든 경로 인자는 **최상위 디렉터리**를 기준으로 해석됩니다. 예를 들어 `dataset_path`에 `rvc/logs/...` 처럼 상대 경로를 넘겨도 자동 보정됩니다.

## 기타

- Windows용 실행 배치(`run-applio.bat`)와 Linux/macOS용 스크립트(`run-applio.sh`)가 최상위 디렉터리에 배치되어 있습니다.
- FastAPI 설정, 서비스 로직, 정적 파일 등은 모두 `app/` 내부에 존재하므로 IDE에서는 이 폴더를 Python 모듈 루트로 인식하면 됩니다.
- RVC 관련 유지보수(모델, 데이터, 학습 설정 등)는 `rvc/` 폴더 안에서 수행하세요.

