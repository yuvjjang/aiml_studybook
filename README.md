# AI · 머신러닝: 기초부터 최신까지

수식과 **인터랙티브 그래픽**으로 AI/ML 전 분야를 처음부터 배우는 온라인 학습 책.
Quarto 정적 사이트로 빌드되며, 핵심 개념마다 Plotly 기반 인터랙티브 시각화(슬라이더·애니메이션)가 함께 제공됩니다.

> 선형대수와 확률에서 출발해 고전 머신러닝, 딥러닝, 트랜스포머, 생성 모델, 언어·오디오·비디오·멀티모달까지 **끊기지 않는 한 줄기**로 잇는 것이 목표입니다.

---

## 목차 (커리큘럼)

| Part | 주제 | 챕터 | 핵심 질문 |
|------|------|------|-----------|
| **0** | 수학·컴퓨팅 기초 | 6 | 선형대수, 최적화, 확률·통계, 정보이론, 수치 안정성 |
| **1** | 데이터 사이언스 | 8 | EDA, 전처리, 특성 공학, A/B 테스트, 인과추론, 시계열 |
| **2** | 고전 머신러닝 | 13 | 편향-분산, 선형·로지스틱, SVM, 트리·앙상블, 군집화, 차원축소, 평가 |
| **3** | 딥러닝 기초 | 10 | 역전파, 활성화, 손실, 옵티마이저, 정규화, 자동미분 직접 구현 |
| **4** | 신경망 아키텍처 | 7 | CNN, RNN/LSTM, seq2seq와 어텐션의 탄생, 임베딩, GNN |
| **5** | 트랜스포머 | 11 | 어텐션, 멀티헤드, RoPE, GPT 직접 구현, FlashAttention, MoE, 스케일링 법칙, SSM |
| **6** | 생성 모델 | 9 | VAE, GAN, 정규화 흐름, 확산 모델, 잠재 확산, 흐름 매칭 |
| **7** | 언어 (NLP · LLM) | 16 | 토크나이제이션, 사전학습, PEFT, 정렬, RAG, 에이전트, 추론 모델, 서빙 최적화, 안전성 |
| **8** | 오디오 · 음성 | 11 | 샘플링, STFT, MFCC, CTC/RNN-T, wav2vec, Whisper, TTS, 신경 코덱 |
| **9** | 비전 · 비디오 | 10 | 영상처리, 검출·분할, ViT, 자기지도, CLIP, 비디오 모델, 생성, VLM |
| **10** | 강화학습 | 6 | MDP, Q-learning, PPO, 그리고 RLHF·GRPO로의 연결 |
| **11** | 시스템과 MLOps | 7 | 재현성, 분산 학습, 하드웨어, 양자화, 서빙, 모니터링 |
| **12** | 해석가능성과 신뢰 | 5 | SHAP, 기계적 해석가능성, 캘리브레이션, 견고성, 공정성·프라이버시 |
| **A** | 부록 | 4 | 논문 지도, 용어 사전, 수식 치트시트, 학습 자료 |

**총 123챕터 · 계획된 인터랙티브 그래픽 483개.**
챕터별 이론·그래픽 상세 목록은 [PLAN.md](PLAN.md) 참고.

---

## 기술 스택

- **문서**: [Quarto](https://quarto.org) (`.qmd` = Markdown + Python 코드 블록)
- **계산**: NumPy, SciPy
- **그래픽**: Plotly (인터랙티브), `plotly_dark` 테마
- **수식**: MathJax (LaTeX)
- **재사용 모듈**: `src/` (Plotly 헬퍼, 합성 데이터, 어텐션·오디오·영상 참조 구현)
- **실행**: 로컬은 `quarto preview`(작업 중) 또는 정적 사이트(`_site/`) + Python `http.server`(확인용). 원격 Linux 서버 배포 시 systemd 상시 구동

### 렌더 타임 의존성을 NumPy·SciPy·Plotly 로 제한한 이유

이 책은 PyTorch·Hugging Face 를 **설치하지 않고도 전체가 빌드**됩니다.

- 모든 인터랙티브 그래픽은 NumPy 로 직접 계산합니다. 어텐션·STFT·합성곱도 `src/` 에 순수 NumPy 참조 구현이 있습니다.
- 프레임워크 코드는 실행하지 않는 참고 블록(` ```python `)으로 제시합니다.
- 덕분에 빌드가 빠르고, 어느 PC에서 렌더해도 같은 그림이 나옵니다 — 합성 데이터는 전부 시드가 고정되어 있습니다.

자세한 구조는 [docs/architecture.md](docs/architecture.md), 모듈 API는 [docs/api.md](docs/api.md), 데이터 흐름/구조 다이어그램은 [docs/uml.md](docs/uml.md) 참고.

---

## 디렉터리 구조

```
ai_ml_study/
├── _quarto.yml           # 책 설정 (챕터 순서, 테마, 포맷) — 자동 생성
├── index.qmd             # 표지 / 소개
├── PLAN.md               # 챕터별 상세 플래닝 — 자동 생성
├── chapters/             # Part 0~13 챕터 (.qmd)
│   ├── 00_foundations/   ├── 05_transformer/   ├── 10_rl/
│   ├── 01_data_science/  ├── 06_generative/    ├── 11_systems/
│   ├── 02_classical_ml/  ├── 07_language/      ├── 12_trust/
│   ├── 03_deep_learning/ ├── 08_audio/         └── 13_appendix/
│   ├── 04_architectures/ └── 09_vision_video/
├── src/                  # 재사용 계산·시각화 모듈
│   ├── viz.py            # 공통 Plotly 헬퍼 (팔레트, 레이아웃, 슬라이더)
│   ├── datasets.py       # 시드 고정 합성 데이터
│   ├── attention.py      # 어텐션·RoPE·ALiBi NumPy 참조 구현
│   ├── audio.py          # 신호 합성, STFT, 스펙트로그램
│   └── imaging.py        # 2D 합성곱, 주파수 필터, 피라미드
├── scripts/
│   ├── curriculum.py     # ★ 커리큘럼 단일 진실 원천
│   ├── gen_scaffold.py   # → _quarto.yml + 챕터 스텁 + PLAN.md 생성
│   └── build/preview/serve/render-check  (.ps1 = Windows, .sh = macOS/Linux)
├── docs/                 # 아키텍처·API·다이어그램 문서
├── custom.scss, styles.css
├── _site/                # 빌드 산출물 (배포 대상)
└── venv/                 # Python 가상환경
```

### 커리큘럼은 코드에서 생성됩니다

123챕터의 목차·스텁·플래닝 문서를 손으로 동기화하는 것은 현실적이지 않습니다.
그래서 커리큘럼은 [`scripts/curriculum.py`](scripts/curriculum.py) 한 곳에만 정의하고,
거기서 세 산출물을 만듭니다.

```powershell
python scripts\gen_scaffold.py --dry-run   # 무엇이 바뀌는지 확인
python scripts\gen_scaffold.py             # _quarto.yml + PLAN.md + 스텁 갱신
```

**본문을 채운 `.qmd` 는 덮어쓰지 않습니다.** 생성된 스텁에는 `<!-- STUB: ... -->`
주석이 들어 있고, 생성기는 그 주석이 있는 파일만 갱신합니다. 챕터 집필을 시작할 때
그 줄을 지우면 이후 재생성에서 보호됩니다.

---

## 시작하기

### 1. 환경 준비

#### Windows (PowerShell)

```powershell
# 1) Quarto CLI 설치 — 설치 후 새 터미널을 열어야 PATH에 반영됩니다
winget install --id Posit.Quarto -e

# 2) 가상환경 + 의존성
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# 3) Jupyter 커널 등록 (최초 1회)
venv\Scripts\python -m ipykernel install --user --name ai-ml-study

# 4) 로컬 설정 파일
copy .env.example .env
```

#### macOS / Linux

```bash
# Quarto CLI 는 https://quarto.org/docs/get-started/ 에서 설치
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m ipykernel install --user --name ai-ml-study
cp .env.example .env
```

> **venv 경로 차이** — Windows는 `venv\Scripts\`, macOS/Linux는 `venv/bin/` 에 실행 파일이 생성됩니다. `scripts/` 의 스크립트들은 양쪽을 자동으로 감지하므로 신경 쓰지 않아도 됩니다.

### 2. 로컬 실시간 미리보기 (권장)

콘텐츠를 작업하는 동안 저장할 때마다 자동으로 다시 렌더링되고 브라우저가 새로고침됩니다.

```powershell
.\scripts\preview.ps1              # Windows  (포트 지정: -Port 4200)
```
```bash
./scripts/preview.sh               # macOS / Linux  (포트 지정: ./scripts/preview.sh 4200)
```

### 3. 빌드

```powershell
.\scripts\build.ps1                # Windows
```
```bash
./scripts/build.sh                 # macOS / Linux
```

산출물은 `_site/` 에 생성됩니다.

### 4. 빌드 결과를 로컬 서버로 확인

```powershell
.\scripts\serve.ps1                # Windows — http://localhost:8080
```
```bash
venv/bin/python -m http.server 8080 --directory _site   # macOS / Linux
```

`serve.ps1` 은 `127.0.0.1` 에만 바인딩하므로 이 PC 밖에서는 접속되지 않습니다(방화벽 예외 불필요).

> **`.ps1` 실행이 차단된다면** — PowerShell 실행 정책 때문입니다. 현재 사용자에 한해 허용하세요:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## 챕터 작업 흐름

1. 채울 챕터를 고른다 (`PLAN.md` 에 이론 항목과 그래픽 목록이 이미 정리되어 있음).
2. 해당 `.qmd` 의 `<!-- STUB: ... -->` 줄을 지운다.
3. 본문과 Python 코드 청크를 작성한다. 반복되는 계산은 `src/` 로 뺀다.
4. 렌더 에러만 빠르게 확인한다:

```powershell
.\scripts\render-check.ps1 chapters\05_transformer\00_attention_mechanism.qmd
```
```bash
./scripts/render-check.sh chapters/05_transformer/00_attention_mechanism.qmd
```

Claude Code 를 쓴다면 `/note`(콜아웃 추가)와 `/merge`(본문 병합) 커맨드가 이 흐름을 자동화합니다 — [.claude/commands/](.claude/commands/) 참고.

---

## 상시 서비스 (systemd — Linux 서버 배포용)

> 로컬(Windows 포함)에서 작업할 때는 위 "시작하기"만으로 충분합니다.

```bash
cp .env.example .env                  # PUBLIC_HOST, PORT 지정 가능

./scripts/install-service.sh          # 등록 + 시작 (또는 ./scripts/install-service.sh 9000)

./scripts/service.sh status           # 상태
./scripts/service.sh restart          # 재시작
./scripts/service.sh logs             # 실시간 로그

./scripts/uninstall-service.sh        # 등록 해제
```

접속 호스트는 `.env` 의 `PUBLIC_HOST` → 클라우드 메타데이터 외부 IP → 내부 IP 순으로 결정됩니다.
외부에서 접속이 안 되면 클라우드 방화벽에서 **TCP 8080 인바운드**가 열려 있는지 확인하세요.

콘텐츠 갱신은 `./scripts/build.sh` 만 다시 실행하면 됩니다(서비스 재시작 불필요).

---

## 스크립트 요약

| Windows | macOS / Linux | 설명 |
|---------|---------------|------|
| — | — | `python scripts/gen_scaffold.py` — 목차·스텁·PLAN.md 재생성 |
| `scripts\build.ps1` | `scripts/build.sh` | Quarto 책을 `_site/` 로 렌더링 |
| `scripts\preview.ps1 [-Port N]` | `scripts/preview.sh [PORT]` | 실시간 미리보기 서버 |
| `scripts\render-check.ps1 <qmd>` | `scripts/render-check.sh <qmd>` | 챕터 하나만 렌더링해 빌드 에러 확인 |
| `scripts\serve.ps1 [-Port N]` | — | 빌드된 `_site/` 를 로컬 정적 서버로 서빙 |
| `scripts\_common.ps1` | — | `.ps1` 들이 공유하는 헬퍼 (직접 실행하지 않음) |
| — | `scripts/install-service.sh [PORT]` | systemd 서비스 등록·시작 |
| — | `scripts/service.sh {start\|stop\|restart\|status\|logs}` | 서비스 제어 |
| — | `scripts/uninstall-service.sh` | 서비스 중지·등록 해제 |
