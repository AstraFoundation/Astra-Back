# Astra Backend

**Sensitivity-Guided Pareto Search + Surrogate Model + Telemetry Closed-Loop** —
온디바이스 AI 모델 압축 자동화 서비스 Astra의 백엔드.

- **Astra-Front**(`../Astra-Front`)의 zod 검증 API 계약을 1:1로 충족합니다.
- **Astra-PoC**(`../Astra-PoC`)의 실제 압축 엔진(`astra` 패키지)을 구동합니다 —
  모델을 import하면 진짜 **UOSA 민감도 분석 → Optuna 3D Pareto 탐색 → DFCV 검증 →
  압축 ONNX 아티팩트**가 백그라운드에서 실행됩니다.

## 아키텍처

```
POST /api/models/import {fileName}            POST /api/models/upload (multipart)
        │ (파일 바이트 없음 → 확장자/이름 힌트로            │ (실제 파일)
        │  실제 소형 모델 합성: torch CNN/LSTM/Attention,   │
        │  sklearn GradientBoosting)                       │
        ▼                                                  ▼
   JobManager (ThreadPool) ──► astra 파이프라인 (6 페이즈, 실제 수치 로깅)
        │   ingest → detect → OnnxAnalyzer → compute_uosa(UOSA)
        │   → ParetoSearch(Optuna) → Surrogate(sklearn) → DFCV validate → export
        ▼
   SQLite (ingestion_logs · result_cache · runs …)
        │                                   │
        ▼                                   ▼
  SSE /ingestion/:runId/stream      GET /architecture · /pareto
  (폴링 폴백: /logs?after=seq)       (실제 그래프 + 실제 trial 매핑)
```

- **시드 모델 7종**: 첫 부팅 시 front 데모 데이터(HAN/LSTM/DiT 패밀리)가 결정적
  생성기로 시드됩니다 — 대시보드/목록/텔레메트리가 즉시 동작.
- **Import된 모델**: 실제 ONNX 그래프와 실제 Optuna trial이 응답으로 서빙됩니다.
- **Surrogate**: PoC의 빈 스텁을 백엔드에서 구현 — trial 데이터로 sklearn 회귀를
  학습해 실패한 latency 측정을 보정하고 실제 MAE/R²를 로그에 보고.
- **Telemetry Closed-Loop**: 드리프트 감지(acc −0.5pt / p95 +10% 임계) →
  Alert + `accuracy_drift` ActivityEvent + **재최적화 런 자동 큐잉**.

## 실행

압축 엔진(`astra/`)은 이 repo에 **벤더링**되어 있어 별도 설치가 필요 없습니다.
`pip install -e .`이 `app/`과 `astra/`를 함께 설치하고, `[engine]` extra가 엔진의
서드파티 런타임 의존성(onnx/onnxruntime/optuna/sklearn/skl2onnx/torch)을 가져옵니다.

```bash
# 의존성 (Python ≥3.10) — 백엔드 + 벤더 엔진 + 엔진 런타임 deps
pip install -e ".[engine]"
# 테스트만 돌릴 땐 경량 조합으로 충분 (skl2onnx 제외):
#   pip install -e ".[dev,test-engine]"

# 서버
uvicorn app.main:app --port 8000
# 데모/CI용 고속 모드 (tiny 모델 + 4 trials)
ASTRA_FAST_PIPELINE=1 uvicorn app.main:app --port 8000
```

> **단일 워커 전제**: 진행 중인 작업 레지스트리가 프로세스 인메모리라 컨테이너당
> uvicorn 1 프로세스를 가정합니다. 수평 확장은 uvicorn `--workers N`이 아니라
> 컨테이너 복제(`docker compose --scale api=N --scale worker=N`)로 하세요. 작업
> 상태는 Postgres, 큐는 Redis에 있어 어느 복제본이든 같은 작업을 서빙합니다.

### 프론트엔드 연동

SPA는 `fetch('/api' + path)`(동일 출처)를 호출하고 DEV에선 MSW가 가로챕니다.
실제 백엔드에 연결하려면 `Astra-Front/vite.config.ts`에 프록시를 추가하세요:

```ts
export default defineConfig({
  // ...
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
```

MSW는 `onUnhandledRequest: "bypass"`라 핸들러를 제거한 피처부터 점진적으로 실제
API로 전환할 수 있습니다. 인제스천 로그 스트림은
`GET /api/models/:id/ingestion/:runId/stream`(SSE, `IngestionLog` 동일 형태)으로
`startIngestionStream`을 대체하면 됩니다.

## API

Front 계약 (zod `.parse()` 통과 — `scripts/contract_check.mjs`로 검증):

| Method | Path | 응답 |
|---|---|---|
| GET | `/api/dashboard/summary` · `/runs?status=` · `/pareto-snapshot` · `/top-models` · `/compute-cost` · `/activity?limit=` | KPI/런/스냅샷/랭킹/비용/타임라인 |
| GET | `/api/models?q=&onlyDeployed=&sort=key:dir` · `/api/models/:id` | `ModelListItem[]` |
| POST | `/api/models/import` `{fileName}` | `{runId, modelId, fileName}` → 실제 파이프라인 시작 |
| POST | `/api/models/:id/ingestion/complete` | `{ok:true}` (멱등) |
| GET | `/api/models/:id/architecture` · `/pareto` | 실제 그래프/Optuna 매핑 (시드는 생성기) |
| GET | `/api/models/:id/telemetry/{kpi,series?range=,percentiles,deployments,alerts}` | 결정적 시계열 (1h=60 / 6h=72 / 24h=96 / 7d=84 / 30d=60 포인트) |
| GET | `/api/sdk/{snippets,keys,webhooks,recipes}` | snippets는 언어 키 **객체** |

백엔드 확장 (additive):

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/models/:id/architecture/scene?segments=1` | **3D 시각화 풀 페이로드** — 퍼셉트론(뉴런)별 월드 좌표, 레이어 유효 width(`widthFor` 폴백 포함), bipartite 엣지 기하(+`segments=1`이면 라인 세그먼트 좌표 인라인), 민감도/viridis 색상, 카메라 프레이밍, 레이어 설명·LaTeX 수식 — `LayerGraph3D`가 클라이언트에서 계산하던 전부 |
| GET | `/api/models/:id/pareto/scene?maxLatency=&maxAccuracyDrop=&maxSize=` | **Pareto 3D 풀 페이로드** — ±6% 패딩 도메인, 0..4 축 좌표 + 월드 좌표, frontier/기본 색·스케일, 제약 기반 디밍(쿼리로 슬라이더 값 전달, 기본=budget), 축 틱 값·라벨(적응형 포맷), 툴팁 문자열, 카메라 — `ParetoFrontierPlot3D`가 계산하던 전부 |
| POST | `/api/models/upload` (multipart `file`) | 실제 모델 파일 업로드 → 실제 파이프라인 |
| GET | `/api/models/:id/ingestion/:runId` | 런 상태 + 진행률 |
| GET | `/api/models/:id/ingestion/:runId/logs?after=seq` | 로그 폴링 (커서) |
| GET | `/api/models/:id/ingestion/:runId/stream` | SSE 실시간 로그 (`event: log` / `done`) |
| GET | `/api/models/:id/artifact` | 압축된 ONNX 다운로드 |
| POST | `/api/models/:id/telemetry/simulate-drift` | 드리프트 주입 (closed-loop 데모/테스트) |
| GET | `/healthz` | 헬스 체크 |

## 검증

```bash
# 1) 전체 테스트 (fast 파이프라인으로 실제 e2e 포함, ~1분)
python3 -m pytest tests/

# 2) 라이브 서버 스모크 — 전 엔드포인트 + 실제(non-fast) 파이프라인 e2e
uvicorn app.main:app --port 8000 &
python3 scripts/smoke.py

# 3) zod 계약 검증 — front의 실제 스키마로 라이브 응답 .parse()
node scripts/contract_check.mjs --base http://localhost:8000 --front ../Astra-Front

# 4) 3D scene 수치 패리티 — front의 실제 mapRange/viridis 함수로 JS에서 재계산해
#    백엔드 scene 응답과 === 비교 (~16,000개 값)
node scripts/scene_parity_check.mjs --base http://localhost:8000 --front ../Astra-Front

# 5) 폐루프 e2e — signup→파이프라인→배포→실추론→KPI/SSE→실알림까지 13개 체크
python3 scripts/verify_closed_loop.py --base http://localhost:8000

# 6) SDK 타지 검증 — wheel 빌드→fresh venv 설치→로컬 서빙→드리프트 알림 assert
scripts/verify_sdk_e2e.sh http://localhost:8000

# 7) 실모델 압축 게이트 — test-models 업로드→실압축률/certificate/trial export 검증
python3 scripts/verify_real_models.py --base http://localhost:8000 \
    --models squeezenet1.1-7.onnx har-cnn-full.h5
```

### 폐루프 & SDK (요약)

- 드리프트 모니터는 기본으로 API 프로세스 안에서 돕니다(`ASTRA_MONITOR_INLINE_ENABLED=1`
  기본값). arq 워커를 운영하는 스케일드 배포는 0으로 끄고 워커 cron을 사용하세요.
- 서빙 아티팩트는 guarantee 사다리(OFS≥`ASTRA_TAU`)를 통과한 인증본입니다 —
  인제스션 로그에 certificate가 남습니다.
- `clients/python`의 **astra-ai-sdk**(PyPI 배포용)은 `AstraRunner`로 배포 아티팩트를
  내려받아 로컬 서빙하면서 텔레메트리(지연 분해·시스템 스냅샷·입출력 분포 윈도우)를
  `/api/v1/telemetry`로 전송합니다 — Telemetry 탭의 SDK clients/breakdown/output
  패널과 prediction/input drift 알림이 이 데이터로 동작합니다.

## 디렉토리

```
app/
  main.py             앱 팩토리 (/api 마운트, CORS, lifespan: DB init+seed)
  config.py           ASTRA_* 설정 / 결정적 REF 시각
  db.py dbmodels.py   SQLite(WAL) + SQLModel 테이블
  schemas/            zod 1:1 미러 pydantic (camelCase)
  repositories.py     목록 정렬/필터 (front mock 의미론 재현) + result cache
  routers/            dashboard·models·ingestion(SSE)·architecture·pareto·telemetry·sdk
  services/
    jobs.py           ThreadPool JobManager (취소/타임아웃/상태 전이)
    model_factory.py  fileName → 실제 소형 모델 합성 (torch/sklearn)
    architecture_gen.py pareto_gen.py telemetry_gen.py   시드용 결정적 생성기 (front 포팅)
    surrogate.py      sklearn 서로게이트 (PoC 스텁의 백엔드 구현)
    drift.py          드리프트 감지 + closed-loop 액션
    mappers/          op_kind / layout / GraphInfo→Architecture / ParetoResult→Experiment
  engine/adapter.py   유일한 astra 임포트 지점 — 6페이즈 파이프라인 러너
  seed.py             front mockData 기반 첫 부팅 시드
tests/                계약·매퍼·e2e·SSE·드리프트·결정성 (57 tests)
scripts/smoke.py      라이브 전 엔드포인트 스모크
scripts/contract_check.mjs   front zod 스키마 직접 로드 → 계약 검증
```

## 주의 사항

- uvicorn **단일 워커** 전제 (in-memory job registry). 수평 확장은 컨테이너 복제
  + 외부 큐(Redis/arq, `[prod]` extra)로 — 자세한 건 위 "실행" 노트와 `deploy/` 참고.
- `.pb`/`.tflite`/`.mlmodel`은 변환기 미설치로 ONNX 등가 모델을 합성하고 포맷
  라벨만 유지합니다 (실제 multipart 업로드 시 해당 포맷은 ingest 단계에서 실패 →
  모델 status `failed` + ERROR 로그).
- UOSA는 연산자당 ORT 세션을 생성하므로 `ASTRA_MAX_COMPRESSIBLE_OPS`로 캡합니다
  (초과 연산자는 FP32 보호).

### MSW mock과의 의도적 차이

- **`/ingestion/complete`는 비동기 전이**: mock은 항상 동기적으로 `draft`를
  반환하지만, 실제 파이프라인이 아직 실행 중이면 `optimizing`으로 표시했다가
  워커가 완료 시 실제 정확도와 함께 `draft`로 마무리합니다. 워커는 SPA가
  `/complete`를 호출하지 않아도 항상 모델을 종결 상태로 전이시킵니다.
- **목록 정렬의 동률(tie) 순서**: mock의 comparator는 동률에서 0을 반환하지 않는
  비추이적(non-transitive) 구현이라 V8 TimSort 내부 동작에 따라 동률 행이
  재배열됩니다. 백엔드는 안정 정렬(삽입 순서 보존)을 사용합니다 — 동률 그룹의
  순서만 다르며 zod 계약에는 영향이 없습니다.

## 라이선스

아직 라이선스 파일이 없습니다. **공개 저장소로 푸시하기 전에** 라이선스를 정해
`LICENSE` 파일을 추가하세요. `astra/`는 Astra-PoC에서 벤더링한 코드이므로 원
프로젝트의 라이선스와 호환되는지 확인하고 필요한 저작자 표시를 포함해야 합니다.
