"""
simulate.py — 1단계: 시나리오에 맞는 '가짜' 시동 전류 자료를 만든다
=====================================================================

[이 파일이 하는 일]
  1) 정상 시동 전류 파형을 만든다. 매번 조금씩 다르게(부하·전원·노이즈) 만든다.
  2) 6가지 이상 시동 파형을 만든다. 이상 정도도 '약함~심함'으로 섞는다.
  3) 아래 네 가지 파일을 data/ 폴더에 저장한다.
       train.npz       정상만 1200건   → 모델이 "정상"을 배우는 자료
       val.npz         정상만 400건    → 학습을 언제 멈출지, 경보선을 어디에 둘지 정하는 자료
       test.npz        정상 400건 + 이상 6종 × 40건 (정답 표시 포함) → 마지막 성능 평가 전용
       stream_demo.csv 시동이 여러 번 이어지는 '센서 기록' 흉내 → 실시간 감지(run_realtime.py) 연습용

[왜 가짜 자료로 시작하나?]
  - 실제 이상(고장)은 드물어서 모으기 어렵다. 가짜 자료는 "정답"을 알고 있으므로
    모델이 정말로 잡는지 확인하기 좋다. 강의에서 원리를 익힌 뒤, 같은 코드에 실제 자료를 넣으면 된다.
  - 단, 가짜 자료에서 잘 된다고 실제 현장에서 똑같이 잘 된다는 보장은 없다. (README의 '주의' 참고)

[전류 값에 대한 가정]
  - 교류 전류는 원래 1초에 60번 오르내리는 사인파다. 여기서는 센서나 PLC가 이미 계산해 준
    '실효값(RMS) 전류'를 읽는다고 가정한다. 그래서 파형이 매끄러운 곡선 모양이다.
  - 만약 사인파 원신호를 직접 읽는다면, 한 주기(60 Hz면 약 16.7 ms)마다 실효값을 먼저 계산한 뒤 이 코드에 넣는다.

실행:  python simulate.py
"""
from __future__ import annotations

import numpy as np

import config as C

# ----------------------------------------------------------------------------- 이상 유형 (시나리오)
# 이름: (설명, 현장에서 의심되는 원인)
ANOMALY_TYPES = {
    "inrush_high": ("돌입 전류 과다", "전원 전압 상승, 전자접촉기 접점 불량, 회전자 이상"),
    "slow_start": ("기동 지연", "유압 잔압이 남은 채 시동(부하 과다), 베어링 손상, 윤활 부족"),
    "voltage_dip": ("순간 전압 강하", "같은 선로의 다른 큰 설비 동시 기동, 배전 문제"),
    "ripple": ("안정 구간 떨림", "축 정렬 불량, 커플링 마모, 펌프 맥동"),
    "phase_loss": ("결상(한 상 끊김)", "퓨즈 하나 단선, 단자 풀림 — 안정 전류가 비정상적으로 큼"),
    "stall": ("시동 실패(구속)", "기계가 걸려 회전 못 함 — 돌입 전류가 거의 줄지 않음"),
}
LABELS = ["normal"] + list(ANOMALY_TYPES)   # 0 = 정상, 1~6 = 이상 유형 번호


def startup_trace(rng: np.random.Generator, kind: str = "normal", health: float = 0.0) -> np.ndarray:
    """시동 1건의 '센서 기록'을 만든다: 정지(0 근처) 50샘플 + 시동 파형.

    health: 설비 열화 정도 (0 = 새 설비, 1 = 고장 직전). signal_generator.py가 '서서히 나빠지는' 시나리오에 쓴다.
            베어링이 닳아 가면 회전이 무거워져 감쇠가 느려지고(tau 증가), 안정 구간에 작은 떨림이 생긴다고 가정한다.
            0이면 아무 영향이 없다 (train/val/test 자료는 모두 0으로 만든다).

    반환값은 시동 앞뒤를 포함한 긴 기록이다. 실제 현장처럼 '전류가 TRIGGER를 넘는 순간'을
    찾아서 잘라 내는 일은 cut_at_trigger()가 한다. (학습 자료와 현장이 같은 방식으로 잘라야 하기 때문)
    """
    n = 50 + C.WIN + 50
    t = (np.arange(n) - 50) / C.FS            # 초 단위 시간. 0초 = 시동 버튼을 누른 순간
    on = t >= 0                                # 시동 전에는 전류가 없다

    # --- 정상 시동의 '평소 흔들림' — 같은 설비도 매번 조금씩 다르다
    peak = rng.normal(6.0, 0.25)               # 돌입 전류 최댓값: 정격의 약 6배
    tau = rng.normal(0.12, 0.01)               # 감쇠 시간 상수(초): 클수록 천천히 줄어든다
    base = rng.normal(1.0, 0.03)               # 안정 전류: 그날의 부하에 따라 조금 다르다
    rise = 0.004                               # 전류가 치솟는 데 걸리는 시간(초)

    # --- 이상 유형별로 위의 값을 바꾼다. 정도는 '약함~심함'을 고르게 섞는다(u: 0~1)
    u = rng.uniform(0, 1)
    if kind == "inrush_high":
        peak *= 1.15 + 0.35 * u                # 돌입 전류 15~50% 증가
    elif kind == "slow_start":
        tau *= 1.5 + 1.5 * u                   # 감쇠가 1.5~3배 느림
    elif kind == "phase_loss":
        base *= 1.4 + 0.4 * u                  # 안정 전류 1.4~1.8배 (남은 두 상에 전류가 몰림)
    elif kind == "stall":
        tau *= 8 + 8 * u                       # 거의 줄지 않음

    if health > 0:                             # 서서히 나빠지는 열화 — 처음엔 거의 티가 안 나다가 점점 빨라진다(제곱)
        tau *= 1 + 1.2 * health ** 2

    tt = np.clip(t, 0, None)
    i = (1 - np.exp(-tt / rise)) * (base + (peak - base) * np.exp(-tt / tau))
    i = np.where(on, i, 0.0)

    if kind == "voltage_dip":                  # 0.6~1.6초 사이 어느 순간 20~80 ms 동안 전류가 20~45% 줄어듦
        start = 50 + int(rng.uniform(0.6, 1.6) * C.FS)
        width = int(rng.uniform(0.02, 0.08) * C.FS)
        i[start:start + width] *= 1 - (0.2 + 0.25 * u)
    elif kind == "ripple":                     # 0.8초 이후 15~30 Hz로 전류가 떨림
        f = rng.uniform(15, 30)
        amp = 0.05 + 0.10 * u
        late = t >= 0.8
        i[late] += amp * np.sin(2 * np.pi * f * t[late])

    if health > 0:                             # 열화에 따른 안정 구간 떨림 (20 Hz, 크기는 열화 정도에 비례)
        late = t >= 0.8
        i[late] += 0.08 * health * np.sin(2 * np.pi * 20 * t[late])

    i += rng.normal(0, 0.02, n)                # 센서 노이즈 (정격의 2%)
    return i.astype(np.float32)


def cut_at_trigger(trace: np.ndarray) -> np.ndarray:
    """전류가 처음으로 TRIGGER를 넘는 샘플부터 WIN개를 잘라 낸다.
    detector.py의 실시간 감지도 똑같은 규칙으로 자른다 → 학습 자료와 현장 자료의 '시작점'이 일치한다.
    """
    k = int(np.argmax(trace > C.TRIGGER))      # 처음으로 넘는 위치
    return trace[k:k + C.WIN]


def make_set(n_per_kind: dict[str, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """유형별 개수만큼 파형을 만들어 (X, y)로 돌려준다. X: (건수, 1000), y: 유형 번호."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for kind, n in n_per_kind.items():
        for _ in range(n):
            X.append(cut_at_trigger(startup_trace(rng, kind)))
            y.append(LABELS.index(kind))
    X, y = np.stack(X), np.array(y)
    order = rng.permutation(len(y))            # 유형별로 몰려 있지 않게 섞는다
    return X[order], y[order]


def make_stream(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """실시간 연습용: '정지 → 시동 → 정지 → 시동 …'이 이어지는 하나의 긴 기록.
    정상 40회 사이에 이상 6종이 2회씩 섞여 있다. 정답(label)은 강사 확인용으로만 함께 저장한다."""
    rng = np.random.default_rng(seed)
    plan = ["normal"] * 40 + [k for k in ANOMALY_TYPES for _ in range(2)]
    rng.shuffle(plan)
    current, label = [], []
    for kind in plan:
        idle = rng.normal(0, 0.01, C.PRE_IDLE).astype(np.float32)   # 정지 상태 (센서 노이즈만)
        tr = startup_trace(rng, kind)
        current += [idle, tr]
        label += [np.full(C.PRE_IDLE, -1), np.full(len(tr), LABELS.index(kind))]
    return np.concatenate(current), np.concatenate(label)


def main() -> None:
    C.DATA_DIR.mkdir(exist_ok=True)

    X, y = make_set({"normal": C.N_TRAIN}, C.SEED_TRAIN)
    np.savez_compressed(C.DATA_DIR / "train.npz", X=X, y=y)

    X, y = make_set({"normal": C.N_VAL}, C.SEED_VAL)
    np.savez_compressed(C.DATA_DIR / "val.npz", X=X, y=y)

    test_plan = {"normal": C.N_TEST_NORMAL} | {k: C.N_TEST_PER_ANOMALY for k in ANOMALY_TYPES}
    X, y = make_set(test_plan, C.SEED_TEST)
    np.savez_compressed(C.DATA_DIR / "test.npz", X=X, y=y, labels=np.array(LABELS))

    cur, lab = make_stream(C.SEED_STREAM)
    t = np.arange(len(cur)) / C.FS
    with open(C.DATA_DIR / "stream_demo.csv", "w", encoding="utf-8") as f:
        f.write("time_s,current_pu,answer\n")   # answer: -1=정지, 0=정상 시동, 1~6=이상 유형 (강사 확인용)
        for a, b, c in zip(t, cur, lab):
            f.write(f"{a:.3f},{b:.4f},{c}\n")

    print("자료 생성 완료:", C.DATA_DIR)
    print(f"  train : 정상 {C.N_TRAIN}건")
    print(f"  val   : 정상 {C.N_VAL}건")
    print(f"  test  : 정상 {C.N_TEST_NORMAL}건 + 이상 {len(ANOMALY_TYPES)}종 × {C.N_TEST_PER_ANOMALY}건")
    print(f"  stream_demo.csv : 시동 {40 + 2 * len(ANOMALY_TYPES)}회, {len(cur)}샘플 ({len(cur) / C.FS:.0f}초 분량)")


if __name__ == "__main__":
    main()
