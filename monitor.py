"""
monitor.py — 5단계: 계속 들어오는 신호에 모델을 적용해 (1) 이상을 잡고 (2) 앞으로의 이상을 예측한다
=====================================================================

[두 가지 일]
  1) 감지(detection) — "지금 이 시동이 이상한가?"
     detector.py의 판정기를 그대로 쓴다. 시동마다 즉시 판정한다.
  2) 예측(prediction) — "이대로 가면 언제쯤 경보가 날까?"
     시동마다 '건강 지표'를 기록해 두고, 그 추세를 이어 그려서 경보선에 닿는 시점을 계산한다.
     설비 보전에서 '잔여 수명(RUL) 예측'이라고 부르는 방법의 가장 단순한 형태다.
     건강 지표 두 가지를 나란히 보여 준다 (둘 다 '경보선 대비 비율'로 바꿔서, 1이 되면 경보선):
       (가) 모델 점수비   = 오토인코더 복원 오차 ÷ 경보선
       (나) 감쇠 시간비   = 감쇠 시간 ÷ 감쇠 시간 정상 범위의 위 끝
     실제로 돌려 보면 (가)는 정상 범위 안에서 들쭉날쭉하다가 경보선 근처에서 '갑자기' 뛰어서 미리 알려 주는 힘이 약하고,
     (나)는 열화에 따라 '꾸준히' 올라가서 예측에 훨씬 쓸모가 있다.
     → 강의 포인트: 이상을 '잡는' 데 좋은 지표와 '미리 아는' 데 좋은 지표는 다를 수 있다.
        예측에는 고장 원리(여기서는 회전이 무거워짐 → 감쇠가 느려짐)와 직접 연결된 지표가 유리하다.

[예측이 되는 경우와 안 되는 경우 — 강의 핵심]
  - 된다: 베어링 마모, 윤활 부족처럼 '서서히' 나빠지는 고장. 경보선을 넘기 전부터 점수가 조금씩 올라가는 흔적이 남는다.
  - 안 된다: 순간 전압 강하처럼 '갑자기' 생기는 일. 직전까지 아무 흔적이 없으므로, 이 신호만으로는 미리 알 수 없다.
           이런 일은 '감지'만 가능하다. (다른 설비 기동 일정 같은 다른 정보가 있어야 예측을 시도할 수 있다)
  - 그래서 추세 계산에서는 갑자기 튄 값 하나에 흔들리지 않도록 '최근 5회 중앙값'을 쓴다.

[추세 계산 방법] (자세한 것은 forecast() 주석)
  - 최근 TREND_WINDOW회 시동의 (중앙값으로 다듬은) 지표 값에 2차 곡선을 맞추고, 앞으로 이어 그려 1에 닿는 시점을 찾는다.
  - 올라가는 폭이 잡음보다 분명히 클 때만 예측을 낸다. 아니면 "뚜렷한 추세 없음".
  - 시간 단위는 '받은 샘플 수 ÷ 초당 샘플 수'로 계산한 설비 시간이다. (신호를 10배속으로 받아도 설비 기준 시간으로 나온다)

실행 (창 두 개):
  창1: python monitor.py
  창2: python signal_generator.py --speed 10
  monitor는 신호가 IDLE_STOP초 동안 안 들어오면 스스로 끝내고, 요약과 그림(reports/trend.png)을 남긴다. Ctrl+C로도 끝난다.
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
from datetime import datetime

import numpy as np

import config as C
from detector import StartupDetector

TREND_WINDOW = 80     # 추세를 볼 최근 시동 수 (너무 짧으면 잡음에 흔들리고, 너무 길면 최근 변화에 둔하다)
MIN_POINTS = 40       # 이만큼 쌓이기 전에는 예측하지 않는다
SMOOTH = 1            # 튀는 값 누르기(최근 몇 회 중앙값). 1 = 사용 안 함 — 2차 곡선 맞춤이 이미 잡음을 평균해 주고, 중앙값은 추세를 늦게 따라가게 만든다
MIN_T = 1.5           # 곡선이 올라간 폭이 잡음의 1.5배 이상일 때만 '추세 있음'으로 본다
WARN_WITHIN = 30      # 경보선 도달이 30회 이내로 예상되면 '예측 주의'
SUSTAIN = 3           # 모델 경보가 3회 연속이면 '지속 이상'(= 열화가 경보선을 넘음)으로 본다
IDLE_STOP = 10.0      # 신호가 10초 동안 없으면 종료


def score_ratio(result: dict, thresholds: dict) -> float:
    """(가) 모델 점수비: 구간별 모델 점수를 경보선으로 나눈 값 중 가장 큰 것. 1 이상이면 모델 경보와 같은 뜻이다."""
    return max(result["scores"][k] / thresholds[k] for k in thresholds)


def decay_ratio(result: dict, bands: dict) -> float:
    """(나) 감쇠 시간비: 감쇠 시간 ÷ 정상 범위의 위 끝. 1 이상이면 '감쇠시간' 규칙 경보와 같은 뜻이다."""
    return result["features"]["감쇠시간"] / bands["감쇠시간"][1]


def forecast(idx: list[int], ratio: list[float]) -> dict | None:
    """최근 비율 값의 추세로 '비율이 1(경보선)에 닿을 시동 번호'를 예측한다.

    방법: 최근 TREND_WINDOW회의 값에 2차 곡선 y = a·n² + b·n + c 를 맞추고, 앞으로 이어 그려 y = 1이 되는 n을 찾는다.
      - 직선이 아니라 2차 곡선을 쓰는 이유: 열화는 보통 '갈수록 빨라진다'. 직선으로 이어 그리면 도달 시점을
        실제보다 늦게 예측하는 경향이 있다 (늦은 예측은 위험하다 — 이미 고장 난 뒤에 "곧 고장"이라고 알려 주는 셈).
      - 주의: 이 실습의 신호는 열화를 2차식으로 만들었으므로 2차 곡선이 특히 잘 맞는다.
        실제 설비의 열화 모양은 다를 수 있으므로, 실제 자료로 어떤 곡선이 맞는지 따로 확인해야 한다.
    '추세 있음' 판단: 지금 시점의 기울기가 양수이고, 창 안에서 곡선이 올라간 폭이 잡음(잔차 표준편차)의 MIN_T배보다 클 때.
    반환: 자료가 모자라면 None / {"trend": False, ...} / {"trend": True, "rate": 회당 상승폭, "cross_at": 예측 시동 번호 또는 None}
    """
    if len(ratio) < MIN_POINTS:
        return None
    x = np.asarray(idx[-TREND_WINDOW:], dtype=float)
    y = np.asarray(ratio[-TREND_WINDOW:], dtype=float)
    y = np.array([np.median(y[max(0, i - SMOOTH + 1):i + 1]) for i in range(len(y))])   # 튀는 값 누르기: 최근 5회 중앙값
    x0 = x[-1]
    coef = np.polyfit(x - x0, y, 2)            # 계산이 안정되도록 '지금'을 0으로 놓고 맞춘다
    fit = np.polyval(coef, x - x0)
    noise = float(np.std(y - fit)) + 1e-9
    rate = float(coef[1])                      # 지금 시점(x - x0 = 0)의 기울기 = 회당 상승폭
    rise = float(fit[-1] - fit[0])             # 창 안에서 곡선이 올라간 폭
    if rate <= 0 or rise < MIN_T * noise:
        return {"trend": False, "rate": rate}
    roots = np.roots(coef - np.array([0.0, 0.0, 1.0]))   # a·d² + b·d + c = 1 의 해 (d = 지금부터 몇 회 뒤)
    ahead = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if fit[-1] >= 1:
        return {"trend": True, "rate": rate, "cross_at": x0}            # 이미 경보선 위
    return {"trend": True, "rate": rate, "cross_at": x0 + min(ahead) if ahead else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="받을 주소. 같은 PC면 127.0.0.1, 다른 PC에서 보내는 신호를 받으려면 0.0.0.0 (윈도우 방화벽 허용 창이 뜰 수 있음)")
    args = ap.parse_args()

    det = StartupDetector()
    thr, bands = det.bundle["thresholds"], det.bundle["rule_bands"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))      # 이 주소의 5005번 포트로 오는 신호를 받는다
    sock.settimeout(IDLE_STOP)
    print(f"모델: {C.MODEL_PATH.name} (학습 {det.bundle['meta']['trained_at']}) | 포트 {args.port}에서 신호 기다리는 중")

    C.LOG_DIR.mkdir(exist_ok=True)
    log_path = C.LOG_DIR / f"monitor_{datetime.now():%Y%m%d_%H%M%S}.csv"
    lf = open(log_path, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(lf)
    w.writerow(["startup_no", "equip_time_s", "alarm", "where", "model_ratio", "model_pred_cross_at",
                "decay_time_s", "decay_ratio", "decay_rate_per_startup", "decay_pred_cross_at"])

    idx, alarms = [], []
    ind = {"모델": [], "감쇠": []}             # 건강 지표 두 가지의 시동별 값
    preds = {"모델": {}, "감쇠": {}}           # 시동 번호 → 그 시점에 낸 예측
    last_seq, lost, n_samples, start_sample = None, 0, 0, {}
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                print(f"\n{IDLE_STOP:.0f}초 동안 신호가 없어 종료합니다.")
                break
            msg = json.loads(data)
            # ---- 빠진 묶음 확인: seq 번호가 건너뛰면 그 사이 값이 빠진 것 → 기록 중이던 시동은 버린다
            if last_seq is not None and msg["seq"] != last_seq + 1:
                lost += msg["seq"] - last_seq - 1
                if det.recording:
                    det.discard()
                    print("  (신호 일부가 빠져 이번 시동은 판정하지 않음)")
            last_seq = msg["seq"]

            for x in msg["samples"]:
                n_samples += 1
                was_recording = det.recording
                result = det.feed(x)                       # ← 감지: 값 하나씩 넣기
                if det.recording and not was_recording:
                    start_sample[det.n_startups + 1] = n_samples   # 시동 시작 시각(샘플 번호) 기록
                if result is None:
                    continue

                # ---------------- 1) 감지 결과
                n = result["startup_no"]
                idx.append(n); alarms.append(result["alarm"])
                ind["모델"].append(score_ratio(result, thr))
                ind["감쇠"].append(decay_ratio(result, bands))

                # ---------------- 2) 예측 (지표마다)
                fcs = {k: forecast(idx, v) for k, v in ind.items()}
                for k, fc in fcs.items():
                    preds[k][n] = fc
                equip_t = start_sample.get(n, n_samples) / C.FS
                cyc = (equip_t / max(n - 1, 1)) if n > 1 else None   # 시동 한 주기 평균 길이(초, 설비 시간)

                state = ("경보 " + ", ".join(result["where"])) if result["alarm"] else "정상"
                line = f"시동 {n:3d} | {state:28.28s} | 모델 {ind['모델'][-1]:6.2f} | 감쇠 {ind['감쇠'][-1]:5.2f}"
                fc = fcs["감쇠"]                    # 화면에는 예측력이 좋은 감쇠 지표의 예측을 보여 준다 (둘 다 기록 파일에는 남김)
                if fc and fc["trend"]:
                    if fc["cross_at"] is None:
                        line += " | 오르는 중이나 당분간 경보선 도달 없음"
                    elif fc["cross_at"] > n:
                        remain = fc["cross_at"] - n
                        when = f"약 {remain:.0f}회 후" + (f"(설비 시간 {remain * cyc / 60:.1f}분)" if cyc else "")
                        tag = "  ◆ 예측 주의" if remain <= WARN_WITHIN else ""
                        line += f" | 감쇠 지표 상승 중 → 경보선 도달 {when}{tag}"
                    else:
                        line += " | 감쇠 지표 이미 경보선 위"
                elif fc:
                    line += " | 뚜렷한 추세 없음"
                print(line)
                fm = fcs["모델"]
                w.writerow([n, f"{equip_t:.1f}", int(result["alarm"]), "; ".join(result["where"]),
                            f"{ind['모델'][-1]:.4f}", f"{fm['cross_at']:.1f}" if fm and fm["trend"] and fm["cross_at"] is not None else "",
                            f"{result['features']['감쇠시간']:.3f}", f"{ind['감쇠'][-1]:.4f}",
                            f"{fc['rate']:.4f}" if fc else "", f"{fc['cross_at']:.1f}" if fc and fc["trend"] and fc["cross_at"] is not None else ""])
                lf.flush()
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        lf.close()
        sock.close()

    summarize(idx, ind, alarms, preds, lost, log_path)


def summarize(idx, ind, alarms, preds, lost, log_path) -> None:
    """끝난 뒤: 예측이 실제로 맞았는지 되돌아본다 (강의용 확인)."""
    print(f"\n===== 요약 =====\n판정한 시동 {len(idx)}회, 경보 {sum(alarms)}회, 빠진 신호 묶음 {lost}개")
    print("기록:", log_path)
    if not idx:
        return
    # 지표마다: '지속 이상' = 그 지표가 SUSTAIN회 연속 1 이상인 첫 시점 (한 번 튄 값과 구별)
    #            그보다 40·30·20·10·5회 전에 낸 예측이 실제와 얼마나 맞았나
    firsts = {}
    for k, v in ind.items():
        first = next((idx[i] for i in range(len(idx) - SUSTAIN + 1) if all(r >= 1 for r in v[i:i + SUSTAIN])), None)
        firsts[k] = first
        print(f"\n[{k} 지표] ", end="")
        if first is None:
            print(f"{SUSTAIN}회 연속 경보선 위인 구간 없음 → 확인할 대상 없음")
            continue
        print(f"{SUSTAIN}회 연속 경보선 위 시작: 시동 {first}. 그 전에 낸 예측:")
        for before in (40, 30, 20, 10, 5):
            fc = preds[k].get(first - before)
            if fc is None:
                continue
            if fc["trend"] and fc["cross_at"] is not None:
                print(f"  {before:2d}회 전(시동 {first - before:3d}): 시동 {fc['cross_at']:.0f} 무렵 예측 → 실제 {first} (오차 {fc['cross_at'] - first:+.0f}회)")
            elif fc["trend"]:
                print(f"  {before:2d}회 전(시동 {first - before:3d}): 오르는 중이나 도달 시점 계산 안 됨")
            else:
                print(f"  {before:2d}회 전(시동 {first - before:3d}): 아직 뚜렷한 추세 없음")
    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        a = np.array(alarms)
        for ax, (k, v) in zip(axes, ind.items()):
            v = np.array(v)
            ax.semilogy(idx, v, ".", ms=4, label=f"{k} 지표 (경보선 대비 비율)")
            ax.semilogy(np.array(idx)[a], v[a], "rx", ms=5, label="경보가 난 시동")
            ax.axhline(1, c="r", lw=1, ls="--", label="경보선 (= 1)")
            first = firsts[k]
            if first:
                ax.axvline(first, c="gray", ls=":", label=f"{SUSTAIN}회 연속 경보선 위 ({first})")
                for before, c in ((30, "tab:green"), (10, "tab:purple")):
                    fc = preds[k].get(first - before)
                    if fc and fc["trend"] and fc["cross_at"] is not None:
                        ax.axvline(fc["cross_at"], c=c, lw=1, label=f"{before}회 전의 예측 ({fc['cross_at']:.0f})")
            ax.set_ylabel("비율 (로그 눈금)"); ax.legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("시동 번호")
        fig.tight_layout()
        C.REPORT_DIR.mkdir(exist_ok=True)
        fig.savefig(C.REPORT_DIR / "trend.png", dpi=110)
        print("그림:", C.REPORT_DIR / "trend.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
