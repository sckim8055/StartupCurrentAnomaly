"""
evaluate.py — 3단계: test 자료로 '딱 한 번' 성능을 잰다
=====================================================================

[무엇을 재나]
  - 정상 시동을 정상으로 두는가?   → 오경보율(정상인데 경보가 난 비율). 낮을수록 좋다.
  - 이상 시동을 잡아내는가?       → 유형별 감지율(이상 중 경보가 난 비율). 높을수록 좋다.
  - 모델만 / 규칙만 / 둘 다 쓴 경우를 나눠서 비교한다. → 각각 무엇을 잘 잡고 무엇을 놓치는지 보인다.

[주의]
  - 판정은 detector.judge()로 한다. 현장(run_realtime.py)에서 쓰는 것과 '같은 함수'다.
    평가용 코드를 따로 만들면, 평가와 현장 동작이 조금씩 달라지는 실수가 생기기 쉽다.
  - test 결과를 보고 설정(경보선 등)을 고친 뒤 다시 test로 재면 안 된다. 그러면 test가 '배운 자료'가 된다.
    고치고 싶으면 validation으로 다시 정하고, 새 test 자료(새 seed)로 재야 한다.

출력: reports/test_report.md (표), reports/test_results.csv (건별 점수), reports/examples.png (유형별 예시 그림)
실행:  python evaluate.py
"""
from __future__ import annotations

import numpy as np

import config as C
from detector import judge, load_bundle
from model import normalize, reconstruct
from simulate import ANOMALY_TYPES, LABELS


def rate(flags) -> str:
    flags = list(flags)
    return f"{sum(flags)}/{len(flags)} ({100 * sum(flags) / len(flags):.0f}%)"


def main() -> None:
    bundle = load_bundle()
    d = np.load(C.DATA_DIR / "test.npz")
    X, y = d["X"], d["y"]
    res = judge(bundle, X)

    # ---------------------------------------------------------------- 유형별 표
    lines = ["# 시동 전류 이상 감지 — test 평가 결과", "",
             f"- 모델 학습 시각: {bundle['meta']['trained_at']}",
             f"- test: 정상 {int((y == 0).sum())}건, 이상 {int((y > 0).sum())}건 (유형 {len(ANOMALY_TYPES)}종)",
             f"- 경보선: {bundle['meta']['threshold_rule']} / 규칙 범위: {bundle['meta']['rule_band_rule']}", "",
             "| 실제 상태 | 모델만 | 규칙만 | 둘 중 하나(최종) |", "|---|---|---|---|"]
    for lab in range(len(LABELS)):
        r = [res[i] for i in np.where(y == lab)[0]]
        name = "정상 (← 오경보율)" if lab == 0 else ANOMALY_TYPES[LABELS[lab]][0]
        lines.append(f"| {name} | {rate(x['model_alarm'] for x in r)} | {rate(x['rule_alarm'] for x in r)} "
                     f"| {rate(x['alarm'] for x in r)} |")

    # ---------------------------------------------------------------- 전체 지표
    alarm = np.array([x["alarm"] for x in res])
    is_anom = y > 0
    tp = int((alarm & is_anom).sum()); fp = int((alarm & ~is_anom).sum())
    fn = int((~alarm & is_anom).sum()); tn = int((~alarm & ~is_anom).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    lines += ["", "## 전체 (최종 판정 기준)", "",
              "| | 경보 | 경보 없음 |", "|---|---|---|",
              f"| 실제 이상 | {tp} (맞게 잡음) | {fn} (놓침) |",
              f"| 실제 정상 | {fp} (오경보) | {tn} (맞게 통과) |", "",
              f"- 정밀도(경보 중 진짜 이상 비율): {precision:.3f}",
              f"- 재현율(이상 중 잡아낸 비율): {recall:.3f}",
              f"- F1: {f1:.3f}", "",
              "※ 가짜(시뮬레이션) 자료의 결과다. 실제 현장 성능은 실제 자료로 다시 재야 한다."]
    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---------------------------------------------------------------- 건별 결과 (엑셀에서 열어 볼 수 있게)
    seg = list(C.SEGMENTS)
    with open(C.REPORT_DIR / "test_results.csv", "w", encoding="utf-8-sig") as f:
        f.write("no,actual," + ",".join(f"score_{k}" for k in seg) + ",max_current,steady_current,decay_time_s,model_alarm,rule_alarm,alarm,where\n")
        for i, r in enumerate(res):
            f.write(f"{i},{LABELS[y[i]]}," + ",".join(f"{r['scores'][k]:.5f}" for k in seg)
                    + f",{r['features']['최대전류']:.3f},{r['features']['안정전류']:.3f},{r['features']['감쇠시간']:.3f},"
                      f"{int(r['model_alarm'])},{int(r['rule_alarm'])},{int(r['alarm'])},\"{'; '.join(r['where'])}\"\n")

    # ---------------------------------------------------------------- 그림: 유형별 예시 1건씩 (실제 vs 모델이 다시 그린 것)
    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 글꼴이 없을 때 나는 경고를 숨김
        plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]   # 윈도우 한글 글꼴
        plt.rcParams["axes.unicode_minus"] = False
        t = np.arange(C.WIN) / C.FS
        fig, axes = plt.subplots(len(LABELS), 1, figsize=(9, 2.2 * len(LABELS)), sharex=True)
        for ax, lab in zip(axes, range(len(LABELS))):
            i = int(np.where(y == lab)[0][0])
            xn = normalize(X[i:i + 1], bundle["mean"], bundle["std"])
            rec = reconstruct(bundle["model"], xn)[0] * bundle["std"] + bundle["mean"]   # 정규화를 되돌려 원래 단위로
            ax.plot(t, X[i], lw=1, label="실제 전류")
            ax.plot(t, rec, lw=1, ls="--", label="모델이 다시 그린 정상 모양")
            title = "정상" if lab == 0 else ANOMALY_TYPES[LABELS[lab]][0]
            ax.set_title(f"{title} → {'경보' if res[i]['alarm'] else '정상'} {res[i]['where']}", fontsize=9)
            ax.set_ylabel("p.u.")
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("시동 후 시간(초)")
        fig.tight_layout(); fig.savefig(C.REPORT_DIR / "examples.png", dpi=110); plt.close(fig)
        print("\n그림 저장:", C.REPORT_DIR / "examples.png")
    except ImportError:
        print("\n(matplotlib가 없어 예시 그림은 건너뜀)")


if __name__ == "__main__":
    main()
