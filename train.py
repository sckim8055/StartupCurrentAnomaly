"""
train.py — 2단계: 정상 자료로 모델을 학습하고, 경보선을 정하고, 모델 파일을 저장한다
=====================================================================

[흐름]
  1) train.npz(정상)로 정규화 값(평균·표준편차)을 계산한다.
  2) 오토인코더를 학습한다. 목표: 입력 파형을 최대한 비슷하게 다시 그리기 (복원 오차 최소화).
  3) 매 바퀴(epoch)마다 val.npz(정상)에서 복원 오차를 재고, 가장 좋았던 순간의 가중치를 기억한다.
     val 오차가 PATIENCE 바퀴 동안 좋아지지 않으면 멈춘다. (조기 종료 — 너무 오래 학습해서 train에만 맞춰지는 것을 막음)
  4) 가장 좋았던 모델로 val(정상)의 구간별 점수를 계산 → 상위 0.5% 위치 × 1.2 를 경보선으로 정한다.
  5) 간단한 기준(최대전류·안정전류)의 정상 범위를 train에서 계산한다.
  6) 위의 모든 것을 models/startup_ae.pt 한 파일에 저장하고, 사람이 읽을 수 있는 요약(json)과 학습 곡선도 남긴다.

[train / validation / test 를 왜 나누나?]
  - train      : 모델이 '배우는' 자료. 모델은 이 자료에 점점 맞춰진다.
  - validation : 모델이 직접 배우지 않은 자료로 '언제 멈출지', '경보선을 어디에 둘지' 같은 결정을 내린다.
                 train으로 이런 결정을 하면, 모델이 이미 외운 자료라 오차가 실제보다 작게 나와 경보선이 너무 낮아진다.
  - test       : 모든 결정이 끝난 뒤 '딱 한 번' 성능을 잰다. test를 보고 설정을 다시 고치면, test도 결국 '배운 자료'가 되어
                 성능을 부풀려 보게 된다. 그래서 이 파일은 test.npz를 아예 읽지 않는다.

실행:  python train.py      (CPU로 1~2분, GPU가 있으면 더 빠름)
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import torch

import config as C
from model import StartupAE, normalize, reconstruct, rule_bands, segment_scores


def main() -> None:
    torch.manual_seed(C.SEED_TORCH)           # 같은 결과가 다시 나오도록 난수 씨앗 고정 (재현성)
    C.MODEL_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1) 자료 읽기 + 정규화 값 계산 (train에서만!)
    X_tr = np.load(C.DATA_DIR / "train.npz")["X"]
    X_va = np.load(C.DATA_DIR / "val.npz")["X"]
    mean, std = float(X_tr.mean()), float(X_tr.std())
    Xtr = torch.from_numpy(normalize(X_tr, mean, std))[:, None, :]   # (건수, 1, 1000)
    Xva = torch.from_numpy(normalize(X_va, mean, std))[:, None, :]
    print(f"train {len(X_tr)}건, val {len(X_va)}건 | 정규화: 평균 {mean:.3f}, 표준편차 {std:.3f}")

    # ---------------------------------------------------------------- 2)~3) 학습 + 조기 종료
    model = StartupAE()
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)   # Adam: 가장 널리 쓰는 학습 방법(가중치 갱신 규칙)
    loss_fn = torch.nn.MSELoss()                          # 손실 = 입력과 다시 그린 파형의 평균 제곱 오차

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, C.EPOCHS + 1):
        # (a) 학습: train을 무작위 순서로 BATCH개씩 나눠, 묶음마다 가중치를 조금씩 고친다
        model.train()
        tr_loss = 0.0
        for idx in torch.randperm(len(Xtr)).split(C.BATCH):
            xb = Xtr[idx]
            loss = loss_fn(model(xb), xb)       # 목표가 '입력 자기 자신'인 것이 오토인코더의 특징
            opt.zero_grad()                     # 이전 묶음의 기울기 지우기
            loss.backward()                     # 오차를 줄이는 방향(기울기) 계산
            opt.step()                          # 그 방향으로 가중치 갱신
            tr_loss += loss.item() * len(idx)
        tr_loss /= len(Xtr)

        # (b) 검증: 학습하지 않고 val 오차만 잰다
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xva), Xva).item()
        history.append((epoch, tr_loss, va_loss))

        # (c) 가장 좋았던 순간을 기억 + 조기 종료 판단
        if va_loss < best_val - 1e-6:
            best_val, best_epoch, wait = va_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | train 손실 {tr_loss:.5f} | val 손실 {va_loss:.5f}")
        if wait >= C.PATIENCE:
            print(f"  val 손실이 {C.PATIENCE}바퀴 동안 좋아지지 않아 epoch {epoch}에서 멈춤")
            break
    model.load_state_dict(best_state)           # 마지막이 아니라 '가장 좋았던' 가중치로 되돌린다
    model.eval()
    print(f"학습 끝: {time.time() - t0:.1f}초, 가장 좋은 epoch {best_epoch} (val 손실 {best_val:.5f})")

    # ---------------------------------------------------------------- 4) 경보선 정하기 (validation 정상 자료)
    Xn_va = normalize(X_va, mean, std)
    va_scores = segment_scores(Xn_va, reconstruct(model, Xn_va))
    thresholds = {k: float(np.quantile(v, C.THRESHOLD_QUANTILE) * C.THRESHOLD_MARGIN) for k, v in va_scores.items()}

    # ---------------------------------------------------------------- 5) 간단한 기준의 정상 범위 (train 정상 자료)
    bands = rule_bands(X_tr)

    # ---------------------------------------------------------------- 6) 저장
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "fs": C.FS, "win": C.WIN, "trigger": C.TRIGGER, "segments": C.SEGMENTS,
        "latent": C.LATENT, "best_epoch": best_epoch, "best_val_loss": best_val,
        "n_train": len(X_tr), "n_val": len(X_va),
        "threshold_rule": f"val 상위 {100 * (1 - C.THRESHOLD_QUANTILE):.1f}% 위치 x {C.THRESHOLD_MARGIN}",
        "rule_band_rule": f"train 평균 ± {C.RULE_SIGMA} 표준편차",
    }
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "thresholds": thresholds, "rule_bands": bands, "meta": meta}, C.MODEL_PATH)
    summary = meta | {"mean": mean, "std": std, "thresholds": thresholds, "rule_bands": bands}
    (C.MODEL_DIR / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 학습 곡선: 표(csv)는 항상, 그림(png)은 matplotlib가 있을 때만
    C.REPORT_DIR.mkdir(exist_ok=True)
    with open(C.REPORT_DIR / "training_curve.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        f.writelines(f"{e},{a:.6f},{b:.6f}\n" for e, a, b in history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.array(history)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h[:, 0], h[:, 1], label="train")
        ax.plot(h[:, 0], h[:, 2], label="validation")
        ax.axvline(best_epoch, ls="--", c="gray", label=f"best epoch {best_epoch}")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.legend()
        fig.tight_layout(); fig.savefig(C.REPORT_DIR / "training_curve.png", dpi=120); plt.close(fig)
    except ImportError:
        print("  (matplotlib가 없어 학습 곡선 그림은 건너뜀 — csv는 저장됨)")

    print("저장:", C.MODEL_PATH)
    print("경보선:", {k: round(v, 4) for k, v in thresholds.items()})
    print("규칙 정상 범위:", {k: (round(a, 2), round(b, 2)) for k, (a, b) in bands.items()})


if __name__ == "__main__":
    main()
