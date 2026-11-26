"""
detector.py — 학습된 모델을 '현장에서' 쓰는 부분 (판정기)
=====================================================================

[학습과 적용은 다른 일이다]
  - 학습(train.py): 시간이 걸린다. 사무실 PC에서, 정상 자료가 충분히 모였을 때 가끔 한 번 한다.
  - 적용(이 파일): 학습은 하지 않는다. 저장된 모델 파일(models/startup_ae.pt)을 '불러서 쓰기만' 한다.
    시동 1건 판정에 몇 밀리초면 끝나므로 현장 PC(산업용 PC, 엣지 장비)에서 실시간으로 돌릴 수 있다.

[현장 PC에 가져갈 것]
  config.py, model.py, detector.py, models/startup_ae.pt  — 이 네 개면 된다. (자료·학습 코드는 필요 없다)

[이 파일의 두 가지 사용법]
  1) judge(X)              : 이미 잘라 둔 시동 파형 여러 건을 한꺼번에 판정 (evaluate.py가 사용)
  2) StartupDetector.feed(x): 센서 값이 하나씩 들어올 때마다 호출. 시동을 스스로 찾아 2초 모이면 판정 (run_realtime.py가 사용)
  두 방법은 내부에서 '같은 판정 함수'를 쓴다 → 평가 결과와 현장 동작이 어긋나지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import config as C
from model import StartupAE, normalize, reconstruct, rule_features, segment_scores


def load_bundle(path=C.MODEL_PATH) -> dict:
    """train.py가 저장한 파일을 읽는다. 파일 안에는 다음이 함께 들어 있다:
       - state      : 신경망 가중치(학습 결과)
       - mean, std  : train 자료로 계산한 정규화 값
       - thresholds : 구간별 경보선 (validation 정상 자료로 정함)
       - rule_bands : 간단한 기준의 정상 범위 (train 정상 자료로 정함)
       - meta       : 언제, 어떤 설정으로 학습했는지 (기록용)
    모델과 정규화 값·경보선을 '한 파일'에 묶어 두는 이유: 서로 다른 버전이 섞이는 사고를 막기 위해서다.
    """
    b = torch.load(path, map_location="cpu", weights_only=False)
    model = StartupAE(b["meta"]["latent"])
    model.load_state_dict(b["state"])
    model.eval()                      # 평가 모드: 학습 때만 쓰는 동작(드롭아웃 등)을 끈다
    b["model"] = model
    return b


def judge(bundle: dict, X: np.ndarray) -> list[dict]:
    """시동 파형 X (건수, 1000)를 판정해서, 건마다 결과 사전을 돌려준다.

    결과 예: {"alarm": True, "model_alarm": True, "rule_alarm": False,
             "where": ["안정(0.8~2s)"], "scores": {...}, "features": {...}}
    """
    Xn = normalize(X, bundle["mean"], bundle["std"])
    Rn = reconstruct(bundle["model"], Xn)
    seg = segment_scores(Xn, Rn)       # 모델 점수 (구간별)
    feat = rule_features(X)            # 규칙 숫자 (최대전류, 안정전류)

    results = []
    for i in range(len(X)):
        where = [k for k in seg if seg[k][i] > bundle["thresholds"][k]]          # 모델: 경보선을 넘은 구간
        rule_hits = [f"{k} {feat[k][i]:.2f}" for k, (lo, hi) in bundle["rule_bands"].items()
                     if not lo <= feat[k][i] <= hi]                               # 규칙: 정상 범위를 벗어난 숫자
        results.append({
            "alarm": bool(where or rule_hits),     # 둘 중 하나라도 걸리면 경보
            "model_alarm": bool(where),
            "rule_alarm": bool(rule_hits),
            "where": where + rule_hits,
            "scores": {k: float(v[i]) for k, v in seg.items()},
            "features": {k: float(v[i]) for k, v in feat.items()},
        })
    return results


@dataclass
class StartupDetector:
    """센서 값이 하나씩 들어오는 상황을 위한 판정기.

    사용 예:
        det = StartupDetector()           # 프로그램 시작할 때 한 번 (모델 불러오기)
        while True:
            x = 센서에서_전류_읽기()
            result = det.feed(x)          # 시동 2초가 모이기 전에는 None
            if result and result["alarm"]:
                경보_조치(result)
    """
    path: str = str(C.MODEL_PATH)
    bundle: dict = field(init=False)
    buf: list = field(init=False, default_factory=list)
    recording: bool = field(init=False, default=False)
    armed: bool = field(init=False, default=True)     # True = 다음 시동을 기다리는 중
    n_startups: int = field(init=False, default=0)

    def __post_init__(self):
        self.bundle = load_bundle(self.path)    # 무거운 일(파일 읽기)은 여기서 한 번만 한다

    def feed(self, x: float) -> dict | None:
        # 0) 판정이 끝난 뒤에는 전동기가 계속 돌고 있어 전류가 1.0 근처다. 이것을 새 시동으로 착각하지 않도록,
        #    전류가 REARM 아래로 내려가(= 정지) 다시 '대기' 상태가 될 때까지는 시동을 찾지 않는다.
        #    이렇게 두 기준(TRIGGER 0.5 / REARM 0.2)을 따로 두는 방식을 '히스테리시스'라고 한다.
        if not self.armed and not self.recording and x < C.REARM:
            self.armed = True
        # 1) 대기 중일 때 전류가 TRIGGER를 넘으면 → 시동이 시작된 것. 기록을 시작한다.
        #    (simulate.py의 cut_at_trigger와 같은 규칙. 학습 자료와 같은 위치에서 잘라야 모델이 제대로 비교한다)
        if self.armed and not self.recording and x > C.TRIGGER:
            self.recording, self.armed, self.buf = True, False, []
        # 2) 기록 중이면 값을 모은다.
        if self.recording:
            self.buf.append(x)
            # 3) 2초(WIN개)가 모이면 판정하고, 다음 시동을 기다린다.
            if len(self.buf) == C.WIN:
                self.recording = False
                self.n_startups += 1
                r = judge(self.bundle, np.asarray(self.buf, dtype=np.float32)[None, :])[0]
                r["startup_no"] = self.n_startups
                return r
        return None

    def discard(self) -> None:
        """기록 중이던 시동을 버린다. 네트워크에서 값이 빠졌을 때 쓴다.
        값이 빠진 파형은 모양이 망가져 있어서 판정하면 엉뚱한 경보가 나기 때문이다.
        전동기가 멈출 때까지(REARM) 기다렸다가 다음 시동부터 다시 본다."""
        self.recording, self.armed, self.buf = False, False, []
