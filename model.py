"""
model.py — 2단계에서 쓰는 '모델'과 '점수 계산'을 정의한다
=====================================================================

[핵심 아이디어: 오토인코더(AutoEncoder)로 이상 감지]
  - 오토인코더는 입력을 작게 압축(인코더)했다가 다시 원래 크기로 그려 내는(디코더) 신경망이다.
      파형 1000개 값 → [인코더] → 숫자 16개 → [디코더] → 파형 1000개 값
  - 정상 파형만으로 학습하면, 모델은 "정상 파형을 16개 숫자로 요약하는 법"만 배운다.
  - 그래서 처음 보는 이상 파형이 들어오면 제대로 다시 그리지 못한다.
      → 입력과 다시 그린 파형의 차이(복원 오차)가 크면 = 이상
  - 정답(이상 표시)이 없는 정상 자료만으로 학습할 수 있다는 것이 큰 장점이다. 현장에는 정상 자료가 훨씬 많기 때문이다.
  - 참고: 원래 연구의 BIVA 모델도 같은 계열(VAE, 변분 오토인코더)이다. 여기서는 강의용으로 작고 빠른 모델을 쓴다.

[왜 간단한 기준(규칙)도 함께 쓰나?]
  - 모델이 모든 이상을 잘 잡는 것은 아니다. 예를 들어 돌입 전류는 원래 매번 크게 흔들리므로,
    모델은 돌입 전류가 조금 큰 것을 '흔한 차이'로 보고 놓칠 수 있다.
  - "최대 전류가 정상 범위를 벗어났다" 같은 한 줄짜리 규칙은 이런 경우를 쉽게 잡는다.
  - 현장에서는 보통 둘을 함께 쓴다: 모델 = 복잡한 모양 변화, 규칙 = 크기 변화.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import config as C


class StartupAE(nn.Module):
    """1차원 합성곱(Conv1d) 오토인코더.

    합성곱은 '짧은 구간의 모양'을 보는 필터다. 파형처럼 이웃한 값끼리 관련이 큰 자료에 잘 맞는다.
    입력 모양: (건수, 1채널, 1000샘플)
    """

    def __init__(self, latent: int = C.LATENT):
        super().__init__()
        # 인코더: 1000 → 250 → 50 (stride 만큼 길이가 줄어든다), 채널 1 → 8 → 16, 마지막에 숫자 latent개로 요약
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=4, padding=4),   # (1, 1000) → (8, 250)
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=5, padding=4),  # (8, 250) → (16, 50)
            nn.ReLU(),
            nn.Flatten(),                                          # (16, 50) → 800
            nn.Linear(16 * 50, latent),                            # 800 → 16  ← 병목(요약)
        )
        # 디코더: 인코더를 거꾸로 — 16 → 800 → (16, 50) → (8, 250) → (1, 1000)
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16 * 50),
            nn.ReLU(),
            nn.Unflatten(1, (16, 50)),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=5),    # (16, 50) → (8, 250)
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=4),     # (8, 250) → (1, 1000)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ----------------------------------------------------------------------------- 정규화
# 신경망은 값의 크기가 0 근처일 때 학습이 잘 된다. 그래서 (값 - 평균) / 표준편차 로 바꿔서 넣는다.
# 중요: 평균·표준편차는 반드시 'train 자료에서만' 계산하고, val·test·현장에서도 그 값을 그대로 쓴다.
#       (test 자료로 평균을 다시 계산하면, 평가에 미래 정보가 섞이는 셈이 된다)
# 파형 전체에 하나의 평균·표준편차를 쓴다. 샘플 위치마다 따로 정규화하면 원래 크게 흔들리는 돌입 구간의
# 차이가 작아 보이게 되어, 돌입 전류 이상을 놓치기 쉽다.
def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


@torch.no_grad()   # 점수 계산(예측)에는 학습이 필요 없으므로 기울기 계산을 끈다 → 빠르고 메모리도 적게 쓴다
def reconstruct(model: StartupAE, Xn: np.ndarray) -> np.ndarray:
    """정규화된 파형들(건수, 1000)을 모델에 넣어 다시 그린 파형(건수, 1000)을 돌려준다."""
    x = torch.from_numpy(Xn)[:, None, :]       # (건수, 1000) → (건수, 1, 1000)
    return model(x)[:, 0, :].numpy()


def segment_scores(Xn: np.ndarray, Rn: np.ndarray) -> dict[str, np.ndarray]:
    """구간별 이상 점수 = 그 구간의 평균 제곱 오차(MSE).
    구간을 나누는 이유: '어디가' 이상한지 알면 원인을 좁히기 쉽다 (돌입 구간이면 전원·접점, 안정 구간이면 기계 쪽 등)."""
    err = (Xn - Rn) ** 2
    return {name: err[:, a:b].mean(axis=1) for name, (a, b) in C.SEGMENTS.items()}


# ----------------------------------------------------------------------------- 간단한 기준(규칙)
def decay_time(X: np.ndarray) -> np.ndarray:
    """감쇠 시간(초): 최댓값에서 출발해 전류가 '안정 전류 + (최댓값-안정 전류) × 37%' 아래로 처음 내려갈 때까지 걸린 시간.
    37%(= 1/e)는 지수 감쇠의 '시간 상수'를 재는 표준적인 기준이다. 정상은 약 0.12초.
    회전이 무거워지면(베어링 마모, 부하 증가) 이 시간이 길어진다 → 서서히 나빠지는 고장의 '건강 지표'로 쓰기 좋다."""
    out = np.empty(len(X))
    for i, w in enumerate(X):
        k = int(w.argmax())
        steady = w[-C.FS // 2:].mean()
        level = steady + (w[k] - steady) / np.e
        below = np.nonzero(w[k:] < level)[0]
        out[i] = (below[0] if len(below) else len(w) - k) / C.FS   # 끝까지 안 내려가면(구속) 남은 길이 전체
    return out


def rule_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """규칙에 쓰는 세 가지 숫자. X는 정규화 전의 원래 전류(p.u.)."""
    return {
        "최대전류": X.max(axis=1),                    # 돌입 전류 최댓값
        "안정전류": X[:, -C.FS // 2:].mean(axis=1),   # 마지막 0.5초 평균 = 안정된 뒤의 전류
        "감쇠시간": decay_time(X),                     # 전류가 줄어드는 데 걸리는 시간(초)
    }


def rule_bands(X_train: np.ndarray) -> dict[str, tuple[float, float]]:
    """train(정상)에서 각 숫자의 정상 범위 = 평균 ± RULE_SIGMA × 표준편차."""
    out = {}
    for k, v in rule_features(X_train).items():
        m, s = float(v.mean()), float(v.std())
        out[k] = (m - C.RULE_SIGMA * s, m + C.RULE_SIGMA * s)
    return out
