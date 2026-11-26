"""
signal_generator.py — '현장 센서' 흉내: 시동 전류 신호를 끊임없이 보낸다
=====================================================================

[역할]
  실제 현장의 전류 센서(또는 PLC)처럼, 전동기가 '정지 → 시동 → 운전 → 정지'를 반복하는 동안의
  전류 값을 쉬지 않고 네트워크(UDP)로 보낸다. 받는 쪽(monitor.py)이 켜져 있든 아니든 계속 보낸다.
  → 실제 센서도 받는 쪽 사정과 상관없이 값을 내보낸다.

[시나리오: '서서히 나빠지는 설비' + '가끔 갑자기 생기는 일']
  1) 처음 WARMUP회 시동은 새 설비 (열화 0)
  2) 그 뒤 베어링이 서서히 닳는다: 열화 정도(health)가 0 → 1로 시동마다 조금씩 커진다
     (처음엔 거의 티가 안 나다가 점점 빨라진다 — simulate.py의 startup_trace 참고)
  3) 열화가 1에 도달하면 시동 실패(구속)가 난다 = 고장
  4) 이와 별개로, 시동마다 SUDDEN_PROB 확률로 '순간 전압 강하'가 섞인다 (다른 설비 탓 — 미리 알 수 없는 일)
  ※ 이 시나리오의 정답(열화 정도)은 이 창에만 보인다. monitor.py는 전류 값만 받는다.

[네트워크]
  UDP(127.0.0.1:5005)로 50샘플(0.1초 분량)씩 묶어서 보낸다. 묶음 하나 = JSON 한 줄:
    {"seq": 묶음 번호, "fs": 500, "samples": [전류 50개]}
  - UDP는 받는 쪽이 없어도 에러 없이 보낼 수 있고, 가볍다. 대신 묶음이 가끔 빠질 수 있다(같은 PC 안에서는 거의 없음).
    빠지면 안 되는 환경이면 TCP나 MQTT를 쓴다. monitor.py는 seq 번호로 빠진 묶음을 알아챈다.
  - 다른 PC로 보내려면 --host 에 받는 PC의 IP를 넣는다.

실행:
  python signal_generator.py               # 실제 속도 (시동 1회 ≈ 2.5초), Ctrl+C로 멈춤
  python signal_generator.py --speed 10    # 10배 빠르게 (강의 시연용: 고장까지 약 1분)
  python signal_generator.py --speed 10 --max-startups 220   # 220회 시동 후 스스로 끝
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

import config as C
from simulate import startup_trace

WARMUP = 60           # 처음 60회는 새 설비
DEGRADE_OVER = 140    # 그 뒤 140회에 걸쳐 열화 0 → 1 (= 200회째 시동 무렵 고장)
SUDDEN_PROB = 0.03    # 시동마다 3% 확률로 순간 전압 강하
CHUNK = 50            # 한 번에 보내는 샘플 수 (0.1초 분량)


def health_at(n: int) -> float:
    """n번째 시동의 열화 정도 (0~1)."""
    return float(np.clip((n - WARMUP) / DEGRADE_OVER, 0, 1))


def cycles(rng: np.random.Generator, max_startups: int | None):
    """시동 한 주기씩(정지 구간 + 시동 기록) 끝없이 만들어 내는 제너레이터."""
    n = 0
    while max_startups is None or n < max_startups:
        n += 1
        h = health_at(n)
        if h >= 1:
            kind = "stall"                                   # 고장: 시동 실패
        elif rng.uniform() < SUDDEN_PROB:
            kind = "voltage_dip"                             # 갑자기 생기는 일
        else:
            kind = "normal"
        idle = rng.normal(0, 0.01, C.PRE_IDLE).astype(np.float32)
        yield n, h, kind, np.concatenate([idle, startup_trace(rng, kind, health=h if kind != "stall" else 0.0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속 (10 = 10배 빠르게)")
    ap.add_argument("--max-startups", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # UDP 소켓
    rng = np.random.default_rng(args.seed)
    chunk_sec = CHUNK / C.FS / args.speed                    # 묶음 하나를 보낼 때마다 기다릴 시간
    seq = 0
    next_t = time.perf_counter()
    print(f"신호 보내는 중 → {args.host}:{args.port} (배속 {args.speed}x). Ctrl+C로 멈춤")
    try:
        for n, h, kind, trace in cycles(rng, args.max_startups):
            print(f"[센서] 시동 {n:3d} | 실제 열화 {h:4.2f} | {kind}")   # 정답 — 강사만 보는 창
            for k in range(0, len(trace), CHUNK):
                msg = {"seq": seq, "fs": C.FS, "samples": [round(float(v), 4) for v in trace[k:k + CHUNK]]}
                sock.sendto(json.dumps(msg).encode(), (args.host, args.port))
                seq += 1
                next_t += chunk_sec                          # 정해진 속도를 지키도록 '다음 보낼 시각'을 계산
                time.sleep(max(0.0, next_t - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    print("신호 보내기 끝")


if __name__ == "__main__":
    main()
