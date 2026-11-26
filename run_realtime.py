"""
run_realtime.py — 4단계: 학습된 모델을 '실시간'으로 적용하는 예
=====================================================================

[무엇을 흉내 내나]
  - 현장에서는 센서 값이 끊임없이 하나씩 들어온다. 이 파일은 data/stream_demo.csv를
    '센서'처럼 한 줄씩 읽어서 판정기(StartupDetector)에 넣는다.
  - 판정기는 스스로 시동 시작을 찾고(전류 > TRIGGER), 2초가 모이면 판정한다.
  - 경보가 나면 act()가 조치한다. 여기서는 화면 출력 + logs/ 폴더에 기록만 한다.

[실제 현장에 붙일 때 바꿀 곳은 두 군데뿐]
  1) sensor_stream() : csv 대신 실제 센서를 읽도록 바꾼다. (아래 주석에 예시)
  2) act()           : 화면 출력 대신 경보등·메신저 알림·PLC 신호 등으로 바꾼다.
  판정 부분(detector.py)과 모델 파일은 그대로 쓴다.

실행:
  python run_realtime.py              # 최대한 빠르게 재생 (수 초)
  python run_realtime.py --realtime   # 실제 시간 속도로 재생 (1초에 500샘플, 약 2분) — 강의 시연용
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime

import config as C
from detector import StartupDetector
from simulate import ANOMALY_TYPES, LABELS


def sensor_stream(realtime: bool = False):
    """센서 값을 하나씩 내보낸다 (파이썬 '제너레이터': for 문에서 값을 하나씩 꺼내 쓸 수 있다).

    지금은 csv를 읽는다. answer 열(정답)은 강의에서 판정이 맞았는지 확인하려고 함께 내보낼 뿐,
    판정기에는 전류 값만 들어간다. 실제 현장에는 정답이 없다.

    ── 실제 센서로 바꾸는 예 (참고용, 장비마다 주소·방식이 다르다) ──
      # (가) PLC에서 Modbus TCP로 읽기 — pip install pymodbus
      # from pymodbus.client import ModbusTcpClient
      # plc = ModbusTcpClient("192.168.0.10"); plc.connect()
      # while True:
      #     reg = plc.read_holding_registers(100, count=1).registers[0]   # 예: 0.01 A 단위로 저장된 전류
      #     amp = reg * 0.01
      #     yield amp / 정격전류_A, None                                    # p.u.로 바꿔서 내보냄
      #     time.sleep(1 / C.FS)
      # (나) 데이터 수집 장치(DAQ)를 쓰면 보통 '여러 샘플 묶음'으로 읽어 오므로, 묶음 안의 값을 하나씩 yield 한다.
      # 주의: 샘플링 속도(FS)와 전류 단위(p.u.)가 학습 때와 같아야 한다.
    """
    with open(C.DATA_DIR / "stream_demo.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield float(row["current_pu"]), int(row["answer"])
            if realtime:
                time.sleep(1 / C.FS)


def act(result: dict, answer: int, log_writer) -> None:
    """판정 결과에 따른 조치. 실제 현장에서는 이 함수를 바꾼다.
      예) 경보등 켜기, 담당자 메신저 알림, 정비 요청 등록,
          심한 경우 PLC에 '다음 시동 금지' 비트 쓰기 (단, 안전 정지는 기존 보호 장치가 맡고 AI는 '조기 경보' 역할)
    """
    now = datetime.now().strftime("%H:%M:%S")
    status = "경보" if result["alarm"] else "정상"
    truth = "정상" if answer == 0 else ANOMALY_TYPES[LABELS[answer]][0]
    where = ", ".join(result["where"]) or "-"
    print(f"[{now}] 시동 {result['startup_no']:3d} | 판정: {status} | 이상 위치: {where:40s} | (정답: {truth})")
    log_writer.writerow([now, result["startup_no"], status, where, truth])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtime", action="store_true", help="실제 시간 속도로 재생")
    args = ap.parse_args()

    det = StartupDetector()                        # 모델 불러오기: 프로그램 시작 때 한 번만
    print(f"모델 불러옴: {C.MODEL_PATH.name} (학습 {det.bundle['meta']['trained_at']})")
    C.LOG_DIR.mkdir(exist_ok=True)
    log_path = C.LOG_DIR / f"alarms_{datetime.now():%Y%m%d}.csv"   # 날짜별 기록 파일
    new = not log_path.exists()

    hits = {}                                      # 강의용 집계: 정답 유형별로 경보가 몇 번 났나
    t0 = time.perf_counter()
    n_samples = 0
    with open(log_path, "a", newline="", encoding="utf-8-sig") as lf:
        w = csv.writer(lf)
        if new:
            w.writerow(["time", "startup_no", "status", "where", "answer"])
        last_answer = -1
        for x, answer in sensor_stream(args.realtime):
            n_samples += 1
            if answer >= 0:
                last_answer = answer               # 지금 기록 중인 시동의 정답 (강의 확인용)
            result = det.feed(x)                   # ← 실시간 적용의 핵심: 값 하나 넣기
            if result is not None:                 # 시동 1건 판정 완료
                act(result, last_answer, w)
                hits.setdefault(last_answer, []).append(result["alarm"])

    dt = time.perf_counter() - t0
    print(f"\n처리: 샘플 {n_samples}개, 시동 {det.n_startups}건, {dt:.2f}초")
    print("정답 유형별 경보 수:")
    for lab in sorted(hits):
        name = "정상(오경보)" if lab == 0 else ANOMALY_TYPES[LABELS[lab]][0]
        print(f"  {name:12s} {sum(hits[lab])}/{len(hits[lab])}")
    print("기록 파일:", log_path)


if __name__ == "__main__":
    main()
