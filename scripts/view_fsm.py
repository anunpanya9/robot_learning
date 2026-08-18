"""view_fsm.py — เปิดหน้าต่าง MuJoCo viewer ดู FSM หยิบวัตถุใส่ถาดแบบ real-time.

ใช้ดูว่าหุ่นทำงานยังไง (ตรวจ scene/grasp ด้วยตา). ต้องรันบนเครื่องที่มีจอ.

รัน:
    # macOS: ต้องใช้ mjpython (ไม่ใช่ python) — MuJoCo viewer ต้องอยู่ main thread
    PYTHONPATH=src mjpython scripts/view_fsm.py [--seed N] [--episodes K]
    # Linux/Windows: python ปกติได้
    PYTHONPATH=src python scripts/view_fsm.py [--seed N] [--episodes K]

ปุ่ม viewer: ลากเมาส์ = หมุนกล้อง, scroll = ซูม, Esc/ปิดหน้าต่าง = ออก.
ถ้าไม่มีจอ (headless) ให้ใช้ scripts/render_fsm.py แทน (เซฟเป็นวิดีโอ/gif).
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco.viewer
import numpy as np

from robot_learning.collect import ArmIK, FSMPolicy
from robot_learning.env import EnvConfig, SO101PickPlaceEnv


def main() -> None:
    # บน macOS ต้องรันด้วย mjpython — เตือนให้ชัดก่อน error งงๆ.
    # mjpython ตั้ง sys.executable เป็น python ปกติ (เช็คชื่อไม่ได้) แต่จะตั้ง
    # mujoco.viewer._MJPYTHON เป็น object (ไม่ใช่ None) → ใช้ตัวนี้เช็คแทน.
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        print(
            "⚠️  บน macOS ต้องรัน viewer ด้วย mjpython ไม่ใช่ python:\n"
            "    PYTHONPATH=src mjpython scripts/view_fsm.py\n"
            "  (ถ้าอยู่ใน venv: PYTHONPATH=src ./.venv/bin/mjpython scripts/view_fsm.py)\n"
            "  หรือใช้ render_fsm.py เพื่อเซฟเป็น gif แทน (ไม่ต้องเปิดหน้าต่าง)",
            file=sys.stderr,
        )
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    env = SO101PickPlaceEnv(EnvConfig(image_h=64, image_w=64, seed=args.seed))
    ik = ArmIK()
    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, info = env.reset()
            print(f"[ep{ep}] task: {info['task']}")
            fsm = FSMPolicy(ik=ik)
            steps = 0
            while not fsm.done and steps < 500 and viewer.is_running():
                action = fsm.act(obs, env)
                obs, r, term, trunc, info = env.step(action)
                viewer.sync()
                time.sleep(env.cfg.control_dt)  # เดินตามเวลาจริง
                steps += 1
                if term:
                    break
            print(f"[ep{ep}] success={info['is_success']} steps={steps}")
            if not viewer.is_running():
                break
            time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
