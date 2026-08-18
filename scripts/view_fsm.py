"""view_fsm.py — เปิดหน้าต่าง MuJoCo viewer ดู FSM หยิบวัตถุใส่ถาดแบบ real-time.

ใช้ดูว่าหุ่นทำงานยังไง (ตรวจ scene/grasp ด้วยตา). ต้องรันบนเครื่องที่มีจอ.

รัน:
    PYTHONPATH=src python scripts/view_fsm.py [--seed N] [--episodes K]

ปุ่ม viewer: ลากเมาส์ = หมุนกล้อง, scroll = ซูม, Esc/ปิดหน้าต่าง = ออก.
ถ้าไม่มีจอ (headless) ให้ใช้ scripts/render_fsm.py แทน (เซฟเป็นวิดีโอ).
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from robot_learning.collect import ArmIK, FSMPolicy
from robot_learning.env import EnvConfig, SO101PickPlaceEnv


def main() -> None:
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
