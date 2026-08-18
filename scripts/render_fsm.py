"""render_fsm.py — รัน FSM หยิบวัตถุ แล้วเซฟเป็นวิดีโอ (headless ได้).

ใช้ดูผลบนเครื่องที่ไม่มีจอ / แชร์คลิปให้คนอื่นดู. เซฟ 3 มุม: top, wrist,
และมุมมองข้าง (free camera) ต่อกันในเฟรมเดียว.

รัน:
    PYTHONPATH=src python scripts/render_fsm.py --seed 1 --episodes 2 --out out.mp4
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

from robot_learning.collect import ArmIK, FSMPolicy
from robot_learning.env import EnvConfig, SO101PickPlaceEnv


def _side_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.18, 0.10, 0.06]
    cam.distance = 0.65
    cam.azimuth = 120
    cam.elevation = -22
    return cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", type=str, default="fsm_demo.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--subsample", type=int, default=1,
                    help="เก็บทุก N เฟรม (>1 = ไฟล์เล็กลง; gif แนะนำ 4)")
    args = ap.parse_args()

    is_gif = args.out.lower().endswith(".gif")
    # gif: ใช้ resolution เล็ก + subsample เพื่อลดขนาดไฟล์อัตโนมัติ
    h, w = (150, 200) if is_gif else (240, 320)
    if is_gif and args.subsample == 1:
        args.subsample = 4

    env = SO101PickPlaceEnv(EnvConfig(image_h=h, image_w=w, seed=args.seed))
    ik = ArmIK()
    obs, info = env.reset()
    renderer = mujoco.Renderer(env.model, height=h, width=w)
    side_cam = _side_camera(env.model)

    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        print(f"[ep{ep}] task: {info['task']}")
        fsm = FSMPolicy(ik=ik)
        steps = 0
        while not fsm.done and steps < 500:
            action = fsm.act(obs, env)
            obs, r, term, trunc, info = env.step(action)
            if steps % args.subsample == 0:
                # ต่อ 3 มุมในแนวนอน: side | top | wrist
                renderer.update_scene(env.data, camera=side_cam)
                side = renderer.render()
                renderer.update_scene(env.data, camera="top")
                top = renderer.render()
                renderer.update_scene(env.data, camera="wrist")
                wrist = renderer.render()
                frames.append(np.concatenate([side, top, wrist], axis=1))
            steps += 1
            if term:
                break
        print(f"[ep{ep}] success={info['is_success']} steps={steps}")

    if is_gif:
        imageio.mimsave(args.out, frames, fps=max(4, args.fps // 2),
                        loop=0, palettesize=64)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"saved {len(frames)} frames -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
