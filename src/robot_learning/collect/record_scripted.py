"""record_scripted.py — เก็บ dataset อัตโนมัติด้วย FSM เป็น LeRobotDataset v3.0.

วน N episodes: reset (สุ่มสี/ตำแหน่ง) → FSM หยิบ ball ใส่ถาด → บันทึกทุก frame
(2 กล้อง + joint state + action + task string). เก็บเฉพาะ episode ที่สำเร็จ
(FSM ล้มเหลว/วัตถุไม่เข้าถาด → ทิ้ง) เพื่อคุณภาพ dataset.

Task string ต่อ episode = "pick up the <color> ball and put it in the tray"
(ภาษาอังกฤษ — SmolVLA เข้าใจ; deploy ให้ LLM แปลไทย→อังกฤษก่อน).

รัน:
    PYTHONPATH=src python -m robot_learning.collect.record_scripted \\
        --episodes 300 --repo-id $HF_USER/so101_pickplace_color \\
        --root data/so101_pickplace [--push]

โครงสร้าง observation/action keys (ต้องตรงกับตอน train/deploy SmolVLA):
    observation.images.top    : HxWx3 uint8  (กล้องมองลงโต๊ะ)
    observation.images.wrist  : HxWx3 uint8  (กล้องปลายแขน)
    observation.state         : float32[6]   (joint positions)
    action                    : float32[6]   (joint targets)
    task                      : str          (instruction)

หมายเหตุ: ตอนเก็บ ใช้ image ขนาดที่ตั้งใน EnvConfig (default 256x256). ปรับได้
ด้วย --image-size. ยิ่งเล็ก เก็บเร็ว/ไฟล์เล็ก แต่ VLA เห็นรายละเอียดน้อยลง.
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from robot_learning.collect.fsm_policy import FSMPolicy
from robot_learning.collect.ik import ArmIK
from robot_learning.env import EnvConfig, SO101PickPlaceEnv
from robot_learning.utils.constants import ALL_JOINTS


def build_features(image_h: int, image_w: int) -> dict:
    """feature spec ของ LeRobotDataset (v3.0)."""
    joint_names = list(ALL_JOINTS)
    return {
        "observation.images.top": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": joint_names,
        },
    }


def record(
    episodes: int,
    repo_id: str,
    root: str | None,
    image_size: int,
    fps: int,
    seed: int,
    push: bool,
    max_steps: int,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = build_features(image_size, image_size)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=True,
    )

    env = SO101PickPlaceEnv(
        EnvConfig(image_h=image_size, image_w=image_size, seed=seed, max_steps=max_steps)
    )
    ik = ArmIK()

    recorded = 0
    attempted = 0
    pbar = tqdm(total=episodes, desc="recording")
    while recorded < episodes:
        attempted += 1
        obs, info = env.reset()
        task = info["task"]
        fsm = FSMPolicy(ik=ik)

        # เก็บ frame ของ episode นี้ไว้ก่อน แล้วค่อย commit ถ้าสำเร็จ
        frames: list[dict] = []
        steps = 0
        success = False
        while not fsm.done and steps < max_steps:
            action = fsm.act(obs, env)
            frames.append(
                {
                    "observation.images.top": obs["pixels"]["top"],
                    "observation.images.wrist": obs["pixels"]["wrist"],
                    "observation.state": obs["agent_pos"].astype(np.float32),
                    "action": action.astype(np.float32),
                    "task": task,
                }
            )
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated:
                success = True
                break

        if not success or fsm.failed:
            continue  # ทิ้ง episode ที่ล้มเหลว (ไม่ save)

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        recorded += 1
        pbar.update(1)
        pbar.set_postfix(attempted=attempted, rate=f"{recorded/attempted:.0%}")

    pbar.close()
    env.close()

    dataset.finalize()
    print(f"\nเก็บสำเร็จ {recorded} episodes (จาก {attempted} ครั้ง) → {dataset.root}")

    if push:
        print("กำลัง push ขึ้น HuggingFace Hub...")
        dataset.push_to_hub()
        print(f"push เสร็จ: {repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="เก็บ LeRobot dataset ด้วย scripted FSM")
    ap.add_argument("--episodes", type=int, default=50, help="จำนวน episode ที่สำเร็จ")
    ap.add_argument("--repo-id", type=str, required=True, help="HF repo id เช่น user/so101_pickplace")
    ap.add_argument("--root", type=str, default="data/so101_pickplace", help="โฟลเดอร์เก็บ local")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--push", action="store_true", help="push ขึ้น HF Hub หลังเก็บเสร็จ")
    args = ap.parse_args()

    record(
        episodes=args.episodes,
        repo_id=args.repo_id,
        root=args.root,
        image_size=args.image_size,
        fps=args.fps,
        seed=args.seed,
        push=args.push,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
