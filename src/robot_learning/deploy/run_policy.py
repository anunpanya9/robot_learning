"""run_policy.py — โหลด SmolVLA checkpoint แล้วสั่งหุ่นใน MuJoCo ตาม task string.

เป็นตัวกลางระหว่าง policy (SmolVLA) กับ env (SO101PickPlaceEnv): แปลง observation
ของ env → batch ที่ SmolVLA รับ, เรียก select_action, ส่ง action กลับเข้า env.

ใช้ได้ 2 แบบ:
  - CLI เดี่ยว:  python -m robot_learning.deploy.run_policy --checkpoint ... --task "..."
  - เป็น library: PolicyRunner(...).run_task("pick up the red ball ...")
    (voice_pipeline เรียกใช้ตัวนี้)

หมายเหตุ observation mapping — key ต้องตรงกับตอน train (record_scripted.py):
    observation.images.top    ← obs["pixels"]["top"]   (CHW, float32 0..1)
    observation.images.wrist  ← obs["pixels"]["wrist"]
    observation.state         ← obs["agent_pos"]
    task                      ← task string
"""

from __future__ import annotations

import argparse

import numpy as np


class PolicyRunner:
    """โหลด SmolVLA policy + รัน closed-loop ใน env ตาม task string."""

    def __init__(self, checkpoint: str, image_size: int = 256, device: str | None = None):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from robot_learning.env import EnvConfig, SO101PickPlaceEnv

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()

        self.env = SO101PickPlaceEnv(
            EnvConfig(image_h=image_size, image_w=image_size)
        )

    def _obs_to_batch(self, obs: dict, task: str) -> dict:
        """แปลง obs ของ env → batch tensor ที่ SmolVLA รับ (มี batch dim = 1)."""
        torch = self.torch

        def img(x):  # HWC uint8 → CHW float32 [0,1], add batch
            t = torch.from_numpy(x).to(self.device).float() / 255.0
            return t.permute(2, 0, 1).unsqueeze(0)

        state = torch.from_numpy(obs["agent_pos"]).to(self.device).float().unsqueeze(0)
        return {
            "observation.images.top": img(obs["pixels"]["top"]),
            "observation.images.wrist": img(obs["pixels"]["wrist"]),
            "observation.state": state,
            "task": [task],
        }

    def run_task(self, task: str, max_steps: int = 450, verbose: bool = True) -> bool:
        """รัน 1 task บน env ปัจจุบัน (ไม่ reset scene). คืน success."""
        self.policy.reset()
        obs = self.env._get_obs()  # obs ปัจจุบัน (ไม่ rebuild scene)
        info = self.env._get_info()
        success = False
        for _ in range(max_steps):
            batch = self._obs_to_batch(obs, task)
            with self.torch.no_grad():
                action = self.policy.select_action(batch)
            action = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                success = True
                break
        if verbose:
            print(f"  task='{task}' → success={success}")
        return success

    def reset_scene(self):
        """สุ่มฉากใหม่ (เรียกก่อนเริ่มชุด task ใหม่)."""
        return self.env.reset()

    def close(self):
        self.env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path ของ SmolVLA checkpoint")
    ap.add_argument("--task", required=True, help='เช่น "pick up the red ball and put it in the tray"')
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    runner = PolicyRunner(args.checkpoint, image_size=args.image_size)
    succ = 0
    for ep in range(args.episodes):
        runner.reset_scene()
        if runner.run_task(args.task):
            succ += 1
    print(f"SUCCESS: {succ}/{args.episodes}")
    runner.close()


if __name__ == "__main__":
    main()
