# Train SmolVLA บน dataset SO-101 pick&place

คู่มือ finetune **SmolVLA** จาก dataset ที่เก็บด้วย `record_scripted.py`
(LeRobotDataset v3.0) เพื่อให้หุ่นหยิบวัตถุตามสีที่สั่งด้วยภาษา

---

## 1. ข้อกำหนด

- **GPU (CUDA)** — SmolVLA 450M ต้องใช้ GPU. เครื่อง Mac (ไม่มี CUDA) ให้
  **เก็บ dataset บน Mac แล้ว train บน Colab/cloud** (ดู `notebooks/train_smolvla_colab.ipynb`)
- dataset อยู่บน HuggingFace Hub (push ด้วย `--push`) หรือ local (`--root`)
- VRAM: batch 64 ≈ 16GB+. ถ้าน้อย ลด batch + เพิ่ม grad accumulation

## 2. ติดตั้ง (บนเครื่อง train)

```bash
pip install 'lerobot[smolvla]'      # ดึง SmolVLA policy + deps
# หรือ full: pip install 'lerobot[all]'
```

## 3. คำสั่ง train

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=$HF_USER/so101_pickplace_color \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/smolvla_so101 \
  --policy.device=cuda \
  --wandb.enable=true          # (ออปชัน) ดู loss curve
```

พารามิเตอร์สำคัญ:
- `--policy.path=lerobot/smolvla_base` — เริ่มจาก pretrained (finetune)
- `--steps` — 20k เป็นจุดเริ่ม; dataset ใหญ่/หลากหลาย → เพิ่มเป็น 50k–100k
- `--batch_size` — ลดถ้า VRAM ไม่พอ (เช่น 16 + `--gradient_accumulation_steps=4`)

## 4. dataset ต้องมี key อะไรบ้าง (ให้ SmolVLA เข้าใจ)

`record_scripted.py` เขียน key เหล่านี้แล้ว (ตรงกับที่ SmolVLA คาดหวัง):

| key | ความหมาย |
|-----|----------|
| `observation.images.top`   | กล้องมองลงโต๊ะ (RGB) |
| `observation.images.wrist` | กล้องปลายแขน (RGB) |
| `observation.state`        | joint positions [6] |
| `action`                   | joint targets [6] |
| `task`                     | instruction เช่น "pick up the red ball and put it in the tray" |

> **สำคัญ:** task string เป็นภาษาอังกฤษ — คือสิ่งที่ SmolVLA ใช้ condition
> การเก็บต้อง balance ทุกสีเท่าๆ กัน (randomizer ทำ round-robin ให้แล้ว)

## 5. จำนวน dataset ที่แนะนำ

- **pilot** ~50 ep → ทดสอบว่า train ได้ (loss ลด)
- **ใช้งานจริง** ~1,000–2,000 ep กระจาย 5 สีเท่าๆ กัน (เก็บอัตโนมัติ ปล่อยข้ามคืน)

เก็บเพิ่ม:
```bash
python -m robot_learning.collect.record_scripted \
  --episodes 1500 --repo-id $HF_USER/so101_pickplace_color \
  --root data/so101_pickplace --push
```

## 6. หลัง train — deploy

ได้ checkpoint ที่ `outputs/smolvla_so101/checkpoints/last/pretrained_model`
เอาไป deploy ด้วย `robot_learning.deploy.run_policy` (ดู deploy/):

```bash
python -m robot_learning.deploy.run_policy \
  --checkpoint outputs/smolvla_so101/checkpoints/last/pretrained_model \
  --task "pick up the red ball and put it in the tray"
```

## 7. เช็ค dataset ก่อน train

```bash
# ดูสถิติ/visualize dataset
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
d=LeRobotDataset('$HF_USER/so101_pickplace_color'); print(d.meta.info); print('episodes',d.num_episodes)"
```

ตรวจว่า: task strings ครบทุกสี, จำนวน frame ต่อ episode สมเหตุผล (~200-300),
key ครบ 4 อัน (top/wrist/state/action).
