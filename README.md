# robot_learning — VLA Pick & Place บน SO-101 (MuJoCo → LeRobot → SmolVLA → Voice)

ระบบ **Vision-Language-Action** สั่งหุ่นแขนกล **SO-101 (SO-ARM101)** ด้วยเสียง/ภาษา
ให้หยิบลูกบอลตามสีใส่ถาด ทำงานทั้งหมดใน **MuJoCo simulation**

```
เสียงพูด (ไทย/อังกฤษ) → Whisper → Ollama LLM → task string
   → SmolVLA policy → หุ่น SO-101 หยิบลูกบอลตามสีใส่ถาด (MuJoCo)
```

สถานะปัจจุบัน: **Phase 1 = ลูกบอล (ball) 5 สี** — FSM เก็บ dataset สำเร็จ 100%
(cube เป็น Phase 2 — jaw ทรงโค้งคีบมุม cube ไม่เสถียร ดู [docs/GRASP.md](docs/GRASP.md))

![FSM หยิบลูกบอลเขียวใส่ถาด](docs/media/fsm_demo.gif)

> ตัวอย่าง FSM หยิบ "ลูกบอลเขียว" ใส่ถาดอัตโนมัติ (3 มุม: มุมข้าง | กล้อง top | กล้อง wrist)
> — สองกล้องขวาคือ observation ที่ป้อนให้ SmolVLA. สร้าง gif ใหม่ได้ด้วย
> `PYTHONPATH=src python scripts/render_fsm.py --out demo.gif`

---

## Pipeline 4 เฟส

| เฟส | ทำอะไร | ไฟล์หลัก |
|-----|--------|----------|
| **A. Sim** | MuJoCo scene: SO-101 + ลูกบอล 5 สี + ถาด + 2 กล้อง (top/wrist) | [`env/`](src/robot_learning/env/) |
| **B. เก็บ Dataset** | FSM (mink IK + grasp-assist) หยิบอัตโนมัติ วนสุ่ม → LeRobotDataset v3.0 | [`collect/`](src/robot_learning/collect/) |
| **C. Train** | finetune SmolVLA จาก `lerobot/smolvla_base` (Colab/GPU) | [`train/`](src/robot_learning/train/), [`notebooks/`](notebooks/) |
| **D. Deploy + Voice** | mic → Whisper → Ollama → SmolVLA → หุ่น | [`deploy/`](src/robot_learning/deploy/) |

---

## ติดตั้ง

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core + voice (บน Mac ก็ได้)
# สำหรับเก็บ dataset ต้องมี lerobot[dataset]:
pip install 'lerobot[dataset]'
# (train ต้องมี GPU → ดู notebooks/train_smolvla_colab.ipynb)
```

Asset ของ SO-101 (MJCF + meshes) อยู่ใน [`assets/so101/`](assets/so101/) แล้ว
(ดึงจาก [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100))

---

## รันแต่ละเฟส

### A. ดูฉาก/หุ่นทำงาน (ต้องมีจอ)
```bash
PYTHONPATH=src python scripts/view_fsm.py --seed 0 --episodes 3   # viewer สด
PYTHONPATH=src python scripts/render_fsm.py --seed 1 --out demo.gif  # เซฟวิดีโอ (headless)
```

### B. เก็บ Dataset
```bash
# pilot 50 ep (ทดสอบ)
PYTHONPATH=src python -m robot_learning.collect.record_scripted \
  --episodes 50 --repo-id $HF_USER/so101_pickplace_color --root data/so101

# เต็ม ~1500 ep (balance 5 สี, ปล่อยข้ามคืน) + push HF Hub
PYTHONPATH=src python -m robot_learning.collect.record_scripted \
  --episodes 1500 --repo-id $HF_USER/so101_pickplace_color --root data/so101 --push
```

### C. Train (Colab/GPU)
เปิด [`notebooks/train_smolvla_colab.ipynb`](notebooks/train_smolvla_colab.ipynb) หรือดู
[`train/train_smolvla.md`](src/robot_learning/train/train_smolvla.md):
```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=$HF_USER/so101_pickplace_color \
  --batch_size=64 --steps=20000 --output_dir=outputs/smolvla_so101 --policy.device=cuda
```

### D. สั่งงานด้วยเสียง
```bash
# พูดจริง (ต้องมี mic + Ollama รันอยู่ + ต่อ checkpoint)
PYTHONPATH=src python -m robot_learning.deploy.voice_pipeline \
  --checkpoint outputs/smolvla_so101/checkpoints/last/pretrained_model

# ทดสอบ logic โดยพิมพ์แทนพูด (ไม่ต้องมี mic/policy)
PYTHONPATH=src python -m robot_learning.deploy.voice_pipeline \
  --text "เก็บลูกบอลสีแดงกับสีน้ำเงินแล้วเท" --dry-run
```

---

## โครงสร้างโปรเจกต์

```
assets/so101/          MJCF + meshes ของ SO-101 + scene_base.xml
config/objects.yaml    นิยามสี/ทรง/ตำแหน่งวัตถุ + physics
src/robot_learning/
  env/                 scene_builder, scene_randomizer, so101_env (Gym API)
  collect/             ik (mink), fsm_policy, record_scripted, teleop_gamepad
  deploy/              run_policy, llm_orchestrator, voice_pipeline
  utils/               constants (joint names, paths, camera names)
scripts/               view_fsm, render_fsm (ดูหุ่นทำงาน)
notebooks/             train_smolvla_colab.ipynb
docs/                  ARCHITECTURE.md, GRASP.md (จุดสำคัญเชิงเทคนิค)
```

---

## จุดสำคัญเชิงเทคนิค (สรุป — รายละเอียดใน docs/)

- **หุ่นเอื้อมหยิบของที่พื้น**: gripper ทรงนี้ approach เฉียง ตัว body ยื่นต่ำ →
  ใช้ **collision filter** ให้มือหุ่นทะลุผิวโต๊ะได้ แต่ยังชนวัตถุ (bitmask)
- **grasp-assist (weld)**: jaw ทรงโค้งคีบลูกกลม/เหลี่ยมลื่น → ใช้ **weld equality**
  ตรึงวัตถุติดมือตอนหยิบ ปล่อยตอนวาง → เก็บ dataset สำเร็จ ~100% (ดู [docs/GRASP.md](docs/GRASP.md))
- **actuator เสริมแรง**: เพิ่ม gain/forcerange ของ joint แขนให้ hold ท่าตาม IK แม่น
- **task string มาตรฐาน**: `"pick up the <color> ball and put it in the tray"` —
  ใช้ทั้งตอน label dataset และตอน deploy (ต้องตรงกัน SmolVLA ถึงเข้าใจ)

---

## Roadmap

- [x] Phase 1: ball 5 สี — sim + dataset + deploy pipeline ครบ
- [ ] Phase 2: cube (ปรับ jaw geometry / grasp), routine `dump_tray` จริง
- [ ] เก็บ dataset เต็ม + train SmolVLA จริง + วัด success rate ของ policy
- [ ] ต่อ Ollama + mic จริง, ทดสอบ voice end-to-end
- [ ] (option) sim-to-real: SO-101 ตัวจริง
