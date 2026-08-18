# สถาปัตยกรรม robot_learning

ภาพรวมว่าแต่ละส่วนต่อกันยังไง — สำหรับคนที่มาอ่านโค้ดทีหลัง

---

## Data flow

```
                    scene_randomizer.sample()
                    (สุ่มสี/ตำแหน่ง + เลือก target แบบ round-robin)
                              │  RandomizeResult
                              ▼
  scene_builder.build_scene()  ─────►  mujoco.MjModel
  (compose MJCF: หุ่น+วัตถุ+ถาด+2กล้อง+wrist cam+weld,
   ใส่ actuator เสริมแรง + collision filter + gripper friction)
                              │
                              ▼
  SO101PickPlaceEnv  ◄──────────────────┐
   reset() → obs {pixels{top,wrist}, agent_pos}   │ action[6]
   step(action) → obs, reward, terminated         │ (joint targets)
   + grasp-assist (weld) อัตโนมัติ                 │
                              │ obs                │
                              ▼                    │
              ┌──────── policy ─────────┐          │
              │  FSMPolicy (oracle)     │──────────┘  เก็บ dataset
              │  หรือ GamepadPolicy     │
              │  หรือ SmolVLA (deploy)  │
              └─────────────────────────┘
```

## ความรับผิดชอบของแต่ละโมดูล

### `env/` — โลกจำลอง (ไม่รู้เรื่อง policy)
- **`scene_randomizer.py`**: สุ่มฉาก 1 episode. `build_task_string()` สร้าง
  instruction มาตรฐาน (ใช้ร่วมกับ deploy). balance target แบบ round-robin.
- **`scene_builder.py`**: ประกอบ MJCF จากโค้ด (mjSpec) — เพราะสี/ตำแหน่ง/จำนวน
  วัตถุเปลี่ยนทุก episode + ต้องแนบ wrist cam เข้า body หุ่น + สร้าง weld.
  ทำ post-compile tweaks: `strengthen_arm_actuators`, `set_surface_collision_filter`,
  `set_gripper_friction`.
- **`so101_env.py`**: Gymnasium-style env. รับผิดชอบ physics + observation +
  success check + **grasp-assist (weld)**. ให้ oracle access (`target_position()`,
  `ee_position()`) สำหรับ FSM.

### `collect/` — สร้าง demo + เก็บ dataset
- **`ik.py`** (`ArmIK`): inverse kinematics ด้วย mink บนโมเดลแขนล้วน (nq=6)
  แยกจาก sim (ที่มี freejoint วัตถุ). solve pinch → joint targets.
- **`fsm_policy.py`** (`FSMPolicy`): state machine หยิบวัตถุ (oracle).
  interpolate ctrl นุ่มนวล + สั่ง grasp-assist. interface `.act(obs, env)`.
- **`teleop_gamepad.py`** (`GamepadPolicy`): บังคับด้วยจอย Xbox — interface
  เดียวกับ FSM (สลับ source ได้).
- **`record_scripted.py`**: วน N ep ด้วย FSM → LeRobotDataset v3.0.
  เก็บเฉพาะ ep สำเร็จ. key: `observation.images.{top,wrist}`, `observation.state`,
  `action`, `task`.

### `deploy/` — สั่งงานด้วยเสียง (Phase D)
- **`run_policy.py`** (`PolicyRunner`): โหลด SmolVLA + แปลง obs→batch + closed-loop.
- **`llm_orchestrator.py`** (`LLMOrchestrator`): Ollama LLM แปลงคำสั่ง (ไทย/อังกฤษ)
  → `Plan` (task queue + dump flag). มี keyword fallback ถ้าไม่มี Ollama.
  **map color→task string เอง** (ผ่าน `build_task_string`) เพื่อให้ตรง dataset.
- **`voice_pipeline.py`**: mic → Whisper → orchestrator → run_policy. รองรับ
  `--text` (พิมพ์แทนพูด) + `--dry-run` (แสดง plan เฉยๆ).

### `utils/constants.py` — ค่าคงที่กลาง
joint names, asset paths, camera names, gripper open/closed, pinch site.
**ทุกโมดูลอ้างที่เดียว** ไม่ hardcode ซ้ำ.

---

## สัญญาที่ต้องตรงกัน (contracts)

1. **observation/action keys** — record_scripted เขียน = run_policy อ่าน = SmolVLA คาดหวัง
   (`observation.images.top/wrist`, `observation.state`, `action`)
2. **task string format** — `build_task_string()` ใช้ทั้งตอน label dataset และ deploy
   → ถ้าแก้ format ต้องแก้ที่เดียว (scene_randomizer) แล้ว re-train
3. **IK model = sim model** — actuator/base ต้อง config เหมือนกัน (ไม่งั้น FK คลาด)
4. **camera names** = `constants.CAMERAS` = key ใน dataset = ที่ scene_builder สร้าง

---

## ทำไมออกแบบแบบนี้

- **policy แยกจาก env**: FSM/gamepad/SmolVLA ใช้ env เดียวกัน interface `.act()` เดียว
  → เก็บ dataset กับ deploy ใช้โลกเดียวกันเป๊ะ (sim-to-sim ตรง)
- **build scene จากโค้ด**: randomization + weld + wrist cam ทำใน XML ตรงๆ ไม่ได้
- **grasp-assist ใน env (ไม่ใช่ policy)**: physics assist เป็นเรื่องของโลก ไม่ใช่
  ของ policy → SmolVLA ที่ deploy ก็ได้ assist เหมือนตอนเก็บ (consistent)
