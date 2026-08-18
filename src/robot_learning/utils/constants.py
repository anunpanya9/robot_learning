"""ค่าคงที่กลางของโปรเจกต์.

ค่าเหล่านี้มาจากการตรวจ MJCF ของ SO-101 จริง (assets/so101/so101.xml):

    joints (6, ตามลำดับ qpos/ctrl):
        shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    end-effector site:  gripperframe  (อยู่บน body "gripper")
    gripper joint range: -0.17453 (ปิด) .. 1.74533 (เปิด)

รวมพาธ asset ไว้ที่เดียว เพื่อให้ env/collect/deploy อ้างพาธเดียวกัน
โดยไม่ต้อง hardcode ซ้ำ.
"""

from __future__ import annotations

from pathlib import Path

# ── พาธ (คำนวณจากตำแหน่งไฟล์นี้: src/robot_learning/utils/constants.py) ──
# parents[3] = repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
ASSETS_DIR: Path = REPO_ROOT / "assets" / "so101"
SO101_MJCF: Path = ASSETS_DIR / "so101.xml"
OBJECTS_CONFIG: Path = REPO_ROOT / "config" / "objects.yaml"

# ── ชื่อ joint / actuator (ลำดับตรงกับ MJCF) ──
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT: str = "gripper"
ALL_JOINTS: tuple[str, ...] = ARM_JOINTS + (GRIPPER_JOINT,)

# ── end-effector sites ──
# gripperframe: frame เดิมจาก MJCF (ใช้วัด/อ้างอิงมือทั่วไป)
# pinch: จุดคีบจริงระหว่างปลาย jaw (เพิ่มเข้ามา) — ใช้เป็น target ของ IK ตอนหยิบ
EE_SITE: str = "gripperframe"
PINCH_SITE: str = "pinch"

# ── ค่ามุม gripper (จาก ctrlrange ของ actuator "gripper") ──
GRIPPER_OPEN: float = 1.5      # เปิดกว้างพอคีบวัตถุ (< max 1.745)
GRIPPER_CLOSED: float = -0.1   # ปิด (> min -0.1745 เพื่อบีบวัตถุ)

# ── ความสูง platform วางวัตถุ ──
# gripper ของ SO-101 คีบแบบเฉียง และตัว gripper เองยื่นต่ำกว่าจุด pinch ~6มม.
# ถ้าวางวัตถุบนพื้น (z≈0.02) ตัว gripper จะชนโต๊ะก่อนคีบถึง. แก้ด้วยการวาง
# วัตถุ+ถาดบน "platform" ยกสูง PLATFORM_HEIGHT → gripper คีบได้โดยไม่ชนผิว.
# (ฐานหุ่นคงอยู่ที่ z=0 ตามเดิม)
PLATFORM_HEIGHT: float = 0.06
BASE_HEIGHT: float = 0.0

# ── ชื่อกล้องใน scene (ต้องตรงกับ scene builder และ key ใน dataset) ──
# SmolVLA คาดหวังหลายมุมกล้อง → ใช้ top (มองลงโต๊ะ) + wrist (ติดปลายแขน)
CAMERAS: tuple[str, ...] = ("top", "wrist")
