"""collect — สร้าง demonstration ด้วย FSM/gamepad และเก็บเป็น LeRobotDataset."""

from robot_learning.collect.fsm_policy import FSMConfig, FSMPolicy, Phase
from robot_learning.collect.ik import ArmIK

__all__ = ["FSMConfig", "FSMPolicy", "Phase", "ArmIK"]

# GamepadPolicy import แบบ lazy (ต้องมี pygame) — ไม่ใส่ใน __all__ หลัก
# ใช้: from robot_learning.collect.teleop_gamepad import GamepadPolicy
