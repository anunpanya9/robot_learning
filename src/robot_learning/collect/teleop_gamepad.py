"""teleop_gamepad.py — บังคับหุ่นด้วยจอย Xbox (hook เผื่อเก็บ demo เอง).

ออกแบบ interface ให้เหมือน FSMPolicy (`.act(obs, env) -> action[6]`, `.done`)
เพื่อให้ record_scripted / pipeline สลับ source ได้โดยไม่แก้โค้ดหลัก:

    policy = FSMPolicy(...)          # อัตโนมัติ
    policy = GamepadPolicy(...)      # บังคับเอง (จอย Xbox)

การควบคุม (Xbox, ผ่าน pygame):
    left stick   → เลื่อน end-effector ในระนาบ XY (IK)
    right stick Y→ ขึ้น/ลง (Z)
    LT / RT      → ปิด / เปิด gripper
    A            → toggle grasp-assist (ตรึง/ปล่อยวัตถุ)
    Start        → จบ episode (mark done)

หมายเหตุ: โมดูลนี้เป็น "hook" — โครงพร้อมใช้ แต่ยังไม่ผูกกับ record loop
โดย default (record_scripted ใช้ FSM). เปิดใช้ทีหลังด้วย --source gamepad
(ต้องต่อจอย + ติดตั้ง pygame). การ map ปุ่มอาจต่างตามรุ่นจอย/ระบบ ปรับ
index ใน _read_axes/_read_buttons ตามจริง.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_learning.collect.ik import ArmIK
from robot_learning.utils.constants import GRIPPER_CLOSED, GRIPPER_OPEN


@dataclass
class GamepadConfig:
    xy_speed: float = 0.004      # เมตรต่อ step ที่ stick เต็ม
    z_speed: float = 0.004
    deadzone: float = 0.15       # ตัด noise รอบศูนย์กลาง stick


@dataclass
class GamepadPolicy:
    """บังคับ end-effector ด้วยจอย Xbox ผ่าน IK. Interface เหมือน FSMPolicy."""

    cfg: GamepadConfig = field(default_factory=GamepadConfig)
    ik: ArmIK = field(default_factory=ArmIK)

    _joystick: object = field(init=False, default=None)
    _ee_target: np.ndarray | None = field(init=False, default=None)
    _grip: float = field(init=False, default=GRIPPER_OPEN)
    _done: bool = field(init=False, default=False)
    _grasp_on: bool = field(init=False, default=False)

    def __post_init__(self):
        import pygame  # import ตอนใช้จริง (ไม่บังคับ dependency)

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("ไม่พบจอย — ต่อ Xbox controller ก่อน")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self._pygame = pygame

    def _dz(self, v: float) -> float:
        return 0.0 if abs(v) < self.cfg.deadzone else v

    def act(self, obs, env) -> np.ndarray:
        pg = self._pygame
        pg.event.pump()
        js = self._joystick

        # อ่าน stick (index อาจต่างตามระบบ — Xbox ทั่วไป: 0=LX,1=LY,3=RX,4=RY)
        lx = self._dz(js.get_axis(0))
        ly = self._dz(js.get_axis(1))
        ry = self._dz(js.get_axis(4))
        # trigger: LT=axis2, RT=axis5 (ค่า -1..1)
        lt = js.get_axis(2)
        rt = js.get_axis(5)

        # เริ่ม ee_target จากตำแหน่งปัจจุบันถ้ายังไม่มี
        if self._ee_target is None:
            self._ee_target = env.ee_position().copy()

        # อัปเดตเป้าหมาย ee (stick: ly กลับด้าน)
        self._ee_target[0] += -ly * self.cfg.xy_speed
        self._ee_target[1] += lx * self.cfg.xy_speed
        self._ee_target[2] += -ry * self.cfg.z_speed

        # gripper: RT เปิด, LT ปิด
        if rt > 0.5:
            self._grip = GRIPPER_OPEN
        elif lt > 0.5:
            self._grip = GRIPPER_CLOSED

        # ปุ่ม A (0) = toggle grasp-assist, Start (7) = done
        if js.get_button(0):
            self._grasp_on = not self._grasp_on
            env.set_grasp_enabled(self._grasp_on)
        if js.get_button(7):
            self._done = True

        q_arm, _ = self.ik.solve(self._ee_target, obs["agent_pos"][:5])
        return np.concatenate([q_arm, [self._grip]]).astype(np.float32)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def failed(self) -> bool:
        return False  # teleop: ผู้ใช้ตัดสินเอง (ไม่ auto-fail)
