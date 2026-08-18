"""fsm_policy — Scripted finite-state-machine สำหรับหยิบวัตถุ target ใส่ถาด.

เป็น "oracle policy": อ่านตำแหน่งวัตถุ/ถาดจาก sim ground-truth (ผ่าน env)
ใช้สร้าง demonstration อัตโนมัติเพื่อเก็บ dataset (ไม่ใช่ policy ที่ deploy จริง).

State machine ต่อ 1 episode:
    APPROACH_ABOVE : IK ไปเหนือวัตถุ (gripper เปิด)
    DESCEND        : ลงไปที่วัตถุ (pinch ที่ center + offset เล็กน้อย)
    GRASP          : ปิด gripper ช้าๆ (interpolate) แล้ว settle
    LIFT           : ยกขึ้น
    MOVE_TO_TRAY   : ไปเหนือถาด
    RELEASE        : เปิด gripper วางลงถาด
    RETRACT        : ถอยขึ้น
    DONE

การเคลื่อนที่ทุก phase ใช้ **interpolation** จาก ctrl ปัจจุบันไป target ทีละนิด
(ไม่กระโดด) เพื่อไม่ให้แขนกระชากไปปัดวัตถุ. พารามิเตอร์ความสูง/offset ได้จาก
การ calibrate ให้ jaw คีบลูกกลมติด (ดู docs/GRASP.md).

Interface `.act(obs, env) -> action[6]` เข้ากันได้กับ gamepad teleop (signature
เดียวกัน) เพื่อสลับ source ได้.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from robot_learning.collect.ik import ArmIK
from robot_learning.utils.constants import GRIPPER_CLOSED, GRIPPER_OPEN


class Phase(enum.Enum):
    APPROACH_ABOVE = enum.auto()
    DESCEND = enum.auto()
    GRASP = enum.auto()
    LIFT = enum.auto()
    MOVE_TO_TRAY = enum.auto()
    LOWER_TO_TRAY = enum.auto()   # ลง pinch ไปก้นถาด (ยังถือ) — วางเบาๆ ไม่ตกจากสูง
    RELEASE = enum.auto()
    RETRACT = enum.auto()
    DONE = enum.auto()


@dataclass
class FSMConfig:
    approach_height: float = 0.07    # สูงเหนือวัตถุตอน approach (เมตร)
    approach_back: float = 0.0       # ก้ามปู: approach จากบนตรงๆ (ครอบลงมา 2 นิ้ว)
                                     # ไม่ถอยหลัง (side-grasp ใช้ >0 สอดจากหลัง)
    grasp_z_offset: float = 0.006    # pinch ที่ ~center ลูก (ไม่ต่ำจนกดลูกจมพื้น)
    lift_height: float = 0.14
    tray_height: float = 0.10        # สูงเหนือถาดตอน MOVE_TO_TRAY
    tray_release_height: float = 0.04  # ลง pinch ใกล้ก้นถาดก่อนปล่อย (วางเบาๆ ไม่ตก)
    # จำนวน control-step ต่อ phase (แขน interpolate ไปให้ถึง — ต้องพอให้
    # แขนวิ่งจากท่า home ที่สูง ลงมาถึงวัตถุ ไม่งั้น descend ค้างกลางทาง)
    steps_move: int = 70
    steps_descend: int = 90          # descend ต้องนานพอให้ pinch ลงทับลูกจริง
    steps_grasp: int = 45            # ปิด jaw
    steps_settle: int = 20           # ค้างให้ jaw บีบแน่นก่อนยก
    steps_release: int = 30          # เปิด gripper + settle (ลงถาดทำใน LOWER_TO_TRAY)
    ik_fail_thresh: float = 0.05     # IK error เกินนี้ = เอื้อมไม่ถึง


# ลำดับ phase
_ORDER = [
    Phase.APPROACH_ABOVE,
    Phase.DESCEND,
    Phase.GRASP,
    Phase.LIFT,
    Phase.MOVE_TO_TRAY,
    Phase.LOWER_TO_TRAY,
    Phase.RELEASE,
    Phase.RETRACT,
    Phase.DONE,
]


@dataclass
class FSMPolicy:
    """FSM oracle policy. สร้างใหม่ต่อ 1 episode."""

    cfg: FSMConfig = field(default_factory=FSMConfig)
    ik: ArmIK = field(default_factory=ArmIK)

    phase: Phase = field(init=False, default=Phase.APPROACH_ABOVE)
    _phase_step: int = field(init=False, default=0)
    _q_start: np.ndarray | None = field(init=False, default=None)  # ctrl arm ตอนเข้า phase
    _q_target: np.ndarray | None = field(init=False, default=None)  # ctrl arm เป้าหมาย
    _grip_start: float = field(init=False, default=GRIPPER_OPEN)
    _grip_target: float = field(init=False, default=GRIPPER_OPEN)
    _failed: bool = field(init=False, default=False)

    # ── วางแผน phase (เรียกตอนเข้า phase ใหม่) ────────────────────────
    def _plan_phase(self, obs, env) -> None:
        cur_arm = obs["agent_pos"][:5].astype(np.float64)
        self._q_start = cur_arm.copy()
        self._grip_start = float(obs["agent_pos"][5])
        self._q_target = cur_arm.copy()  # default: อยู่กับที่
        self._grip_target = self._grip_start

        tgt = env.target_position()
        tray = env.tray_position()

        # ทิศ radial (จากฐานหุ่นที่ origin → ลูก) ในระนาบ xy — jaw ของ side-grasp
        # ยื่นเข้าหาลูกตามแนวนี้. approach จาก "ด้านหลังลูก" (ถอยตามแนว radial)
        # แล้วเลื่อนเข้า → jaw สอดเข้าคีบไม่เสยลูกกระเด็น.
        radial = np.array([tgt[0], tgt[1], 0.0])
        n = np.linalg.norm(radial)
        radial = radial / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        back = self.cfg.approach_back      # ถอยหลังจากลูก (เมตร)

        def ik_to(goal):
            q, err = self.ik.solve(goal, cur_arm)
            if err > self.cfg.ik_fail_thresh:
                self._failed = True
            return q

        if self.phase == Phase.APPROACH_ABOVE:
            # ด้านหลังลูก + สูงเล็กน้อย (เตรียมสอดเข้า)
            goal = tgt - back * radial + [0, 0, self.cfg.approach_height]
            self._q_target = ik_to(goal)
            self._grip_target = GRIPPER_OPEN
        elif self.phase == Phase.DESCEND:
            # เลื่อนเข้าหาลูกในแนว radial (pinch ไปที่ลูกพอดี) — jaw สอดเข้าคีบ
            self._q_target = ik_to(tgt + [0, 0, self.cfg.grasp_z_offset])
            self._grip_target = GRIPPER_OPEN
            # เปิด grasp-assist: weld attach ตอน pinch เข้าถึงลูก (ก่อน jaw ปิด)
            env.set_grasp_enabled(True)
        elif self.phase == Phase.GRASP:
            self._grip_target = GRIPPER_CLOSED  # ปิด jaw หนีบลูก (ดูสมจริง)
        elif self.phase == Phase.LIFT:
            self._q_target = ik_to(tgt + [0, 0, self.cfg.lift_height])
            self._grip_target = GRIPPER_CLOSED
        elif self.phase == Phase.MOVE_TO_TRAY:
            self._q_target = ik_to(tray + [0, 0, self.cfg.tray_height])
            self._grip_target = GRIPPER_CLOSED
        elif self.phase == Phase.LOWER_TO_TRAY:
            # ลง pinch ไปก้นถาด (ระดับ ~เดียวกับตอนคีบวัตถุ) — *ยังถือ* (ยังไม่ปล่อย)
            # → พอถึง RELEASE วัตถุถูกวางเบาๆ ที่ก้นถาด ไม่ตกจากที่สูงกระเด็น/กลิ้งออก
            self._q_target = ik_to(tray + [0, 0, self.cfg.tray_release_height])
            self._grip_target = GRIPPER_CLOSED
        elif self.phase == Phase.RELEASE:
            # อยู่ที่ก้นถาดแล้ว — เปิด gripper + ปลด weld (วางเบาๆ)
            self._q_target = ik_to(tray + [0, 0, self.cfg.tray_release_height])
            self._grip_target = GRIPPER_OPEN
            env.set_grasp_enabled(False)
        elif self.phase == Phase.RETRACT:
            self._q_target = ik_to(tray + [0, 0, self.cfg.tray_height + 0.06])
            self._grip_target = GRIPPER_OPEN

    def _phase_budget(self) -> int:
        if self.phase == Phase.DESCEND:
            return self.cfg.steps_descend
        if self.phase == Phase.GRASP:
            return self.cfg.steps_grasp + self.cfg.steps_settle
        if self.phase == Phase.LOWER_TO_TRAY:
            return self.cfg.steps_move   # เวลาลงก้นถาด (แขนวิ่งลง)
        if self.phase == Phase.RELEASE:
            return self.cfg.steps_release
        return self.cfg.steps_move

    def _advance_phase(self) -> None:
        idx = _ORDER.index(self.phase)
        self.phase = _ORDER[min(idx + 1, len(_ORDER) - 1)]
        self._phase_step = 0

    # ── public API: act(obs, env) -> action[6] ───────────────────────
    def act(self, obs, env) -> np.ndarray:
        if self._phase_step == 0:
            self._plan_phase(obs, env)

        budget = self._phase_budget()
        # interpolate เชิงเส้นจาก start ไป target ภายใน budget steps
        alpha = min(1.0, (self._phase_step + 1) / max(1, budget - self.cfg.steps_settle
                                                       if self.phase == Phase.GRASP else budget))
        q_arm = (1 - alpha) * self._q_start + alpha * self._q_target
        grip = (1 - alpha) * self._grip_start + alpha * self._grip_target
        action = np.concatenate([q_arm, [grip]]).astype(np.float32)

        self._phase_step += 1
        if self._phase_step >= budget:
            self._advance_phase()
        return action

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def failed(self) -> bool:
        return self._failed
