"""ik — inverse kinematics helper สำหรับ SO-101 (ใช้ mink).

ออกแบบให้ IK ทำงานบน "โมเดลแขนล้วน" (so101.xml, nq=6) แยกจาก sim หลัก
ที่มี freejoint ของวัตถุ (nq ใหญ่) — เพราะถ้า solve บนโมเดลเต็ม IK จะไป
ขยับ freejoint ของวัตถุด้วย. Solve เสร็จได้มุมแขน 5 ตัว → เอาไปเป็น ctrl.

พารามิเตอร์ที่ calibrate แล้ว (ดู docs):
    orientation_cost ≈ 0.05  → gripper ชี้ลงเกือบดิ่ง (z-axis site ~ world +z)
                               โดย position error ยังต่ำ (~5 มม.)
    iters ≈ 150–200 ต่อ 1 target
    solver = "daqp" (ตัวที่ติดตั้งมากับ qpsolvers บนเครื่องนี้)
"""

from __future__ import annotations

import mujoco
import numpy as np

try:
    import mink
    from mink import SE3, SO3
except ImportError as e:  # pragma: no cover
    raise ImportError("ต้องติดตั้ง mink: pip install mink") from e

from robot_learning.utils.constants import PINCH_SITE, SO101_MJCF

# orientation cost สูงพอให้ jaw ชี้ลงตาม target rotation (ดู _grasp_down_rotation)
_DEFAULT_ORIENTATION_COST = 0.5
_DEFAULT_ITERS = 250
_SOLVER = "daqp"
_DT = 0.02

# ท่าอ้างอิงสำหรับ orientation ของ pinch ที่ IK จะเล็ง.
# "ก้ามปู": หมุน wrist_roll≈96° ให้นิ้วสองข้างประกบซ้าย-ขวา (คีบแนวนอน แบบ
# gripper parallel) เข้าหาลูกจากด้านข้าง — เหมาะกับ sphere/cube/cylinder.
# calibrate ให้ pinch = "จุดหนีบจริงกึ่งกลางปลายนิ้ว" (ไม่ใช่โคน/servo).
# IK หมุน orientation ตาม yaw ของลูก (ดู _target_rotation). docs/GRASP.md §8
_GRASP_REF_QPOS = (0.0, -0.28, 0.81, 0.50, 1.70, 0.5)


class ArmIK:
    """IK บนโมเดลแขนล้วน. สร้างครั้งเดียว แล้วเรียก .solve() ซ้ำได้."""

    def __init__(
        self,
        orientation_cost: float = _DEFAULT_ORIENTATION_COST,
        iters: int = _DEFAULT_ITERS,
    ):
        # สร้างโมเดลแขนล้วนจาก spec + ยกฐานให้ตรงกับ sim (สำคัญมาก:
        # ถ้าฐาน IK ไม่ตรงกับ sim → FK คลาดเคลื่อน หยิบพลาด)
        from robot_learning.env.scene_builder import raise_base, strengthen_arm_actuators

        spec = mujoco.MjSpec.from_file(str(SO101_MJCF))
        raise_base(spec)
        self.model = spec.compile()
        strengthen_arm_actuators(self.model)
        self.iters = iters
        self._sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE)
        self._site_name = PINCH_SITE
        self._orientation_cost = orientation_cost
        # target orientation อ้างอิง: จับจากท่า side-grasp (jaw เฉียงเข้าหาลูก
        # เกือบแนวนอน, ปลายไม่แตะพื้น). ท่านี้ jaw ชี้ไปทาง +x (หน้าหุ่น).
        # ตอน solve จะ "หมุนตาม yaw ของลูก" ให้ jaw เข้าหาลูกจากทิศ base→ลูก
        # (ไม่งั้นลูกที่เยื้องซ้าย/ขวาจะเอื้อมไม่ถึง). ดู docs/GRASP.md §7
        # เก็บ orientation อ้างอิง 2 ทิศ (wrist_roll ±) — ก้ามปูหมุน roll asymmetric
        # ทิศ + เอื้อม y บวก (ซ้าย), ทิศ − เอื้อม y ลบ (ขวา). เลือกตามฝั่ง y ของลูก
        # ใน _target_rotation → workspace เต็มทั้งซ้าย-ขวา (100%).
        self._ref_R_pos = self._ref_orientation(_GRASP_REF_QPOS)
        neg = list(_GRASP_REF_QPOS)
        neg[4] = -neg[4]  # กลับทิศ wrist_roll
        self._ref_R_neg = self._ref_orientation(tuple(neg))

    def _ref_orientation(self, qpos) -> np.ndarray:
        ref = mujoco.MjData(self.model)
        ref.qpos[:6] = qpos
        mujoco.mj_forward(self.model, ref)
        return ref.site_xmat[self._sid].reshape(3, 3).copy()

    def _target_rotation(self, target_pos: np.ndarray) -> "SO3":
        """orientation เป้าหมาย = ท่าอ้างอิง (เลือกทิศ roll ตามฝั่ง y) หมุนตาม
        azimuth (yaw) ของลูก → นิ้วก้ามปูประกบลูกจากทิศฐานหุ่น เอื้อมได้ทั้ง 2 ฝั่ง."""
        ref_R = self._ref_R_pos if target_pos[1] >= 0 else self._ref_R_neg
        yaw = float(np.arctan2(target_pos[1], target_pos[0]))
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return SO3.from_matrix(Rz @ ref_R)

    def solve(
        self, target_pos: np.ndarray, q_seed: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """แก้ IK ให้ gripperframe ไปที่ target_pos (ชี้ลง).

        Args:
            target_pos: ตำแหน่งเป้าหมาย world [3]
            q_seed: มุมแขนเริ่มต้น [>=5] (ใช้ 5 ตัวแรก) — ช่วย convergence

        Returns:
            (q_arm[5], pos_error) — มุมแขน 5 ตัว และ error ระยะ (เมตร)
        """
        q0 = np.zeros(6)
        q0[:5] = np.asarray(q_seed, dtype=np.float64)[:5]
        # seed wrist_roll ให้ตรงกับฝั่ง y (ก้ามปูหมุน roll asymmetric) — ช่วยให้
        # convergence ดีทั้งซ้าย-ขวา ไม่ว่า FSM ส่ง seed อะไรมา
        target_pos = np.asarray(target_pos)
        q0[4] = abs(_GRASP_REF_QPOS[4]) * (1.0 if target_pos[1] >= 0 else -1.0)

        cfg = mink.Configuration(self.model)
        cfg.update(q0)

        frame_task = mink.FrameTask(
            frame_name=self._site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=self._orientation_cost,
        )
        posture = mink.PostureTask(self.model, cost=1e-4)
        posture.set_target(q0)
        target_R = self._target_rotation(np.asarray(target_pos))
        frame_task.set_target(
            SE3.from_rotation_and_translation(target_R, np.asarray(target_pos))
        )

        for _ in range(self.iters):
            vel = mink.solve_ik(cfg, [frame_task, posture], dt=_DT, solver=_SOLVER)
            cfg.integrate_inplace(vel, _DT)

        # วัด error
        d = mujoco.MjData(self.model)
        d.qpos[:] = cfg.q
        mujoco.mj_forward(self.model, d)
        ee = d.site_xpos[self._sid]
        err = float(np.linalg.norm(ee - np.asarray(target_pos)))
        return cfg.q[:5].copy(), err
