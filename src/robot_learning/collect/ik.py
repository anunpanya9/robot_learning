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

# ท่าอ้างอิงที่ jaw ชี้ลงสวย (ใช้จับ orientation ของ pinch เป็น target ให้ IK)
_GRASP_REF_QPOS = (0.0, -0.7, 0.9, 1.5, 0.0, 0.5)


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
        # target orientation: จับจากท่าอ้างอิงที่ jaw ชี้ลงสวย (pinch z-axis ≈ world +z)
        # การล็อก orientation นี้ทำให้ทุก grasp เข้าหาวัตถุแบบ "jaw ชี้ลง" เหมือนกัน
        # → gripper คร่อมวัตถุจากบน ไม่ปัดวัตถุด้านข้าง
        ref = mujoco.MjData(self.model)
        ref.qpos[:6] = _GRASP_REF_QPOS
        mujoco.mj_forward(self.model, ref)
        R = ref.site_xmat[self._sid].reshape(3, 3).copy()
        self._target_R = SO3.from_matrix(R)

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
        frame_task.set_target(
            SE3.from_rotation_and_translation(self._target_R, np.asarray(target_pos))
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
