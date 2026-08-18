# การคีบวัตถุใน MuJoCo — บันทึกการ calibrate (ส่วนที่ยากที่สุด)

เอกสารนี้บันทึกปัญหาและวิธีแก้เรื่องการทำให้ SO-101 หยิบวัตถุใน sim ได้จริง
เพื่อให้แก้/ปรับต่อได้ทีหลังโดยไม่ต้องไล่ debug ใหม่ทั้งหมด

---

## ข้อเท็จจริงของ gripper SO-101 (จากการวัด MJCF จริง)

- **6 joints**: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- gripper เป็น **position actuator** (kp≈998, forcerange≈±2.94 Nm ตาม servo จริง)
- **gripper คีบแบบเฉียง** ไม่ใช่ top-down ดิ่ง — jaw ชี้ไปข้างหน้า-ลง (~45°)
- **จุดคีบจริง (pinch)** อยู่ระหว่าง moving jaw กับ fixed finger:
  local `[0.017, 0, -0.0345]` ในกรอบ body `gripper` → เพิ่มเป็น site `pinch` ใน MJCF
- **ตัว gripper body ยื่นต่ำกว่า pinch** ~6mm (ตรงๆ) แต่ **มากถึง ~6-7cm** เมื่อ
  แขนเข้าเฉียง → ชนโต๊ะก่อนถึงวัตถุ
- **ช่องเปิด jaw ตอนปิดสุด (grip=-0.1)** = ~1.3cm → วัตถุต้องเล็กกว่านี้

---

## ปัญหา 5 ข้อที่เจอ + วิธีแก้ (ตามลำดับที่ค้นพบ)

### 1. IK ให้ gripper ชี้ผิดทิศ (jaw ชี้ไปข้างหน้า ไม่ลง)
**อาการ**: IK orientation=identity ทำให้ jaw ชี้แนวนอน คร่อมวัตถุไม่ได้
**แก้**: จับ orientation ของ pinch จาก "ท่าอ้างอิงที่ jaw ชี้ลงสวย"
(`_GRASP_REF_QPOS = (0, -0.7, 0.9, 1.5, 0, 0.5)`) มาเป็น target ของ IK +
ตั้ง `orientation_cost=0.5` → jaw ชี้ลงทุก grasp
→ [`collect/ik.py`](../src/robot_learning/collect/ik.py)

### 2. แขน sag ไม่ hold ท่าตาม ctrl (สู้ gravity ไม่ไหว)
**อาการ**: สั่ง joint ไป X แต่แขนค้างที่ Y (โดยเฉพาะ shoulder_lift) เพราะ
forcerange servo จริง (±2.94) น้อยไป
**แก้**: `strengthen_arm_actuators` — เพิ่ม kp×3, forcerange=±10 สำหรับ joint แขน
(ต้องปรับ `gainprm[0]` และ `biasprm[1]=-kp` คู่กัน). gripper เพิ่มแยก (×2, ±8)
→ [`env/scene_builder.py`](../src/robot_learning/env/scene_builder.py)

### 3. gripper ชนโต๊ะ เอื้อมลงหยิบของที่พื้นไม่ได้
**อาการ**: descend แล้ว body gripper ชนพื้น (contact z=-0.002) ค้ำไว้ pinch
ลงไม่ถึงวัตถุ (ค้างที่ z≈0.07-0.13)
**แก้ (ที่ใช้จริง)**: **collision filter** — ตั้ง contype/conaffinity ให้มือหุ่น
ทะลุพื้น/ถาดได้ แต่ยังชนวัตถุ:
- พื้น/ถาด: bit3 (contype=4, conaffinity=4)
- วัตถุ: bit1|bit3 = 5 (ชนทั้งมือและพื้น)
- มือหุ่น: bit1 (default) → ชนวัตถุ (1&1) แต่ไม่ชนพื้น (1&4=0)
→ `set_surface_collision_filter` ใน scene_builder
> (เคยลอง "ยกฐานหุ่น" และ "platform ยกวัตถุ" แต่ทำให้ shoulder ชนขอบ / gripper
>  ยังชน platform — collision filter คือทางที่ clean ที่สุด)

### 4. แขนกระชากไปปัดวัตถุ (จาก home ที่สูง)
**อาการ**: ctrl กระโดดไป target ทันที + kp สูง → แขนสะบัดปัดวัตถุกระเด็น
**แก้**: FSM ใช้ **interpolation** (ค่อยๆ เลื่อน ctrl จากปัจจุบันไป target ภายใน
N steps) + home pose ที่ไม่มี contact (`[0, -1.0, 1.3, 0.3, 0, 1.5]`, pinch z≈0.13)

### 5. jaw คีบวัตถุลื่นหลุด (ปัญหาใหญ่สุด)
**อาการ**: jaw ทรงโค้งบีบลูกกลม → ball กลิ้งหนี +x; cube เล็กหลุดใต้ jaw
จูน friction/solref/ขนาด/แรงบีบแล้ว ball ติดบ้าง (~20%) แต่ไม่เสถียร cube ไม่ติดเลย
**แก้ (ที่ใช้จริง)**: **grasp-assist ด้วย weld equality** — เทคนิคมาตรฐาน sim manipulation:
- สร้าง weld eq ต่อวัตถุ (gripper ↔ obj) แบบ inactive ใน scene_builder
- FSM เรียก `env.set_grasp_enabled(True)` ตอน DESCEND → env activate weld เมื่อ
  pinch เข้าใกล้วัตถุ < 0.055 (ตรึง obj ติดมือ **ก่อน** jaw ดันกระเด็น)
- ปล่อย (`set_grasp_enabled(False)`) ตอน RELEASE → ปลด weld วางลงถาด
- **จุดพลาดที่เคยเจอ**: ตั้ง `eq_data` ไม่ครบ (anchor ค้างค่า default `[0,1,0]`)
  ทำให้วัตถุดีดกระเด็น → ต้องตั้งครบ: `data[0:3]=0` (anchor), `[3:6]=relpos`,
  `[6:10]=relquat`, `[10]=1` (torquescale)
→ `_set_weld_relpose`, `_update_grasp_assist` ใน [`env/so101_env.py`](../src/robot_learning/env/so101_env.py)

**ผลลัพธ์**: หยิบ ball ทุกสีใส่ถาดสำเร็จ **100%** balance ทุกสี

![FSM หยิบลูกบอลใส่ถาด](media/fsm_demo.gif)

> มุมข้าง | กล้อง top | กล้อง wrist — เห็น FSM approach → descend → grasp-assist
> ตรึงลูกบอล → ยกไปถาด → ปล่อย

### 5.1 ทำ grasp ให้ดูสมจริง (แก้อาการ "ลูกลอยห่างมือ")

**อาการเดิม**: weld ตรึงลูกที่ตำแหน่งลูก *จริง* ตอน attach — ซึ่งอาจห่าง pinch
4-5.6ซม. → ลูกลอยห่างมือ ดูเหมือนไม่ได้จับ (แต่ยกได้) → dataset ดูไม่น่าเชื่อถือ

**แก้ 3 จุด** (ให้ jaw ทับลูกจริงก่อนตรึง + ลูกอยู่ตรงจุดหนีบ):
1. **snap-to-pinch**: `_set_weld_relpose` ตรึงลูกที่ตำแหน่ง **pinch (จุดหนีบ)** ไม่ใช่
   ตำแหน่งลูกจริง → ลูก "snap" เข้าไปอยู่ระหว่าง jaw (pinch-obj เหลือ ~0.6-1.0ซม.)
2. **attach ตั้งแต่ปลาย DESCEND** (jaw ยังเปิด, pinch ทับลูก) — จับ *ก่อน* jaw ปิด
   ดันลูกกระเด็น. ลด `_GRASP_ATTACH_DIST` เป็น 0.04
3. **descend ลึกขึ้น** (`grasp_z_offset=0.006`, `steps_descend=90`) ให้ pinch ทับลูกจริง

ผลลัพธ์: ลูกอยู่ระหว่าง jaw + jaw ปิดหนีบ (ดูสมจริง) และ success กลับมา **100% (10/10)**

---

## 6. กล้อง wrist ต้องเห็นวัตถุตอนหยิบ (สำคัญต่อ VLA)

**อาการเดิม**: กล้อง wrist มองผิดทิศ (เห็นแต่ตัว gripper ด้านข้าง) → ตอนหยิบ
"ลูกหายไปจากเฟรม". **ปัญหาใหญ่ต่อ SmolVLA** เพราะ VLA ต้องเห็นวัตถุ target
เพื่อเล็งหยิบ — ถ้ากล้องไม่เห็น โมเดลเรียนไม่ได้.

**แก้**: calibrate กล้อง wrist ให้ "มองลงไปที่จุดหนีบ (pinch)" — คำนวณ quat จาก
look-direction (cam → pinch ในกรอบ body gripper) วางกล้องเหนือ-เยื้อง jaw:
```python
_WRIST_CAM_POS  = (0.05, 0.11, 0.06)
_WRIST_CAM_QUAT = (0.1313, 0.062, 0.4224, 0.8947)
_WRIST_CAM_FOVY = 70
```
ผล: ตอน approach/descend เห็น **วัตถุ target + jaw ทั้งสองข้าง + บริบทพื้นโต๊ะ**
ชัดเจน (ตอนคีบ วัตถุอาจถูก jaw บังบางส่วน — ปกติของ wrist cam จริง).
→ `_WRIST_CAM_*` ใน [`env/scene_builder.py`](../src/robot_learning/env/scene_builder.py)

> **หมายเหตุ "ปลาย gripper เฉียดพื้น"** (ดู §7 side-grasp ที่ใช้จริง): ปลาย jaw
> เฉียดพื้น ~1-2cm ตอนคีบ เป็น **ข้อจำกัดเชิงฟิสิกส์** ที่พิสูจน์แล้วว่าเลี่ยงไม่ได้
> ถ้าวัตถุอยู่ที่พื้น — เพราะลูกเล็ก (center z≈11mm) + jaw ของ SO-101 ยาว.
>
> **ทดสอบครบทุกทางแล้ว:**
> - *top-down*: jaw ชี้ดิ่ง ปลายเฉียดพื้นมากสุด
> - *side-grasp* (§7): jaw เฉียงคีบด้านข้าง ปลายเฉียดน้อยลง — ✅ **ใช้อันนี้**
> - *หมุน wrist_roll คีบซ้าย-ขวา*: ยังมีนิ้วต้องลงใกล้พื้น (jaw ยาว)
> - *ลูกใหญ่ขึ้น*: center สูงขึ้นช่วยได้นิด แต่ลูก >12mm เกินช่อง jaw (คีบไม่ได้)
> - *platform ยกวัตถุ*: แก้ได้จริง (platform เล็ก x[0.17,0.33] shoulder ไม่ชน) แต่
>   ต้อง rework scene/IK/FSM — **ยังไม่ทำ** (side-grasp ดีพอ, ไปโฟกัส train/voice)
>
> สรุป: **side-grasp เฉียดพื้น ~1-2cm ยอมรับได้** — physics ถูก (ไม่ทะลุลึก),
> success 100%, dataset ใช้ได้. ถ้าอยากให้ปลายพ้นพื้นสมบูรณ์ → ทำ platform (ดูข้างบน).

---

## 7. เปลี่ยนเป็น side-grasp (ปลาย jaw ไม่แตะพื้น + ลูกไม่ลอย)

**อาการเดิม (top-down)**: jaw ชี้ดิ่งลงคีบลูกที่พื้น → ปลาย jaw เฉียดพื้น (ดูเหมือน
ทะลุ) + พื้นเงาสะท้อนลูกกลับหัวทำให้ดูเหมือน "ลูกลอย" + weld attach ตอน pinch ยัง
สูงเหนือลูก → ลูกกระโดดลอยขึ้น.

**แก้ 3 จุด:**
1. **side-grasp** — เปลี่ยน `_GRASP_REF_QPOS` เป็นท่า jaw เฉียงเข้าหาลูกเกือบแนวนอน
   (`0, 0.30, 0.60, 0.30, 0`) → ปลาย jaw ต่ำสุด ~3mm เหนือพื้น (ไม่แตะ). IK
   **หมุน orientation ตาม yaw ของลูก** (`_target_rotation`) ให้ jaw เข้าหาลูกจาก
   ทิศ base→ลูก — ไม่งั้นลูกที่เยื้องซ้าย/ขวาเอื้อมไม่ถึง (fixed orientation reach แค่ y=0).
2. **approach จากด้านหลังลูก** — FSM `APPROACH` ไปที่ `tgt - back*radial + [0,0,h]`
   (ถอยหลังตามแนว base→ลูก) แล้ว `DESCEND` เลื่อนเข้าในแนว radial → jaw สอดเข้าคีบ
   **ไม่เสยลูกกระเด็น** (top-down approach จากบนทำ jaw แนวราบเสยลูก).
3. **พื้นด้าน** — `reflectance=0` (เดิม 0.05) ตัดเงาสะท้อนที่ทำให้ดูลูกลอย +
   attach เฉพาะตอน pinch ถึงระดับลูก (`_GRASP_ATTACH_ZGAP`) กันลูกกระโดด.

ผลลัพธ์: jaw คีบด้านข้างลูกดูธรรมชาติ, ปลายไม่แตะพื้น, ลูกไม่ลอย (jump ~3mm),
**success 100% (10/10)** ทุกสี.
→ `_GRASP_REF_QPOS`, `_target_rotation` ใน [`collect/ik.py`](../src/robot_learning/collect/ik.py);
  `approach_back`, radial approach ใน [`collect/fsm_policy.py`](../src/robot_learning/collect/fsm_policy.py)

---

## 8. ก้ามปู (นิ้วประกบซ้าย-ขวา) + 3 ทรง (ball/cube/cylinder) — ที่ใช้จริง

**เป้าหมาย**: คีบแบบ "ก้ามปู" (นิ้วสองข้างประกบซ้าย-ขวาแนวนอน แบบ parallel
gripper ตามภาพ Gripper-X) แทน "ตู้คีบตุ๊กตา" (นิ้วหุบบน-ล่าง) + รองรับ 3 ทรง.

**geometry ที่ค้นพบ**: แกน closing ของ jaw = **local x ของ gripper body**.
ก้ามปู = ทำให้ local-x ขนานพื้น (world-z ≈ 0) → นิ้วประกบแนวนอน. ได้จากท่า
`_GRASP_REF_QPOS = (0, 0.23, 1.15, -0.58, 1.68, 0.4)` — สำคัญคือ **wrist_roll≈96°**.

**ปัญหา workspace asymmetric + วิธีแก้** ("หมุนซ้ายติดก็หมุนขวาแทน"):
ก้ามปู wrist_roll=+96° เอื้อม y บวก (ซ้าย) ได้ แต่ y ลบ (ขวา) เอื้อมไม่ถึง
(err 25mm). แก้: IK เก็บ ref 2 ทิศ (`_ref_R_pos`/`_ref_R_neg`, roll ±) และ
**เลือกทิศตามฝั่ง y ของลูก** (`_target_rotation` + seed ใน `solve`) → reach
เต็ม 15/15 ทั้งซ้าย-ขวา.

**FSM approach จากบน** (`approach_back=0`): ก้ามปูนิ้วครอบลงมา 2 ข้าง ต้อง
approach จากบนตรงๆ (ต่างจาก side-grasp ที่สอดจากหลัง). attach zgap ผ่อนเป็น
0.022 (ก้ามปู z_gap ตอนคีบ ~17mm).

**3 ทรง**: `config/objects.yaml` มี ball(sphere)/cube(box)/cylinder. grasp-assist
weld คีบได้ทุกทรง (cube เหลี่ยม/cylinder สูงก็ติด). scene_builder รองรับ
cylinder (radius+half_length). task string: "pick up the `<color>` `<shape>` ...".

**ผลลัพธ์: success 100% (15/15) — ball 5/5, cube 5/5, cylinder 5/5.**
→ `_GRASP_REF_QPOS`, `_target_rotation`, `_ref_R_pos/neg` ใน [`collect/ik.py`](../src/robot_learning/collect/ik.py)

### 8.1 แก้ pinch site ให้ตรงปลายนิ้วจริง (สำคัญ — วัตถุเคยไปติด servo)

**อาการ**: วัตถุที่คีบไม่อยู่กลางปลายนิ้ว แต่ไปติดที่ **servo/โคน gripper** ตลอด.

**ต้นตอ**: pinch site เดิม (`pos="0.017 0 -0.0345"`) calibrate จาก **geom center**
ของ mesh ตอนแรก ซึ่งหลอก — geom center ของ mesh `sts3215` (servo) อยู่ใกล้จุดนั้น
ทำให้ pinch ไปเล็งที่ servo. วัดจริงพบ pinch เดิม **ห่างจุดหนีบจริง 66.6mm!**

**แก้**: คำนวณ **จุดกึ่งกลางปลายนิ้วจริง** จาก mesh vertices ของนิ้วทั้งสอง —
moving finger (`moving_jaw` mesh) + fixed finger (`wrist_roll_follower` mesh) —
หา region ปลายสุด (x มากสุด) แล้วเฉลี่ย → pinch ใหม่ `pos="0.008 0 -0.100"`
(ห่างจุดหนีบจริงแค่ 0.9mm). ต้อง re-calibrate `_GRASP_REF_QPOS` ใหม่ (pinch ลึกลง
6.6cm กระทบ IK). ผล: วัตถุอยู่ **กลางปลายนิ้ว** + success ยัง 100%.

> บทเรียน: อย่า calibrate จุดคีบจาก geom center — ใช้ **mesh vertices ของปลายนิ้ว**
> เท่านั้น (servo/มอเตอร์เป็น geom แยกที่ center อาจหลอกให้เข้าใจผิด). และต้องวัด
> ที่ **grip กลางๆ** + ยืนยันหลายระดับ grip (pinch ต้องคงที่ 0-1mm ทุก grip).

### 8.2 วางวัตถุ "ระดับก้นถาด" (ไม่ตกจากที่สูง) + ball ไม่กระเด็น

**อาการที่เจอ**:
1. *คีบทะลุ* — pinch เพี้ยน (ดู §8.1)
2. *episode จบก่อนวาง (viewer ปิด)* — `_check_success` เดิมเช็คแค่ "ลูกอยู่เหนือถาด"
   → success=True ตั้งแต่ลูก *ลอยผ่านเหนือถาด* (ยังคีบอยู่) ก่อน FSM ปล่อย.
   แก้: success ต้อง **ในถาด + ปล่อยแล้ว (weld ไม่ active) + z ต่ำ (ตกก้นถาด)**.
3. *ball กระเด็นตอนคีบ* — ball กลม นิ้วก้ามปูครอบลงมาดันกลิ้งหนีก่อน attach.
   แก้: เพิ่ม `_GRASP_ATTACH_DIST=0.055`, `_ZGAP=0.05` → จับตั้งแต่ descend (ก่อนนิ้วดัน).
4. *วางจากที่สูงกระเด็น* — เดิม RELEASE ปล่อยจากเหนือถาด 10cm → ball ตกกลิ้งออก.
   แก้: เพิ่มเฟส **`LOWER_TO_TRAY`** (ลง pinch ไปก้นถาด `tray_release_height=0.04`
   *ยังถือ*) ก่อน `RELEASE` (ปล่อยตรงนั้น) → วางเบาๆ ระดับก้นถาด ไม่ตก.

ผล: success 100% (15/15) ทุกทรง + วัตถุอยู่กลางนิ้ว + วางในถาดสมจริง.

> **หมายเหตุ**: IK ก้ามปู (orientation หมุน + 2 ทิศ) ช้ากว่าเดิม → เก็บ dataset
> ช้าลง. ถ้าต้องเก็บเยอะ พิจารณา cache IK solution ต่อ (color,shape,pos-grid)
> หรือลด iters (150 พอสำหรับ FSM).

---

## พารามิเตอร์ที่ปรับได้ (ถ้าอยากจูนต่อ)

| ที่ไหน | พารามิเตอร์ | ค่าปัจจุบัน | ผล |
|--------|-------------|-------------|-----|
| `ik.py` | `_GRASP_REF_QPOS` | (0,0.30,0.60,0.30,0,0.5) | ท่า side-grasp (jaw เฉียง ปลายไม่แตะพื้น) |
| `ik.py` | `_target_rotation` | หมุนตาม yaw ลูก | jaw เข้าหาลูกจากทิศ base→ลูก (reach ทั้งโซน) |
| `ik.py` | `_DEFAULT_ITERS` | 250 (150 ตอนเทส) | มาก=แม่น แต่ช้า |
| `scene_builder.py` | `ARM_KP_SCALE` / `ARM_FORCERANGE` | 3.0 / 10 | hold ท่าแม่น |
| `scene_base.xml` | `tablemat reflectance` | 0 | พื้นด้าน ไม่เงาสะท้อน (ลูกไม่ดูลอย) |
| `so101_env.py` | `_GRASP_ATTACH_DIST` / `_ZGAP` | 0.04 / 0.015 | ระยะ+ความสูง pinch ที่ weld จับ |
| `so101_env.py` | `_set_weld_relpose` | snap-to-pinch | ตรึงลูกที่จุดหนีบ (ดูสมจริง) |
| `fsm_policy.py` | `approach_back` | 0.06 | ถอยหลังตามแนว radial (side-grasp สอดเข้า) |
| `fsm_policy.py` | `grasp_z_offset` | 0.006 | pinch ที่ ~center ลูก |
| `fsm_policy.py` | `steps_move/descend/grasp` | 70/90/45 | เวลาแขนวิ่งต่อ phase |
| `objects.yaml` | `ball.radius` | 0.010 | ขนาดวัตถุ (ต้อง < ช่อง jaw) |
