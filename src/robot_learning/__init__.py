"""robot_learning — VLA pick & place บน SO-101 (MuJoCo → LeRobot → SmolVLA → Voice).

โครงสร้าง sub-package:
    env/      MuJoCo scene builder, Gymnasium environment, scene randomizer, กล้อง
    collect/  Scripted FSM policy + ตัวเก็บ LeRobotDataset (+ hook จอย Xbox)
    deploy/   โหลด policy + voice pipeline (mic→whisper→ollama→task) + orchestrator
    utils/    ค่าคงที่/พาธ (ชื่อ joint, asset path, ชื่อกล้อง)

ดูภาพรวม pipeline ทั้งหมดใน README.md และ docs/ ที่ repo root
"""

__version__ = "0.1.0"
