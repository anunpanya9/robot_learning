"""voice_pipeline.py — สั่งหุ่นด้วยเสียง: mic → Whisper → LLM → SmolVLA → MuJoCo.

Pipeline เต็ม (Phase D):
    1. อัดเสียงจาก mic (sounddevice)
    2. ถอดเสียง → ข้อความ (faster-whisper, รองรับไทย/อังกฤษ)
    3. LLM (Ollama) แปลงเป็น task queue (llm_orchestrator)
    4. รันแต่ละ task ด้วย SmolVLA policy ใน MuJoCo (run_policy)
    5. ถ้าสั่ง "เท" → routine dump_tray หลังเก็บครบ

รัน:
    PYTHONPATH=src python -m robot_learning.deploy.voice_pipeline \\
        --checkpoint outputs/smolvla_so101/checkpoints/last/pretrained_model \\
        [--model qwen2.5] [--seconds 4] [--text "สั่งพิมพ์แทนพูด"]

ไม่มี mic/checkpoint ก็ทดสอบส่วน LLM ได้ด้วย --text (ข้ามการอัดเสียง) และ
--dry-run (ข้ามการโหลด policy — แค่แสดง task queue).
"""

from __future__ import annotations

import argparse

from robot_learning.deploy.llm_orchestrator import LLMOrchestrator, Plan


def record_and_transcribe(seconds: float, whisper_model: str, language: str | None) -> str:
    """อัดเสียงจาก mic แล้วถอดเป็นข้อความด้วย faster-whisper."""
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    sr = 16000
    print(f"🎤 พูดได้เลย ({seconds} วินาที)...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language=language)
    text = " ".join(seg.text for seg in segments).strip()
    print(f"📝 ถอดเสียง: {text!r}")
    return text


def run_plan(plan: Plan, checkpoint: str | None, dry_run: bool, image_size: int) -> None:
    """รันแต่ละ task ใน plan ด้วย SmolVLA (หรือแค่แสดงถ้า dry_run)."""
    tasks = plan.task_strings()
    print(f"\n📋 แผนงาน ({len(tasks)} สเต็ป, dump={plan.dump_after}):")
    for i, t in enumerate(tasks):
        print(f"   {i+1}. {t}")

    if dry_run or checkpoint is None:
        print("\n(dry-run: ไม่โหลด policy — แค่แสดงแผน)")
        return

    from robot_learning.deploy.run_policy import PolicyRunner

    runner = PolicyRunner(checkpoint, image_size=image_size)
    runner.reset_scene()
    for t in tasks:
        print(f"\n▶️  รัน: {t}")
        ok = runner.run_task(t)
        print(f"   → {'สำเร็จ' if ok else 'ล้มเหลว'}")
    if plan.dump_after:
        print("\n🗑️  เท tray (routine dump_tray)")
        # dump routine: ปัจจุบันเป็น placeholder — เอียง/พลิกถาด (ทำใน Phase 2)
    runner.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="SmolVLA checkpoint (ไม่ใส่ = dry-run)")
    ap.add_argument("--model", default="qwen2.5", help="Ollama model")
    ap.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    ap.add_argument("--language", default=None, help="บังคับภาษา (th/en) หรือ auto")
    ap.add_argument("--seconds", type=float, default=4.0, help="ความยาวการอัดเสียง")
    ap.add_argument("--text", default=None, help="พิมพ์คำสั่งแทนพูด (ข้าม mic)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true", help="แค่แสดง task queue ไม่รัน policy")
    args = ap.parse_args()

    # 1-2. เสียง → ข้อความ (หรือใช้ --text)
    if args.text is not None:
        text = args.text
        print(f"📝 คำสั่ง (พิมพ์): {text!r}")
    else:
        text = record_and_transcribe(args.seconds, args.whisper_model, args.language)

    # 3. LLM → plan
    orc = LLMOrchestrator(model=args.model)
    plan = orc.parse(text)

    # 4-5. รัน
    run_plan(plan, args.checkpoint, args.dry_run, args.image_size)


if __name__ == "__main__":
    main()
