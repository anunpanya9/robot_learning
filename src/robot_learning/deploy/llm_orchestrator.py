"""llm_orchestrator.py — แปลงคำสั่งภาษาธรรมชาติ → task queue ให้ SmolVLA.

ใช้ local LLM (Ollama) ทำ 2 หน้าที่:
  (ก) แปล/สกัด intent: คำพูด (ไทย/อังกฤษ) → instruction อังกฤษที่ SmolVLA เข้าใจ
      เช่น "หยิบลูกบอลสีแดง" → "pick up the red ball and put it in the tray"
  (ข) orchestrate หลายสเต็ป: "เก็บของสีแดงทั้งหมดแล้วเท" → คิวหลาย pick + dump

LLM ตอบเป็น JSON เท่านั้น (บังคับ schema) เพื่อ parse ต่อได้แน่นอน:
    {
      "steps": [
        {"color": "red", "action": "pick"},
        ...
      ],
      "dump_after": true|false
    }

โค้ดนี้ map color→task string ให้ SmolVLA เอง (ไม่ให้ LLM แต่ง string อิสระ
เพื่อให้ตรงกับ task ที่ train มา — สำคัญมากต่อความแม่นของ policy).

ต้องมี Ollama รันอยู่ (`ollama serve`) + pull model (`ollama pull qwen2.5`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from robot_learning.env.scene_randomizer import build_task_string
from robot_learning.env.scene_builder import SpawnedObject

# สีที่ระบบรองรับ (ตรงกับ config/objects.yaml)
SUPPORTED_COLORS = ["red", "yellow", "green", "blue", "white"]

SYSTEM_PROMPT = """You are the command parser for a robot arm that picks colored
balls and puts them in a tray. Convert the user's request (Thai or English) into
a strict JSON plan. Only these colors exist: red, yellow, green, blue, white.

Rules:
- "steps": ordered list of {"color": <one of the colors>, "action": "pick"}.
- If the user asks to collect ALL of one color, still emit one pick step for that
  color (the runtime repeats until none of that color remain).
- "dump_after": true if the user wants to empty/pour the tray after picking.
- Respond with ONLY the JSON object, no prose, no markdown fences.

Examples:
User: "หยิบลูกบอลสีแดงใส่ถาด"
JSON: {"steps": [{"color": "red", "action": "pick"}], "dump_after": false}

User: "เก็บของสีน้ำเงินทั้งหมดแล้วเท"
JSON: {"steps": [{"color": "blue", "action": "pick"}], "dump_after": true}

User: "pick the green ball then the yellow one"
JSON: {"steps": [{"color": "green", "action": "pick"}, {"color": "yellow", "action": "pick"}], "dump_after": false}
"""


@dataclass
class Plan:
    steps: list[dict]        # [{"color","action"}]
    dump_after: bool

    def task_strings(self) -> list[str]:
        """แปลงแต่ละ step → task string ให้ SmolVLA (ผ่าน build_task_string)."""
        out = []
        for s in self.steps:
            # ใช้ SpawnedObject จำลองเพื่อ reuse build_task_string (ให้ตรงกับ dataset)
            fake = SpawnedObject(
                name="", color=s["color"], shape="ball",
                rgba=(0, 0, 0, 1), geom_type="sphere", size=0.01, pos=(0, 0, 0),
            )
            out.append(build_task_string(fake))
        return out


class LLMOrchestrator:
    """คุย Ollama, คืน Plan. มี fallback แบบ keyword ถ้า LLM ใช้ไม่ได้."""

    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def parse(self, text: str) -> Plan:
        try:
            raw = self._ask_ollama(text)
            data = json.loads(raw)
            return self._validate(data)
        except Exception as e:  # fallback: keyword matching
            print(f"[LLM fallback] ({e}) → ใช้ keyword matching")
            return self._keyword_fallback(text)

    def _ask_ollama(self, text: str) -> str:
        import ollama

        client = ollama.Client(host=self.host)
        resp = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            format="json",       # บังคับ JSON output
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    def _validate(self, data: dict) -> Plan:
        steps = []
        for s in data.get("steps", []):
            color = str(s.get("color", "")).lower()
            if color in SUPPORTED_COLORS:
                steps.append({"color": color, "action": "pick"})
        if not steps:
            raise ValueError("ไม่พบสีที่รองรับใน LLM output")
        return Plan(steps=steps, dump_after=bool(data.get("dump_after", False)))

    def _keyword_fallback(self, text: str) -> Plan:
        """สกัดสีจากคำ (ไทย+อังกฤษ) ถ้า LLM ใช้ไม่ได้."""
        th = {"แดง": "red", "เหลือง": "yellow", "เขียว": "green",
              "น้ำเงิน": "blue", "ฟ้า": "blue", "ขาว": "white"}
        low = text.lower()
        found = [c for c in SUPPORTED_COLORS if c in low]
        found += [en for th_w, en in th.items() if th_w in text]
        # unique รักษาลำดับ
        seen, steps = set(), []
        for c in found:
            if c not in seen:
                seen.add(c)
                steps.append({"color": c, "action": "pick"})
        if not steps:
            raise ValueError(f"หาสีไม่เจอในคำสั่ง: {text!r}")
        dump = any(w in text for w in ["เท", "dump", "pour", "empty"])
        return Plan(steps=steps, dump_after=dump)
