#!/usr/bin/env python3
"""MCP-side world server with private world model and memory."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal
from urllib import error, request

ActionName = Literal["look", "move", "pickup", "drop", "say", "status"]


@dataclass
class ToolCall:
    name: ActionName
    args: Dict[str, str]


@dataclass
class RobotState:
    location: str = "lab"
    inventory: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    locations: Dict[str, Dict[str, object]] = field(
        default_factory=lambda: {
            "lab": {
                "description": "A clean robotics lab with blinking consoles.",
                "neighbors": ["hallway"],
                "items": ["battery"],
            },
            "hallway": {
                "description": "A narrow hallway with a service door.",
                "neighbors": ["lab", "storage"],
                "items": [],
            },
            "storage": {
                "description": "A dark storage room with spare parts.",
                "neighbors": ["hallway"],
                "items": ["wrench", "antenna"],
            },
        }
    )


class LLMResponsesClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 60) -> None:
        self.url = f"{base_url.rstrip('/')}/responses"
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
        }
        req = request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Не удалось подключиться к local server по Responses API. "
                f"Проверь что endpoint доступен: {self.url}"
            ) from exc

        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        output = data.get("output", [])
        if isinstance(output, list):
            for item in output:
                content = item.get("content", []) if isinstance(item, dict) else []
                if isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") in {"output_text", "text"}:
                            chunk_text = chunk.get("text")
                            if isinstance(chunk_text, str) and chunk_text.strip():
                                return chunk_text.strip()

        raise RuntimeError("Responses API вернул ответ без текстового контента.")


class LongTermMemory:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.touch()

    def append(self, role: str, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"role": role, "text": text}, ensure_ascii=False) + "\n")

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        query_tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9_]+", query.lower()))
        scored: List[tuple[int, Dict[str, str]]] = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get("text", ""))
                tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9_]+", text.lower()))
                scored.append((len(query_tokens.intersection(tokens)), {"role": str(row.get("role", "memory")), "text": text}))

        scored.sort(key=lambda item: item[0], reverse=True)
        relevant = [entry for score, entry in scored if score > 0][:top_k]
        return relevant if relevant else [entry for _, entry in scored[-top_k:]]


class WorldModel:
    SYSTEM_PROMPT = (
        "You are a world simulator for a robot. "
        "You receive canonical transition output and must narrate it naturally. "
        "Do not alter facts. Keep the answer brief (max 2 sentences)."
    )

    def __init__(self, llm: LLMResponsesClient, memory: LongTermMemory) -> None:
        self.state = WorldState()
        self.llm = llm
        self.memory = memory

    def handle_tool_call(self, robot: RobotState, call: ToolCall) -> str:
        raw_result = self._apply_rules(robot, call)
        payload = {
            "robot_location": robot.location,
            "robot_inventory": robot.inventory,
            "tool_call": {"name": call.name, "args": call.args},
            "raw_result": raw_result,
            "world": self.state.locations,
        }
        recalled = self.memory.recall(raw_result, top_k=3)
        memory_block = "\n".join(f"- {row['role']}: {row['text']}" for row in recalled)
        prompt = (
            "Сделай живое описание результата на базе JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n\nWorld private memory:\n"
            + memory_block
        )
        narration = self.llm.generate(self.SYSTEM_PROMPT, prompt, temperature=0.3)
        self.memory.append("world_narration", narration)
        return narration

    def _apply_rules(self, robot: RobotState, call: ToolCall) -> str:
        if call.name == "look":
            loc = self.state.locations[robot.location]
            items = ", ".join(loc["items"]) if loc["items"] else "nothing notable"
            neighbors = ", ".join(loc["neighbors"])
            return f"You are at {robot.location}. {loc['description']} Visible items: {items}. Paths: {neighbors}."
        if call.name == "move":
            target = call.args.get("to", "")
            loc = self.state.locations[robot.location]
            if target in loc["neighbors"]:
                robot.location = target
                return f"Moved to {target}."
            return f"Cannot move to '{target}' from {robot.location}."
        if call.name == "pickup":
            item = call.args.get("item", "")
            loc = self.state.locations[robot.location]
            items = loc["items"]
            if isinstance(items, list) and item in items:
                items.remove(item)
                robot.inventory.append(item)
                return f"Picked up {item}."
            return f"No '{item}' here."
        if call.name == "drop":
            item = call.args.get("item", "")
            if item in robot.inventory:
                robot.inventory.remove(item)
                items = self.state.locations[robot.location]["items"]
                if isinstance(items, list):
                    items.append(item)
                return f"Dropped {item}."
            return f"You are not carrying '{item}'."
        if call.name == "say":
            return f"Robot says: {call.args.get('text', '')}"
        if call.name == "status":
            inv = ", ".join(robot.inventory) if robot.inventory else "empty"
            return f"Location={robot.location}; Inventory={inv}"
        return f"Unknown tool: {call.name}"


class MCPWorldServer:
    """MCP side server wrapper around WorldModel."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def call_tool(self, robot: RobotState, tool: ToolCall) -> str:
        return self.world_model.handle_tool_call(robot, tool)
