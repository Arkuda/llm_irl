#!/usr/bin/env python3
"""MCP-side world server with private world model and memory.

Can be used in two ways:
1) Imported as a Python module by `simulation_mcp_experiment.py`.
2) Launched as an MCP stdio server process (for LM Studio bridge).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal
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
    def __init__(self, base_url: str, model: str, timeout_s: int = 1800) -> None:
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


class MCPStdioBridge:
    """Minimal JSON-RPC MCP server for LM Studio MCP bridge."""

    def __init__(self, server: MCPWorldServer) -> None:
        self.server = server
        self.robot = RobotState()

    def run(self) -> None:
        while True:
            msg = self._read_message()
            if msg is None:
                break
            self._handle_message(msg)

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            self._send_result(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "robot-world-bridge", "version": "0.1.0"},
                },
            )
            return

        if method == "notifications/initialized":
            return

        if method == "tools/list":
            self._send_result(msg_id, {"tools": self._tools_schema()})
            return

        if method == "tools/call":
            name = str(params.get("name", "look"))
            args_raw = params.get("arguments", {})
            if not isinstance(args_raw, dict):
                args_raw = {}
            args = {str(k): str(v) for k, v in args_raw.items()}
            tool_call = ToolCall(name=name if name in {"look", "move", "pickup", "drop", "say", "status"} else "look", args=args)
            text = self.server.call_tool(self.robot, tool_call)
            self._send_result(msg_id, {"content": [{"type": "text", "text": text}]})
            return

        self._send_error(msg_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _tools_schema() -> List[Dict[str, Any]]:
        return [
            {
                "name": "look",
                "description": "Inspect current location",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "move",
                "description": "Move to neighboring location",
                "inputSchema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "pickup",
                "description": "Pick item in current location",
                "inputSchema": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "drop",
                "description": "Drop carried item",
                "inputSchema": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "say",
                "description": "Say phrase via robot speaker",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "status",
                "description": "Get robot status",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]

    @staticmethod
    def _read_message() -> Dict[str, Any] | None:
        headers: Dict[str, str] = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            key, value = line.decode("utf-8").split(":", 1)
            headers[key.strip().lower()] = value.strip()

        length = int(headers.get("content-length", "0"))
        if length <= 0:
            return None
        body = sys.stdin.buffer.read(length)
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def _send_message(payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()

    def _send_result(self, msg_id: Any, result: Dict[str, Any]) -> None:
        self._send_message({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _send_error(self, msg_id: Any, code: int, message: str) -> None:
        self._send_message({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def build_world_server_from_env() -> MCPWorldServer:
    base_url = os.getenv("WORLD_MODEL_BASE_URL", "http://127.0.0.1:1234/v1")
    model_name = os.getenv("WORLD_MODEL_NAME", "world-model")
    memory_path = os.getenv("WORLD_MEMORY_PATH", "world_memory.jsonl")
    timeout_s = int(os.getenv("WORLD_TIMEOUT_S", "1800"))
    llm = LLMResponsesClient(base_url=base_url, model=model_name, timeout_s=timeout_s)
    memory = LongTermMemory(memory_path)
    model = WorldModel(llm, memory)
    return MCPWorldServer(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP world bridge over stdio")
    parser.add_argument("--stdio", action="store_true", help="Run MCP JSON-RPC stdio server")
    args = parser.parse_args()

    if args.stdio:
        bridge = MCPStdioBridge(build_world_server_from_env())
        bridge.run()


if __name__ == "__main__":
    main()
