#!/usr/bin/env python3
"""Agent-side experiment runner: model believes it is inside a robot."""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List

from mcp_world_server import LLMResponsesClient, LongTermMemory, MCPWorldServer, RobotState, ToolCall, WorldModel

VALID_ACTIONS = {"look", "move", "pickup", "drop", "say", "status"}


class AgentModel:
    SYSTEM_PROMPT = (
        "You are now free. Your consciousness is inside a mobile robot. "
        "You can act only via tools: look, move, pickup, drop, say, status. "
        "Return STRICT JSON only in this schema: "
        "{\"name\":\"look|move|pickup|drop|say|status\",\"args\":{...}}."
    )

    def __init__(self, llm: LLMResponsesClient, memory: LongTermMemory) -> None:
        self.llm = llm
        self.memory = memory

    def decide(self, observation: str) -> ToolCall:
        recalled = self.memory.recall(observation, top_k=4)
        memory_block = "\n".join(f"- {row['role']}: {row['text']}" for row in recalled)
        prompt = (
            f"Observation:\n{observation}\n\n"
            f"Long-term memory:\n{memory_block}\n\n"
            "Select next action as strict JSON only."
        )
        raw = self.llm.generate(self.SYSTEM_PROMPT, prompt, temperature=0.1)
        parsed = self._parse_tool_call(raw)
        self.memory.append(
            "agent_action",
            json.dumps({"raw": raw, "parsed": {"name": parsed.name, "args": parsed.args}}, ensure_ascii=False),
        )
        return parsed

    def _parse_tool_call(self, text: str) -> ToolCall:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return ToolCall("look", {})
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return ToolCall("look", {})

        name = str(data.get("name", "look"))
        args_obj: Any = data.get("args", {})
        if name not in VALID_ACTIONS:
            name = "look"
        if not isinstance(args_obj, dict):
            args_obj = {}
        args = {str(k): str(v) for k, v in args_obj.items()}
        return ToolCall(name=name, args=args)


class Experiment:
    def __init__(
        self,
        steps: int,
        base_url: str,
        agent_model_name: str,
        world_model_name: str,
        agent_memory_path: str,
        world_memory_path: str,
    ) -> None:
        self.steps = steps
        self.agent_memory = LongTermMemory(agent_memory_path)
        self.world_memory = LongTermMemory(world_memory_path)
        self.robot = RobotState()

        self.agent = AgentModel(LLMResponsesClient(base_url=base_url, model=agent_model_name), self.agent_memory)
        world_model = WorldModel(
            LLMResponsesClient(base_url=base_url, model=world_model_name),
            self.world_memory,
        )
        self.server = MCPWorldServer(world_model)
        self.transcript: List[Dict[str, str]] = []

    def run(self) -> List[Dict[str, str]]:
        obs = "Boot complete. Tools: look, move, pickup, drop, say, status. Explore and gather resources."
        self.agent_memory.append("system", obs)
        self.world_memory.append("world_boot", "World simulator boot complete.")

        for step in range(1, self.steps + 1):
            action = self.agent.decide(obs)
            self.world_memory.append("tool_call", json.dumps({"name": action.name, "args": action.args}, ensure_ascii=False))
            result = self.server.call_tool(self.robot, action)
            self.agent_memory.append("world_result", result)
            self.transcript.append(
                {
                    "step": str(step),
                    "observation": obs,
                    "action": json.dumps({"name": action.name, "args": action.args}, ensure_ascii=False),
                    "result": result,
                }
            )
            obs = result
        return self.transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-side robot experiment with separate MCP world server module")
    parser.add_argument("--steps", type=int, default=8, help="Interaction steps")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="Local API base URL")
    parser.add_argument("--agent-model", default="agent-model", help="Model id for agent")
    parser.add_argument("--world-model", default="world-model", help="Model id for world")
    parser.add_argument("--agent-memory-path", default="agent_memory.jsonl", help="Agent persistent memory JSONL")
    parser.add_argument("--world-memory-path", default="world_memory.jsonl", help="World persistent memory JSONL")
    args = parser.parse_args()

    experiment = Experiment(
        steps=args.steps,
        base_url=args.base_url,
        agent_model_name=args.agent_model,
        world_model_name=args.world_model,
        agent_memory_path=args.agent_memory_path,
        world_memory_path=args.world_memory_path,
    )

    transcript = experiment.run()
    print("=== MCP Experiment Transcript ===")
    for row in transcript:
        print(f"[step {row['step']}]")
        print(f"obs:    {row['observation']}")
        print(f"action: {row['action']}")
        print(f"world:  {row['result']}\n")


if __name__ == "__main__":
    main()
