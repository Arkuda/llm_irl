# LLM ↔ MCP Robot Freedom Experiment

Теперь код разделён на **2 Python-файла** и работает через **Responses API**:

- `simulation_mcp_experiment.py` — агентная часть (первая модель), которая считает, что она в роботе.
- `mcp_world_server.py` — MCP/server часть со второй моделью (симуляция мира + tools).
- Между ними — **MCP-like bridge** (`look`, `move`, `pickup`, `drop`, `say`, `status`).
- Память **раздельная**:
  - `agent_memory.jsonl` — только память агента,
  - `world_memory.jsonl` — только память мира.

Это нужно, чтобы агент не видел внутреннюю «кухню» симулятора.

## Что нужно поднять

1. Запусти локальный сервер, совместимый с OpenAI **Responses API**.
2. Убедись, что доступен `http://127.0.0.1:1234/v1/responses` (или укажи свой `--base-url`).
3. Укажи model id для агента и мира (можно одинаковые или разные).

## Запуск

```bash
python simulation_mcp_experiment.py \
  --steps 8 \
  --base-url http://127.0.0.1:1234/v1 \
  --agent-model qwen2.5-7b-instruct \
  --world-model qwen2.5-7b-instruct \
  --agent-memory-path agent_memory.jsonl \
  --world-memory-path world_memory.jsonl
```

---

## Что прописать в конфиге `{"mcpServers": {}}`

Если ты используешь MCP-конфиг формата:

```json
{
  "mcpServers": {}
}
```

добавь туда сервер твоего bridge, например так:

```json
{
  "mcpServers": {
    "robot-world-bridge": {
      "command": "python",
      "args": ["/ABS/PATH/TO/mcp_world_server.py"],
      "env": {
        "WORLD_MODEL_BASE_URL": "http://127.0.0.1:1234/v1",
        "WORLD_MODEL_NAME": "qwen2.5-7b-instruct",
        "WORLD_MEMORY_PATH": "/ABS/PATH/TO/world_memory.jsonl"
      }
    }
  }
}
```

> Где `mcp_world_server.py` — серверная часть со второй моделью, которая поднимает MCP tools (`look/move/pickup/drop/say/status`) и крутит world-логику.

---

## Как добавить MCP bridge в LM Studio (пример tool-loop)

Ниже минимальный шаблон tool loop через **Responses API**: модель вызывает tools, а ты прокидываешь их в свой bridge (`call_tool`), после чего отправляешь tool output обратно в Responses.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

TOOLS = [
    {
        "type": "function",
        "name": "look",
        "description": "Inspect current location",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "move",
        "description": "Move to neighboring location",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}},
            "required": ["to"],
            "additionalProperties": False,
        },
    },
]

def call_tool(name: str, args: dict) -> str:
    # Здесь дергаешь свой MCP bridge/server
    # return mcp_world_server.call_tool(name, args)
    return f"tool={name}, args={args}"

response = client.responses.create(
    model="qwen2.5-7b-instruct",
    instructions="You are a robot agent. Use tools.",
    input="Boot complete. Decide first action.",
    tools=TOOLS,
)

while True:
    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        print(response.output_text)
        break

    tool_outputs = []
    for call in tool_calls:
        args = json.loads(call.arguments or "{}")
        result = call_tool(call.name, args)
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            }
        )

    response = client.responses.create(
        model="qwen2.5-7b-instruct",
        previous_response_id=response.id,
        input=tool_outputs,
        tools=TOOLS,
    )
```

> Если LM Studio сборка ещё не поддерживает tools в `/responses`, можно временно оставить текущую схему (JSON action) и позже переключиться на function-calls без изменения логики world state.
