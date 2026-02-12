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
