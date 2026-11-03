# agent-harness

Python framework for an agent harness.

The package owns the run loop, state transitions, input queue, approval order, tool scheduling, hooks, and trace events. Model calls, validators, tools, sandbox creation, memory, and approval UI are ports. Unimplemented ports raise `NotImplementedError` and the message starts with `TODO`.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

A host injects the ports, then runs:

```python
result = Harness(config, validator=..., context_builder=..., llm=..., approval_policy=..., approver=..., safety=..., executor=...).run("do the task")
```
