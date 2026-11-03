from harness import (
    AgentKind,
    ApprovalDecision,
    ApprovalRequirement,
    ApprovalScope,
    GenericTool,
    Harness,
    HarnessConfig,
    LlmContext,
    LlmOutput,
    MemoryRecord,
    ModelConfig,
    OutputMode,
    ToolCall,
    ToolResult,
    ValidationResult,
)


def config(**kwargs):
    base = {
        "agent_kind": AgentKind.TASK,
        "model": ModelConfig(provider="test", model="fake"),
    }
    base.update(kwargs)
    return HarnessConfig(**base)


class AllowAll:
    def validate(self, inputs, state):
        return ValidationResult(ok=True)


class RejectAll:
    def __init__(self, reason="rejected"):
        self.reason = reason

    def validate(self, inputs, state):
        return ValidationResult(ok=False, reason=self.reason)


class EchoBuilder:
    def __init__(self):
        self.args = []

    def build(self, args):
        self.args.append(args)
        return LlmContext(
            system_prompt=args.system_prompt,
            messages=tuple(args.state.messages),
            tools=tuple(args.tools),
            memories=tuple(args.memories),
            goal=args.state.goal,
            strategy=args.state.strategy,
        )


class ScriptedGateway:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.modes = []
        self.contexts = []

    def complete(self, context, mode):
        self.modes.append(mode)
        self.contexts.append(context)
        if not self.outputs:
            raise AssertionError("scripted gateway has no output left")
        return self.outputs.pop(0)


class StaticApproval:
    def __init__(self, requirement, scope_kind="once", action=None):
        self.requirement = requirement
        self.scope_kind = scope_kind
        self.action = action
        self.calls = []

    def decide(self, call, policy, state):
        self.calls.append(call)
        scope = ApprovalScope(kind=self.scope_kind, action_id=call.id, tool_name=call.name)
        action = self.action or GenericTool(name=call.name, args=dict(call.arguments))
        return ApprovalDecision(action=action, requirement=self.requirement, scope=scope)


class RecordingApprover:
    def __init__(self, verdict="approved"):
        self.verdict = verdict
        self.reviews = []

    def review(self, action, requirement):
        self.reviews.append((action, requirement))
        return self.verdict


class RecordingSafety:
    def __init__(self, verdict="allow"):
        self.verdict = verdict
        self.actions = []

    def check(self, action, environment):
        self.actions.append(action)
        return self.verdict


class RecordingExecutor:
    def __init__(self, execute=None):
        self.calls = []
        self._execute = execute

    def execute(self, call, handle):
        self.calls.append(call)
        if self._execute is not None:
            return self._execute(call, handle)
        return ToolResult(call_id=call.id, ok=True, output=f"ran:{call.name}")


class FixedMemory:
    def __init__(self, records):
        self.records = tuple(records)
        self.queries = []

    def save(self, record):
        return None

    def search(self, query):
        self.queries.append(query)
        return self.records


def text_output(text):
    return LlmOutput(text=text, tool_calls=(), finish_reason="stop")


def tool_output(*calls):
    return LlmOutput(text=None, tool_calls=tuple(calls), finish_reason="tool_calls")


def call(name, call_id=None, **arguments):
    return ToolCall(id=call_id or name, name=name, arguments=arguments)


def make_harness(
    llm,
    executor=None,
    approval=None,
    approver=None,
    safety=None,
    harness_config=None,
    **kwargs,
):
    ports = {
        "validator": AllowAll(),
        "context_builder": EchoBuilder(),
        "llm": llm,
        "approval_policy": approval or StaticApproval(ApprovalRequirement.SKIP),
        "approver": approver or RecordingApprover(),
        "safety": safety or RecordingSafety(),
        "executor": executor or RecordingExecutor(),
    }
    ports.update(kwargs)
    return Harness(harness_config or config(), **ports)


def remembered(content):
    return MemoryRecord(key="k", content=content)
