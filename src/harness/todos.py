from __future__ import annotations


class TodoValidator:
    def validate(self, inputs, state):
        raise NotImplementedError("TODO: safety and domain validation")


class TodoContextBuilder:
    def build(self, args):
        raise NotImplementedError("TODO: assemble the model context")


class TodoLlmGateway:
    def complete(self, context, mode):
        raise NotImplementedError("TODO: call a model provider")


class TodoApprovalPolicy:
    def decide(self, call, policy, state):
        raise NotImplementedError("TODO: interpret human approval rules for this call")


class TodoApprover:
    def review(self, action, requirement):
        raise NotImplementedError("TODO: approval UI or automatic reviewer")


class TodoSafetyAugment:
    def check(self, action, environment):
        raise NotImplementedError("TODO: network, file, and sandbox checks")


class TodoExecutor:
    def execute(self, call, handle):
        raise NotImplementedError("TODO: execute a tool")


class TodoMemoryStore:
    def save(self, record):
        raise NotImplementedError("TODO: durable memory")

    def search(self, query):
        raise NotImplementedError("TODO: memory retrieval")


class TodoStateStore:
    def save(self, snapshot):
        raise NotImplementedError("TODO: persist run state")

    def load(self, run_id):
        raise NotImplementedError("TODO: load run state")


class TodoTraceExporter:
    def export(self, events):
        raise NotImplementedError("TODO: export traces")

    def replay(self, run_id):
        raise NotImplementedError("TODO: replay traces")
