from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Mapping

from harness.types import ApprovalScope, ToolCall


def args_hash(arguments: Mapping[str, object]) -> str:
    payload = json.dumps(dict(arguments), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_scope(scope: ApprovalScope, call: ToolCall) -> ApprovalScope:
    if scope.kind == "session":
        return replace(scope, tool_name=scope.tool_name or call.name)
    if scope.kind == "exact":
        return replace(
            scope,
            tool_name=scope.tool_name or call.name,
            args_hash=scope.args_hash or args_hash(call.arguments),
        )
    return scope


def grant_covers(grant: ApprovalScope, call: ToolCall) -> bool:
    if grant.kind == "session":
        return grant.tool_name == call.name
    if grant.kind == "exact":
        return grant.tool_name == call.name and grant.args_hash == args_hash(call.arguments)
    return False
