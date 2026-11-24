import re

_OPAQUE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}")


def validate_approval_binding(approval_scope: str, workspace_id: str) -> None:
    for identity in (approval_scope, workspace_id):
        if type(identity) is not str or len(identity) > 256 or not _OPAQUE_ID.fullmatch(identity):
            raise ValueError("Approval scope and workspace must be opaque IDs")
