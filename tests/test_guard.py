import pytest
from src.core.guard import GuardError, guard


class Task:
    def __init__(self, text: str):
        self.original_input = text


def test_guard_allows_a_user_supplied_secret():
    guard(Task("call the API with api_key=sk-1234567890abcdefghij"))


def test_guard_allows_a_private_key_and_a_national_id():
    guard(Task("-----BEGIN PRIVATE KEY-----\nMIIB\nholder 11010519491231002X"))
