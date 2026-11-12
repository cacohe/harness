from src.core.guard import NoOpGuard
from src.core.task import Task


def test_guard_allows_a_user_supplied_secret():
    NoOpGuard().guard_input(Task("call the API with api_key=sk-1234567890abcdefghij"))


def test_guard_allows_a_private_key_and_a_national_id():
    NoOpGuard().guard_input(Task("-----BEGIN PRIVATE KEY-----\nMIIB\nholder 11010519491231002X"))
