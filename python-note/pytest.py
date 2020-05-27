import pytest
from unittest import mock

@pytest.fixture
def bob():
    return {"name": "Bob"}

assert result

@pytest.mark.parameterize('num1', 'num2', 'result'[
    (1,2,10),
    ("hello", "world", "hello world"),
    (3,5,7)
])
def test_num(num1, num2, result):
    assert file_name.function_name() == result