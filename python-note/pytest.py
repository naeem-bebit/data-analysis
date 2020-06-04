import pytest
import unittest
from unittest.mock import mock

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

def teardown(module)

mock_object = mock.Mock()
real_object = file_name.function_name()
self.assertIsInstance(mock_object, file_name.function_name)

#import  unittest
## pytest command

#lambda_function is the name of the lambda

pytest --cov=lambda_function --doctest-modules --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html --cov-report term-missing
pytest --cov=lambda_function --cov-report term-missing


pytest integration test
How to do integration test?
pytest integrationtest.py

let say we want to test it using command line
# content of test_sample.py
def test_answer(cmdopt):
    if cmdopt == "type1":
        print("first")
    elif cmdopt == "type2":
        print("second")
    assert 0  # to see what was printed

# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cmdopt", action="store", default="type1", help="my option: type1 or type2"
    )


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")

#CLI to test
pytest -q --cmdopt=type2