import pytest
from unittest import mock

@pytest.fixture
def bob():
    return {"name": "Bob"}