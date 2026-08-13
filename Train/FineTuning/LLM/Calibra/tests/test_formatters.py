import pytest
from calibra.config import DataConfig
from calibra.data.formatters.common import normalize_messages
from calibra.data.formatters.sft import SFTFormatter


def test_normalize_missing_system():
    messages = normalize_messages([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}], system_prompt="system", add_system_prompt_if_missing=True)
    assert messages[0]["role"] == "system"


def test_agent_requires_messages():
    formatter = SFTFormatter(DataConfig(mode="agent_sft"))
    with pytest.raises(ValueError):
        formatter.prepare({"q": "hello", "a": "hi"})
