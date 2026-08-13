from calibra.rewards import CompositeReward, Reward
from calibra.tools import Tool, ToolRegistry


def test_tool_registry_and_composite_reward():
    registry = ToolRegistry()
    registry.add(Tool("add", "add values", lambda x, y: x + y, {"type": "object"}))
    assert registry.call("add", {"x": 2, "y": 3}) == 5
    reward = CompositeReward([Reward(lambda value: value), Reward(lambda value: 1)], [2, 3])
    assert reward(4) == 11
