from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: Mapping[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool | None = None, **kwargs: Any):
        if tool is not None:
            self.add(tool)
            return tool
        def decorator(handler: Callable[..., Any]):
            self.add(Tool(handler=handler, **kwargs))
            return handler
        return decorator

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise KeyError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(**dict(arguments or {}))

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]
