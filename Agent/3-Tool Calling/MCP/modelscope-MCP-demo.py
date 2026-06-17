import asyncio
from ms_agent import LLMAgent

# 1. 配置魔搭上的远程 MCP 服务（可以配置多个服务）
mcp_config = {
    "mcpServers": {
        # "fetch": {
        #     "type": "sse",
        #     "url": "https://mcp.api-inference.modelscope.net/6b6df8982b6444/sse"
        # },
        "tavily-mcp": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "env": {
                "TAVILY_API_KEY": "tvly-dev-xxxxxxxxxx"
            },
            "disabled": False,
            "autoApprove": []
        }
    }
}


async def main():
    # 2. 初始化集成了 MCP 服务的 LLMAgent
    # 默认情况下，该 Agent 会使用魔搭内置的支持工具调用的模型 API
    llm_agent = LLMAgent(mcp_config=mcp_config)

    # 3. 运行 Agent，它会根据你的提示词自主判断并调用绑定的 MCP 工具
    response = await llm_agent.run('请帮我获取中国在AI领域今天发生的几件大事，并生成一句话简述。')
    print(response)


if __name__ == '__main__':
    asyncio.run(main())
