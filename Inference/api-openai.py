from openai import OpenAI

client = OpenAI(
    # api_key='sk-64dmCZkGx9urlHX6CYkh0Ro2XPVz5vRLmdQtzPzMeBk0Gqbj',
    # base_url='https://118.25.43.185/v1',

    #
    # api_key = 'sk-CrD0vEl2rfvQRaIm4PHSEKFSbztecs8gbEgkrjym4eMPiZai',
    # base_url = 'https://api.zhangsan.yun/v1',

    # 百炼
    # api_key='sk-ws-H.REPRILH.gDzQ.MEUCIBBDsiI6u1VSFm6cLNGJQwjPNCNtKNFZp_KVR6SZ8O57AiEAk9IJSRRLGFJiyQAkGRDEUNE44s_A-EtZQGmymJ7n3UM',
    # base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',

    # # 皮皮虾中转
    # api_key = 'sk-SR0TFoDRJPoKZaUjH2Mo2EYFiRKOsDASJDcFon1l2gE5Gu2C',
    # base_url = 'https://big.ppxsh.com/v1',

    # 万来code
    api_key = 'sk-dbcca64e5e39ba36e3b93f200dbf36c4877776d7b735ee0b0257ce1635b2963f',
    base_url = 'https://api.wanlai.ai/v1',
)

response = client.chat.completions.create(
    model="gpt-5.5",
    # model="qwen3.6-35b-a3b",
    # model="claude-sonnet-4-5-20250929",
    # model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "你好，你是谁？"}
    ],
)

print(response.choices[0].message.content)
