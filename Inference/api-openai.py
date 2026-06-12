from openai import OpenAI

client = OpenAI(
    # api_key='sk-64dmCZkGx9urlHX6CYkh0Ro2XPVz5vRLmdQtzPzMeBk0Gqbj',
    # base_url='https://118.25.43.185/v1',
    api_key = 'sk-CrD0vEl2rfvQRaIm4PHSEKFSbztecs8gbEgkrjym4eMPiZai',
    base_url = 'https://api.zhangsan.yun/v1'
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "user", "content": "你好，你在干嘛？"}
    ],
)

print(response.choices[0].message.content)
