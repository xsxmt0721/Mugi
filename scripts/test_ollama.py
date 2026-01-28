import requests

def chat_test():
    url = "http://mugi-models:11434/api/generate"
    data = {
        "model": "deepseek-r1:14b",
        "prompt": "你好，请确认你的身份。",
        "stream": False
    }
    response = requests.post(url, json=data)
    print(response.json()['response'])

if __name__ == "__main__":
    chat_test()