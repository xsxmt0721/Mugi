from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Mugi 系统在线", "service": "DeepSeek + Neo4j"}

# 这里定义你的 AI 业务逻辑接口
@app.post("/chat")
async def chat(user_input: dict):
    # 此处编写调用 Ollama 和 Neo4j 的代码
    return {"reply": "这是来自 DeepSeek 的回复"}

if __name__ == "__main__":
    # 必须监听 0.0.0.0 才能让容器外的 cpolar 访问到
    uvicorn.run(app, host="0.0.0.0", port=8000)