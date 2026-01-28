import requests
from neo4j import GraphDatabase
import time

def test_neo4j():
    print("--- 1. 测试 Neo4j 连接 ---")
    uri = "bolt://mugi-db:7687"
    user = "neo4j"
    password = "mugi_password"
    
    try:
        # 增加重试机制，因为数据库启动可能稍慢
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN '连接成功' AS msg")
            print(f"✅ Neo4j 响应: {result.single()['msg']}")
        driver.close()
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")

def test_ollama():
    print("\n--- 2. 测试 Ollama 模型服务 ---")
    url = "http://mugi-models:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ Ollama 响应成功！")
            models = response.json().get('models', [])
            if not models:
                print("💡 提示：服务器已通，但尚未拉取任何模型（如 deepseek）。")
            else:
                print(f"📦 已发现模型: {[m['name'] for m in models]}")
        else:
            print(f"❌ Ollama 返回错误码: {response.status_code}")
    except Exception as e:
        print(f"❌ Ollama 连接失败: {e}")

if __name__ == "__main__":
    test_neo4j()
    test_ollama()