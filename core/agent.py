import os
import time
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载 .env 配置文件
load_dotenv()

class MugiBaseAgent:
    """
    Mugi 外部 LLM 调用基类 (基于 OpenAI 协议适配 DeepSeek)
    用于指南提取、模拟数据生成等非本地推理任务
    """
    
    def __init__(self, system_prompt: str = "你是一个严谨的医疗人工智能助手。"):
        # 从环境变量读取配置
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("MODEL_NAME", "deepseek-chat")
        self.temperature = float(os.getenv("TEMPERATURE", 0.6))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 8192))
        self.timeout = int(os.getenv("TIMEOUT", 240))
        
        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # 初始系统提示词
        self.system_prompt = system_prompt

    def _get_payload(self, user_content: str, history: Optional[List[Dict]] = None) -> List[Dict]:
        """构造对话消息体"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def call(self, user_content: str, history: Optional[List[Dict]] = None) -> str:
        """
        同步调用方法
        :param user_content: 用户输入的指令或文本
        :param history: 可选的历史对话记录
        :return: 模型生成的文本内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._get_payload(user_content, history),
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            # 在 Mugi 框架中，日志记录至关重要
            print(f"Error during LLM API call: {e}")
            return f"ERROR: {str(e)}"

    def call_with_json(self, user_content: str) -> Dict[str, Any]:
        """
        强制要求 JSON 格式返回 (适用于指南逻辑提取和特征向量生成)
        """
        # 注意：DeepSeek 等模型支持 json_object 模式
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._get_payload(user_content),
                response_format={'type': 'json_object'},
                temperature=self.temperature
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {"error": str(e)}
        
if __name__ == "__main__":
    agent = MugiBaseAgent()
    result = agent.call("请简要介绍一下人工智能在医疗领域的应用。")
    print(result)