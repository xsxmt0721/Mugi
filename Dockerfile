# 1. 依然使用 CUDA 12.4 + Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 2. 设置工作目录
WORKDIR /app

# 3. 安装 Python 3.10 (Ubuntu 22.04 自带，无需额外源)
# 同时安装 python3-pip 和必要的构建工具
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 确保 pip 本身是最新的
RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 复制依赖并安装
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 复制项目文件
COPY . .

CMD ["python3", "main.py"]