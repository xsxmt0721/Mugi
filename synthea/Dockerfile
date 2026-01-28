FROM eclipse-temurin:17-jdk-jammy

# 安装 Python 和 Flask
RUN apt-get update && \
    apt-get install -y python3 python3-pip git curl && \
    pip3 install flask && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 克隆并预构建 Synthea
RUN git clone https://github.com/synthetichealth/synthea.git .
RUN ./gradlew build -x test

# 配置 Synthea 导出 FHIR
RUN sed -i 's/exporter.fhir.export = false/exporter.fhir.export = true/g' \
    src/main/resources/synthea.properties

# 复制 API 脚本
COPY server.py /app/server.py

# 暴露端口（供虚拟网络内部使用）
EXPOSE 5000

# 启动 API 服务
ENTRYPOINT ["python3", "/app/server.py"]