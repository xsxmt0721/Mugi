from flask import Flask, request, jsonify
import subprocess
import os
import glob

app = Flask(__name__)

# Synthea 输出的物理路径（容器内）
OUTPUT_DIR = "/app/output/fhir"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    count = data.get('p', 1)           # 生成人数
    module = data.get('m', '')         # 疾病模块
    modules = data.get('modules')      # 兼容 modules 列表
    options = data.get('options')      # 额外 CLI 参数列表
    config_overrides = data.get('config')  # synthea.properties 覆盖项
    state = data.get('state')
    city = data.get('city')
    
    # 记录生成前的最新文件，以便对比找出新生成的文件
    old_files = set(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
    
    # 构建命令
    cmd = ["./run_synthea", "-p", str(count)]
    if module:
        cmd += ["-m", str(module)]
    elif isinstance(modules, list) and modules:
        cmd += ["-m", ",".join([str(m) for m in modules if m])]

    if isinstance(options, list):
        cmd += [str(o) for o in options if o is not None]

    if isinstance(config_overrides, dict) and config_overrides:
        for k, v in config_overrides.items():
            cmd.append(f"--{k}={v}")

    if state:
        cmd.append(str(state))
        if city:
            cmd.append(str(city))
    
    try:
        # 执行生成
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 识别新生成的文件
        current_files = set(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
        new_files = list(current_files - old_files)
        
        return jsonify({
            "status": "success",
            "new_patients_count": len(new_files),
            "files": [os.path.basename(f) for f in new_files],
            "log": process.stdout[-200:] # 返回末尾日志
        })
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr}), 500

if __name__ == '__main__':
    # 监听 5000 端口
    app.run(host='0.0.0.0', port=5000)