import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
from core.config import SYNTHEA_DATA_INPUT, SYNTHEA_DATA_VP
import json
from utils.utils_synthea import synthea_get_data, extract_medical_content

if __name__ == "__main__":
    '''
    # 定义 JSON 描述：60岁男性，检查肺癌，生成10份数据
    config = {
        "patient_count": 10,
        "disease_module": "lung_cancer",
        "gender": "M",
        "age": 60
    }

    print("测试配置:")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    patients = synthea_get_data(config_json=config)
    print(f"Generated {len(patients)} patients.")
    '''
    for file in os.listdir(SYNTHEA_DATA_INPUT):
        if file.endswith('.json'):
            out_path = os.path.join(SYNTHEA_DATA_VP, file)
            input_path = os.path.join(SYNTHEA_DATA_INPUT, file)
            extract_medical_content(input_path, out_path)            
            print(f"Processed file: {file}, output saved to: {out_path}")