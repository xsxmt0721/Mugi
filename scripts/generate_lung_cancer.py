import sys
from pathlib import Path

# 将项目根目录添加到搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import time
from core.config import SYNTHEA_DATA_INPUT, SYNTHEA_DATA_VP
from utils.utils_synthea import (
    synthea_get_data,
    extract_medical_content,
    is_lung_cancer_case,
)


def main():
    """循环生成肺癌病例，直到达到目标数量。"""
    target_count = 20
    if len(sys.argv) >= 2:
        try:
            target_count = int(sys.argv[1])
        except ValueError:
            print("目标数量参数无效，使用默认值 20。")
    # 1. 配置：每批次 100 人
    config = {
        "patient_count": 100,
        "disease_module": "lung_cancer",
        "gender": "M",
        "age": {"min": 55, "max": 80},
    }

    print("==== Lung Cancer Case Collection ====")
    print("批次配置:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"目标肺癌病例数: {target_count}")

    os.makedirs(SYNTHEA_DATA_VP, exist_ok=True)

    collected = 0
    batch = 0

    while collected < target_count:
        batch += 1
        print(f"\n[Batch {batch}] 生成 100 份 FHIR 数据...")
        files = synthea_get_data(config_json=config)

        if not files:
            print("未生成任何患者文件，请检查 Synthea 服务或配置。")
            return

        print(f"Synthea 返回 {len(files)} 个新 FHIR 文件。开始筛选肺癌病例...")

        for fname in files:
            input_path = os.path.join(SYNTHEA_DATA_INPUT, fname)
            if not os.path.isfile(input_path):
                print(f"  ! 跳过：找不到输入文件 {input_path}")
                continue

            case_dict = extract_medical_content(input_path, None)
            if not is_lung_cancer_case(case_dict):
                continue

            out_path = os.path.join(SYNTHEA_DATA_VP, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(case_dict, f, ensure_ascii=False, indent=2)
            collected += 1
            print(f"  ✓ 收集肺癌病例 {collected}/{target_count}: {fname}")

            if collected >= target_count:
                break

        time.sleep(0.5)

    print(f"完成：共收集 {collected} 例肺癌病例。输出目录: {SYNTHEA_DATA_VP}")


if __name__ == "__main__":
    main()
