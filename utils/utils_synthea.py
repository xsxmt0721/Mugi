import base64
import json
import os
import requests
import time
from core.config import SYNTHEA_API_URL

def synthea_get_data(config_json=None, patient_count=5, disease_module="lung_cancer"):
    """
    根据传入的 JSON 配置请求 Synthea 生成数据。

    config_json 可以是：
    - dict：直接传入配置对象
    - str：JSON 字符串，或 JSON 文件路径

    支持的常用字段（对应 Synthea CLI 选项）：
    - population / patient_count / p: 生成人数（-p）
    - module / disease_module / m / modules: 模块（-m）
    - gender / g: 性别（-g）
    - age_range / age / a: 年龄范围（-a，格式 "min-max"）
    - seed / s: 随机种子（-s）
    - clinician_seed / cs: 临床随机种子（-cs）
    - reference_date / r: 参考日期 YYYYMMDD（-r）
    - overflow_population / o: 溢出人口（-o）
    - local_config_file / c: 本地配置文件路径（-c）
    - modules_dir / d: 本地模块目录（-d）
    - initial_population_snapshot / i: 初始人口快照（-i）
    - updated_population_snapshot / u: 更新后人口快照（-u）
    - update_time_period_days / t: 更新时间段（-t）
    - fixed_record_path / f: 固定记录路径（-f）
    - keep_matching_patients_path / k: 保留匹配患者路径（-k）
    - state / city: 位置（作为位置参数）
    - config: 任意 synthea.properties 配置（--config*=value）
    """
    def _load_config(raw):
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            if os.path.isfile(raw):
                with open(raw, "r", encoding="utf-8") as f:
                    return json.load(f)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError("config_json 不是有效 JSON 字符串或文件路径")
        raise ValueError("config_json 仅支持 dict 或 str")

    def _normalize_gender(gender):
        if gender is None:
            return None
        g = str(gender).strip().lower()
        if g in {"m", "male", "man", "1"}:
            return "M"
        if g in {"f", "female", "woman", "0"}:
            return "F"
        return str(gender)

    def _age_to_range(age_val):
        if age_val is None:
            return None
        if isinstance(age_val, str):
            return age_val
        if isinstance(age_val, (int, float)):
            a = int(age_val)
            return f"{a}-{a}"
        if isinstance(age_val, dict):
            a_min = age_val.get("min")
            a_max = age_val.get("max")
            if a_min is None or a_max is None:
                return None
            return f"{int(a_min)}-{int(a_max)}"
        return None

    cfg = _load_config(config_json)

    # 兼容旧参数
    if not cfg:
        cfg = {
            "patient_count": patient_count,
            "disease_module": disease_module,
        }

    population = cfg.get("population", cfg.get("patient_count", cfg.get("p")))
    module = cfg.get("module", cfg.get("disease_module", cfg.get("m")))
    modules = cfg.get("modules")
    if module is None and isinstance(modules, list) and modules:
        module = ",".join([str(m) for m in modules if m])

    payload = {
        "p": population if population is not None else patient_count,
        "m": module if module is not None else disease_module,
    }

    options = []
    gender = _normalize_gender(cfg.get("gender", cfg.get("g")))
    if gender:
        options += ["-g", gender]

    age_range = cfg.get("age_range", cfg.get("a", cfg.get("age")))
    age_range = _age_to_range(age_range)
    if age_range:
        options += ["-a", age_range]

    seed = cfg.get("seed", cfg.get("s"))
    if seed is not None:
        options += ["-s", str(seed)]

    clinician_seed = cfg.get("clinician_seed", cfg.get("cs"))
    if clinician_seed is not None:
        options += ["-cs", str(clinician_seed)]

    reference_date = cfg.get("reference_date", cfg.get("r"))
    if reference_date:
        options += ["-r", str(reference_date)]

    overflow_population = cfg.get("overflow_population", cfg.get("o"))
    if overflow_population is not None:
        options += ["-o", str(overflow_population)]

    local_config_file = cfg.get("local_config_file", cfg.get("c"))
    if local_config_file:
        options += ["-c", str(local_config_file)]

    modules_dir = cfg.get("modules_dir", cfg.get("d"))
    if modules_dir:
        options += ["-d", str(modules_dir)]

    initial_snapshot = cfg.get("initial_population_snapshot", cfg.get("i"))
    if initial_snapshot:
        options += ["-i", str(initial_snapshot)]

    updated_snapshot = cfg.get("updated_population_snapshot", cfg.get("u"))
    if updated_snapshot:
        options += ["-u", str(updated_snapshot)]

    update_days = cfg.get("update_time_period_days", cfg.get("t"))
    if update_days is not None:
        options += ["-t", str(update_days)]

    fixed_record_path = cfg.get("fixed_record_path", cfg.get("f"))
    if fixed_record_path:
        options += ["-f", str(fixed_record_path)]

    keep_matching_path = cfg.get("keep_matching_patients_path", cfg.get("k"))
    if keep_matching_path:
        options += ["-k", str(keep_matching_path)]

    state = cfg.get("state")
    city = cfg.get("city")
    if state:
        payload["state"] = state
    if city:
        payload["city"] = city

    config_overrides = cfg.get("config")
    if isinstance(config_overrides, dict) and config_overrides:
        payload["config"] = config_overrides

    if options:
        payload["options"] = options

    # 使用服务名 mugi-synthea 直接访问
    api_url = SYNTHEA_API_URL

    print(f"--- 正在请求生产虚拟患者数据 ---")
    try:
        response = requests.post(api_url, json=payload, timeout=600)  # 生成可能较慢，设置长超时
        result = response.json()

        if result.get('status') == 'success':
            print(f"成功！本次生成了 {result.get('new_patients_count', 0)} 个新样本。")
            # 返回文件列表供后续数据解析模块使用
            return result.get('files', [])
        else:
            print(f"生成失败: {result.get('message')}")
    except Exception as e:
        print(f"通信错误: {e}")
    return []


def extract_medical_content(input_json_path, output_json_path=None):
    """
    提取诊断/症状/处方/检测等医学相关内容，解码 base64 文本并过滤无用信息。

    :param input_json_path: 输入 JSON 文件路径
    :param output_json_path: 输出 JSON 文件路径，若为 None 则仅返回字典
    :return: 处理后的字典
    """
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    def _first_reference(ref_list):
        if isinstance(ref_list, list) and ref_list:
            if isinstance(ref_list[0], dict):
                return ref_list[0].get("reference")
        return None

    def _extract_coding(codeable):
        if not isinstance(codeable, dict):
            return None
        coding = codeable.get("coding")
        if isinstance(coding, list) and coding:
            c0 = coding[0] if isinstance(coding[0], dict) else {}
            return {
                "code": c0.get("code"),
                "display": c0.get("display") or codeable.get("text"),
            }
        text = codeable.get("text")
        if text:
            return {"code": None, "display": text}
        return None

    def _decode_base64(data_str):
        if not data_str:
            return None
        try:
            raw = base64.b64decode(data_str)
            text = raw.decode("utf-8", errors="ignore")
            return text.strip() if text else None
        except Exception:
            return None

    data = _load_json(input_json_path)
    output = {
        "patient_id": None,
        "conditions": [],
        "observations": [],
        "procedures": [],
        "medications": [],
        "immunizations": [],
        "diagnostic_reports": [],
        "documents": [],
    }

    entries = data.get("entry") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        return output

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue
        r_type = resource.get("resourceType")

        if r_type == "Patient":
            output["patient_id"] = resource.get("id")
            continue

        if r_type == "Condition":
            output["conditions"].append({
                "code": _extract_coding(resource.get("code")),
                "clinical_status": _extract_coding(resource.get("clinicalStatus")),
                "verification_status": _extract_coding(resource.get("verificationStatus")),
                "onset": resource.get("onsetDateTime") or resource.get("onsetPeriod"),
                "recorded": resource.get("recordedDate"),
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type == "Observation":
            value = None
            if "valueQuantity" in resource:
                vq = resource.get("valueQuantity")
                value = {
                    "value": _safe_get(vq, "value"),
                    "unit": _safe_get(vq, "unit"),
                }
            elif "valueCodeableConcept" in resource:
                value = _extract_coding(resource.get("valueCodeableConcept"))
            else:
                value = resource.get("valueString") or resource.get("valueBoolean")

            output["observations"].append({
                "code": _extract_coding(resource.get("code")),
                "category": _extract_coding((resource.get("category") or [{}])[0]),
                "value": value,
                "effective": resource.get("effectiveDateTime") or resource.get("effectivePeriod"),
                "interpretation": _extract_coding((resource.get("interpretation") or [{}])[0]),
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type == "Procedure":
            output["procedures"].append({
                "code": _extract_coding(resource.get("code")),
                "performed": resource.get("performedDateTime") or resource.get("performedPeriod"),
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type in {"MedicationRequest", "MedicationAdministration", "MedicationStatement"}:
            med_code = _extract_coding(resource.get("medicationCodeableConcept"))
            output["medications"].append({
                "type": r_type,
                "medication": med_code,
                "status": resource.get("status"),
                "authored_on": resource.get("authoredOn"),
                "effective": resource.get("effectiveDateTime") or resource.get("effectivePeriod"),
                "dosage": resource.get("dosageInstruction"),
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type == "Immunization":
            output["immunizations"].append({
                "vaccine": _extract_coding(resource.get("vaccineCode")),
                "status": resource.get("status"),
                "occurrence": resource.get("occurrenceDateTime"),
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type == "DiagnosticReport":
            text_list = []
            for item in resource.get("presentedForm", []) or []:
                decoded = _decode_base64(_safe_get(item, "data"))
                if decoded:
                    text_list.append(decoded)
            output["diagnostic_reports"].append({
                "code": _extract_coding(resource.get("code")),
                "effective": resource.get("effectiveDateTime"),
                "issued": resource.get("issued"),
                "results": [_safe_get(r, "reference") for r in (resource.get("result") or [])],
                "text": text_list,
                "encounter": _safe_get(resource.get("encounter"), "reference"),
            })
            continue

        if r_type == "DocumentReference":
            text_list = []
            for content in resource.get("content", []) or []:
                attachment = _safe_get(content, "attachment")
                decoded = _decode_base64(_safe_get(attachment, "data"))
                if decoded:
                    text_list.append(decoded)
            if text_list:
                output["documents"].append({
                    "type": _extract_coding(resource.get("type")),
                    "date": resource.get("date"),
                    "text": text_list,
                    "encounter": _first_reference(_safe_get(resource.get("context"), "encounter")),
                })
            continue

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def is_lung_cancer_case(case_dict):
    """
    判断病例是否为肺癌（基于抽取后的病例字典）。
    :param case_dict: extract_medical_content 返回的字典
    :return: True/False
    """
    if not isinstance(case_dict, dict):
        return False

    keywords = [
        "lung cancer",
        "non-small cell lung",
        "small cell lung",
        "malignant neoplasm of lung",
        "neoplasm of lung",
        "carcinoma of lung",
        "suspected lung cancer",
        "pulmonary carcinoma",
    ]

    def _has_keywords(text):
        if not text:
            return False
        t = str(text).lower()
        return any(k in t for k in keywords)

    for cond in case_dict.get("conditions", []) or []:
        display = None
        if isinstance(cond, dict):
            code = cond.get("code") or {}
            if isinstance(code, dict):
                display = code.get("display") or code.get("code")
        if _has_keywords(display):
            return True

    for report in case_dict.get("diagnostic_reports", []) or []:
        if not isinstance(report, dict):
            continue
        for text in report.get("text", []) or []:
            if _has_keywords(text):
                return True

    for doc in case_dict.get("documents", []) or []:
        if not isinstance(doc, dict):
            continue
        for text in doc.get("text", []) or []:
            if _has_keywords(text):
                return True

    return False