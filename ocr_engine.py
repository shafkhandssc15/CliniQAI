import os
import re
import json
import time
import base64
import requests
import joblib
import numpy as np
import torch
import easyocr
from PIL import Image
import io

_easyocr_reader = None
_clinical_model = None
_label_encoder = None
_feature_columns = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        gpu_available = torch.cuda.is_available()
        _easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
    return _easyocr_reader

def load_local_classifier():
    global _clinical_model, _label_encoder, _feature_columns
    if _clinical_model is None:
        model_path = os.path.join("models", "clinical_classifier.joblib")
        le_path = os.path.join("models", "label_encoder.joblib")
        cols_path = os.path.join("models", "feature_columns.joblib")
        if os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(cols_path):
            _clinical_model = joblib.load(model_path)
            _label_encoder = joblib.load(le_path)
            _feature_columns = joblib.load(cols_path)
    return _clinical_model, _label_encoder, _feature_columns

def extract_text_from_image(image_bytes):
    reader = get_easyocr_reader()
    start_time = time.time()
    results = reader.readtext(image_bytes)
    
    if not results:
        return "", time.time() - start_time

    # Group detected bounding boxes into horizontal lines using spatial clustering
    boxes_with_info = []
    for bbox, text, conf in results:
        if not text or not str(text).strip():
            continue
        y_min = min(p[1] for p in bbox)
        y_max = max(p[1] for p in bbox)
        x_min = min(p[0] for p in bbox)
        x_max = max(p[0] for p in bbox)
        y_center = (y_min + y_max) / 2.0
        h = max(1.0, float(y_max - y_min))
        boxes_with_info.append({
            "bbox": bbox, "text": str(text).strip(), "conf": conf,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "y_center": y_center, "h": h
        })

    boxes_with_info.sort(key=lambda b: b["y_center"])

    lines = []
    for b in boxes_with_info:
        placed = False
        for line in lines:
            line_y_avg = sum(item["y_center"] for item in line) / len(line)
            line_h_avg = sum(item["h"] for item in line) / len(line)
            # Boxes belong to the same line if their vertical centers are within half-line height
            if abs(b["y_center"] - line_y_avg) <= max(10.0, line_h_avg * 0.65):
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    # Sort lines top-to-bottom, and words within each line strictly left-to-right by x_min
    lines.sort(key=lambda line: sum(b["y_center"] for b in line) / len(line))
    for line in lines:
        line.sort(key=lambda b: b["x_min"])

    reconstructed_lines = [" ".join(b["text"] for b in line) for line in lines]
    full_text = "\n".join(reconstructed_lines)
    return full_text, time.time() - start_time

def extract_patient_demographics(text):
    result = {"name": None, "age": None, "gender": None}
    if not text:
        return result

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    EXCLUDE_TOKENS = {
        "doctor", "dr.", "dr ", "ref by", "referred by", "consultant", "physician",
        "hospital", "laboratory", "laboratories", "clinic", "center", "centre", "pathology",
        "test name", "investigation", "parameter", "profile", "specimen", "sample",
        "authorized", "authorise", "signature", "technologist", "pathologist", "biochemist",
        "report name", "printed on", "collection date", "reported on", "address", "phone",
        "රෝහල", "සායනය", "පරීක්ෂණ", "වෛද්‍ය", "மருத்துவமனை", "ஆய்வகம்", "பரிசோதனை"
    }

    NAME_PREFIXES = [
        # English
        r'patient\s*name\s*[:\-#.]?',
        r'name\s*of\s*patient\s*[:\-#.]?',
        r'patient\s*\'?s?\s*name\s*[:\-#.]?',
        r'pt\.?\s*name\s*[:\-#.]?',
        r'client\s*name\s*[:\-#.]?',
        r'customer\s*name\s*[:\-#.]?',
        r'patient\s*[:\-#.]',
        r'\bname\s*[:\-#.]',
        # Sinhala (රෝගියාගේ නම, රෝගී නම, නම)
        r'රෝගියාගේ\s*නම\s*[:\-#.]?',
        r'රෝගී\s*නම\s*[:\-#.]?',
        r'රෝගියා\s*[:\-#.]?',
        r'නම\s*[:\-#.]?',
        # Tamil (நோயாளி பெயர், நோயாளியின் பெயர், பெயர்)
        r'நோயாளி(?:யின்)?\s*பெயர்\s*[:\-#.]?',
        r'நோயாளி\s*[:\-#.]?',
        r'பெயர்\s*[:\-#.]?',
        # Transliterations
        r'rogi\s*nama\s*[:\-#.]?',
        r'rogiyage\s*nama\s*[:\-#.]?',
        r'noyali\s*peyar\s*[:\-#.]?',
        r'peyar\s*[:\-#.]?',
        r'nama\s*[:\-#.]?'
    ]

    TITLE_PATTERN = r'\b(?:mr\.|mrs\.|ms\.|miss|master|mast\.|dr\.|rev\.|baby|b/o|shri|smt|ven\.|මයා|මිය|මෙනවිය|පූජ්‍ය|හිමි|திரு|திருமதி|செல்வி)\b'

    for line in lines:
        ll = line.lower().strip()

        # Age (English, Sinhala: වයස, Tamil: வயது)
        if result["age"] is None:
            age_m = re.search(r'\b(?:age|age\s*/\s*sex|age\s*/\s*gender|වයස|வயது)\s*[:/\-]?\s*(\d{1,3})', line, re.IGNORECASE)
            if age_m:
                v = int(age_m.group(1))
                if 1 <= v <= 120:
                    result["age"] = v
            if result["age"] is None:
                age_m2 = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?|y\b|වසර|வருடங்கள்)', line, re.IGNORECASE)
                if age_m2:
                    v = int(age_m2.group(1))
                    if 1 <= v <= 120:
                        result["age"] = v

        # Gender (English, Sinhala: ස්ත්‍රී / පුරුෂ, Tamil: பெண் / ஆண்)
        if result["gender"] is None:
            if re.search(r'\b(?:sex|gender)\s*[:/\-]?\s*(?:female|f\b)', ll) or re.search(r'\b(female|woman)\b', ll) or any(s in line for s in ["ස්ත්‍රී", "ගැහැණු", "பெண்"]):
                result["gender"] = "Female"
            elif re.search(r'\b(?:sex|gender)\s*[:/\-]?\s*(?:male|m\b)', ll) or re.search(r'\b(male|man)\b', ll) or any(s in line for s in ["පුරුෂ", "පිරිමි", "ஆண்"]):
                if 'fe' not in ll.split('male')[0][-2:]:
                    result["gender"] = "Male"
            elif re.search(r'\b\d{1,3}\s*/\s*f\b', ll):
                result["gender"] = "Female"
            elif re.search(r'\b\d{1,3}\s*/\s*m\b', ll):
                result["gender"] = "Male"

        # Name Extraction (Multilingual: English, Sinhala, Tamil, Arabic)
        if result["name"] is None:
            has_excluded = any(ex in ll for ex in EXCLUDE_TOKENS)
            if has_excluded and not any(k in ll for k in ["patient name", "pt name", "name of patient", "pt. name", "රෝගියා", "නම", "பெயர்", "நோயாளி"]):
                continue

            for p_pat in NAME_PREFIXES:
                match = re.search(rf'{p_pat}\s*([^\n\r\t|;#]+)', line, re.IGNORECASE)
                if match:
                    raw_candidate = match.group(1).strip()
                    # Strip trailing metadata like age, sex, date, sample no, etc.
                    raw_candidate = re.split(r'\b(?:age|sex|gender|date|ref|dr\b|doctor|pid|uhid|ipd|opd|reg|id|bill|lab\s*no|sample|වයස|ස්ත්‍රී|පුරුෂ|வயது|ஆண்|பெண்)\b|[|;\t]', raw_candidate, flags=re.IGNORECASE)[0].strip()
                    clean = re.sub(r'^[^\w\u0D80-\u0DFF\u0B80-\u0BFF]+|[^\w\u0D80-\u0DFF\u0B80-\u0BFF.]+$', '', raw_candidate).strip()
                    clean = re.sub(r'\s+', ' ', clean)
                    clean = re.sub(r'[\d_#=+\\/*]', '', clean).strip(' ,:;-')
                    words = clean.split()
                    filtered_words = [w for w in words if w.lower() not in {"years", "year", "yrs", "yr", "male", "female", "y", "m", "f", "patient", "name", "report", "වයස", "නම", "பெயர்", "வயது", "ස්ත්‍රී", "පුරුෂ"}]
                    clean = " ".join(filtered_words)
                    if len(clean) >= 2 and re.search(r'[\w\u0D80-\u0DFF\u0B80-\u0BFF]', clean) and not any(k in clean.lower() for k in ["hemoglobin", "glucose", "calcium", "serum", "blood", "urine", "count", "profile", "test", "laboratory", "hospital", "pathology"]):
                        result["name"] = clean.strip()
                        break

        # Fallback: Title Matching
        if result["name"] is None:
            has_excluded = any(ex in ll for ex in EXCLUDE_TOKENS)
            if not has_excluded:
                title_match = re.search(rf'({TITLE_PATTERN}\s+[^\n\r\t|;#]+)', line, re.IGNORECASE)
                if title_match:
                    raw_candidate = title_match.group(1).strip()
                    raw_candidate = re.split(r'\b(?:age|sex|gender|date|ref|pid|uhid|reg|id|වයස|ස්ත්‍රී|පුරුෂ|வயது)\b|[|;\t]', raw_candidate, flags=re.IGNORECASE)[0].strip()
                    clean = re.sub(r'[\d_#=+\\/*]', '', raw_candidate).strip(' ,:;-')
                    clean = re.sub(r'\s+', ' ', clean)
                    if len(clean) >= 3 and not any(k in clean.lower() for k in ["doctor", "dr.", "consultant"]):
                        result["name"] = clean.strip()

    return result

def extract_smart_biomarker(line: str, min_val: float, max_val: float, unit_hints: list = None) -> float:
    """
    Extracts the clinical patient test result value from a report line.
    Accurately ignores reference range columns, bracketed ranges, and hyphenated ranges.
    Prioritizes:
    1. Numbers immediately preceding known measurement units
    2. Numbers following the biomarker name before reference ranges
    3. Non-reference-range isolated numbers within physiological limits
    """
    if not line or not line.strip():
        return None

    # 1. Check if a number is directly followed by a recognized measurement unit
    if unit_hints:
        for u in unit_hints:
            pattern = re.compile(rf'(\d+(?:\.\d+)?)\s*(?:{re.escape(u)})\b', re.IGNORECASE)
            m = pattern.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if min_val <= val <= max_val:
                        return val
                except ValueError:
                    pass

    # 2. Identify and sanitize explicit reference ranges:
    # Matches: (0.55 - 4.78), [0.55 - 4.78], 0.55 - 4.78, 0.55 – 4.78, 0.55 to 4.78, 0.55..4.78
    clean_line = re.sub(r'[\(\[\{]?\s*\d+(?:\.\d+)?\s*(?:[-–—]|to|\.\.)\s*\d+(?:\.\d+)?\s*[\)\]\}]?', ' [REF_RANGE] ', line, flags=re.IGNORECASE)
    clean_line = re.sub(r'(?:ref\.?\s*range|reference|normal|ref|biological\s*ref|range)\s*[:\-]?\s*[\(\[\{]?\s*[\d\.\-\–\—\s\<\>to]+\s*[\)\]\}]?', ' [REF_RANGE] ', clean_line, flags=re.IGNORECASE)

    # 3. Find candidate numbers in the range-cleaned line
    num_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    nums = num_pattern.findall(clean_line)
    if nums:
        for n in nums:
            try:
                val = float(n)
                if min_val <= val <= max_val:
                    return val
            except ValueError:
                pass

    # 4. Fallback: Take the first candidate number from the original line (in left-to-right order)
    all_nums = num_pattern.findall(line)
    if all_nums:
        for n in all_nums:
            try:
                val = float(n)
                if min_val <= val <= max_val:
                    return val
            except ValueError:
                pass

    return None

def parse_biomarkers(text):
    text_lower = text.lower()
    lines = text_lower.split('\n')
    extracted = {
        "Age": None, "Gender": None, "Hemoglobin": None, "WBC": None, "Platelets": None,
        "Fasting_Blood_Sugar": None, "Creatinine": None, "TSH": None, "Cholesterol": None,
        "Calcium": None, "Ionized_Calcium": None, "GGT": None, "Albumin": None,
        "Globulin": None, "AG_Ratio": None, "Total_Protein": None, "ALT": None,
        "AST": None, "ALP": None, "Bilirubin": None, "Vitamin_D": None,
        "Vitamin_B12": None, "Ferritin": None, "ESR": None, "eGFR": None
    }
    
    for line in lines:
        if "age" in line and extracted["Age"] is None:
            match = re.search(r'\b(?:age|yr|yrs|years|වයස|வயது)\b.*?\b(\d{1,3})\b', line)
            if match:
                v = int(match.group(1))
                if 1 <= v <= 120: extracted["Age"] = v
        if ("gender" in line or "sex" in line) and extracted["Gender"] is None:
            if any(x in line for x in ["female", " f ", "/f", "ස්ත්‍රී", "ගැහැණු", "பெண்"]):
                extracted["Gender"] = "Female"
            elif any(x in line for x in ["male", " m ", "/m", "පුරුෂ", "පිරිමි", "ஆண்"]):
                extracted["Gender"] = "Male"

        if any(k in line for k in ["hemoglobin", "hb", "hgb", "hemo"]) and extracted["Hemoglobin"] is None:
            val = extract_smart_biomarker(line, 3.0, 24.0, ["g/dl", "g/l", "g%", "gm/dl"])
            if val is not None: extracted["Hemoglobin"] = val

        if any(k in line for k in ["wbc", "white blood", "leukocyte", "tcl", "w.b.c"]) and extracted["WBC"] is None:
            val = extract_smart_biomarker(line, 0.5, 100.0, ["10^3", "10*3", "x10^3", "/cumm", "/ul", "/mm3", "k/ul", "cells/cumm", "10^9/l"])
            if val is not None: extracted["WBC"] = val

        if any(k in line for k in ["platelet", "plt", "thrombocyte", "p.l.t"]) and extracted["Platelets"] is None:
            val = extract_smart_biomarker(line, 10.0, 1500.0, ["10^3", "10*3", "x10^3", "/cumm", "/ul", "/mm3", "k/ul", "lakhs", "10^9/l"])
            if val is not None: extracted["Platelets"] = int(val)

        if any(k in line for k in ["glucose", "fbs", "blood sugar", "sugar", "fasting"]) and extracted["Fasting_Blood_Sugar"] is None:
            val = extract_smart_biomarker(line, 30.0, 800.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None: extracted["Fasting_Blood_Sugar"] = int(val)

        if any(k in line for k in ["creatinine", "creat", "crea", "sr. creatinine"]) and extracted["Creatinine"] is None:
            val = extract_smart_biomarker(line, 0.1, 25.0, ["mg/dl", "mg%", "umol/l", "µmol/l"])
            if val is not None: extracted["Creatinine"] = val

        if any(k in line for k in ["egfr", "gfr", "estimated gfr"]) and extracted["eGFR"] is None:
            val = extract_smart_biomarker(line, 3.0, 200.0, ["ml/min", "ml/min/1.73", "ml/min/1.73m2"])
            if val is not None: extracted["eGFR"] = val

        if any(k in line for k in ["tsh", "thyroid stimulating", "t.s.h"]) and extracted["TSH"] is None:
            val = extract_smart_biomarker(line, 0.001, 250.0, ["ulu/ml", "uiu/ml", "miu/l", "u/ml", "iu/l", "µiu/ml", "ulU/ml"])
            if val is not None: extracted["TSH"] = val

        if any(k in line for k in ["cholesterol", "chol", "total chol"]) and not any(x in line for x in ["hdl", "ldl"]) and extracted["Cholesterol"] is None:
            val = extract_smart_biomarker(line, 40.0, 800.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None: extracted["Cholesterol"] = int(val)

        if any(k in line for k in ["ionized ca", "ionized calcium", "ca++", "free ca"]) and extracted["Ionized_Calcium"] is None:
            val = extract_smart_biomarker(line, 0.5, 10.0, ["mg/dl", "mmol/l"])
            if val is not None:
                if 0.5 <= val <= 3.0:
                    extracted["Ionized_Calcium"] = round(val * 4.0, 2)
                else:
                    extracted["Ionized_Calcium"] = val

        if any(k in line for k in ["calcium", "serum calcium", "ca"]) and not any(x in line for x in ["ionized", "ca++"]) and extracted["Calcium"] is None:
            val = extract_smart_biomarker(line, 3.0, 25.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None:
                if 3.0 <= val <= 6.5 and extracted["Ionized_Calcium"] is None:
                    extracted["Ionized_Calcium"] = val
                elif val > 6.5:
                    extracted["Calcium"] = val

        if any(k in line for k in ["ggt", "gamma gt", "gamma-gt", "ggtp"]) and extracted["GGT"] is None:
            val = extract_smart_biomarker(line, 1.0, 2000.0, ["u/l", "iu/l", "u/lt"])
            if val is not None: extracted["GGT"] = val

        if any(k in line for k in ["albumin", "alb"]) and not any(x in line for x in ["ratio", "a/g"]) and extracted["Albumin"] is None:
            val = extract_smart_biomarker(line, 0.5, 12.0, ["g/dl", "g/l", "g%"])
            if val is not None: extracted["Albumin"] = val

        if any(k in line for k in ["globulin", "glob"]) and not any(x in line for x in ["ratio", "a/g"]) and extracted["Globulin"] is None:
            val = extract_smart_biomarker(line, 0.5, 12.0, ["g/dl", "g/l", "g%"])
            if val is not None: extracted["Globulin"] = val

        if any(k in line for k in ["a/g ratio", "ag ratio", "a/g", "a:g"]) and extracted["AG_Ratio"] is None:
            val = extract_smart_biomarker(line, 0.1, 10.0, ["ratio", ":1"])
            if val is not None: extracted["AG_Ratio"] = val

        if any(k in line for k in ["total protein", "s. protein"]) and extracted["Total_Protein"] is None:
            val = extract_smart_biomarker(line, 1.0, 20.0, ["g/dl", "g/l", "g%"])
            if val is not None: extracted["Total_Protein"] = val

        if any(k in line for k in ["alt", "sgpt"]) and extracted["ALT"] is None:
            val = extract_smart_biomarker(line, 1.0, 3000.0, ["u/l", "iu/l", "u/lt"])
            if val is not None: extracted["ALT"] = val

        if any(k in line for k in ["ast", "sgot"]) and extracted["AST"] is None:
            val = extract_smart_biomarker(line, 1.0, 3000.0, ["u/l", "iu/l", "u/lt"])
            if val is not None: extracted["AST"] = val

        if any(k in line for k in ["alp", "alkaline phosphatase"]) and extracted["ALP"] is None:
            val = extract_smart_biomarker(line, 5.0, 3000.0, ["u/l", "iu/l", "u/lt"])
            if val is not None: extracted["ALP"] = val

        if any(k in line for k in ["bilirubin"]) and not any(x in line for x in ["direct", "indirect"]) and extracted["Bilirubin"] is None:
            val = extract_smart_biomarker(line, 0.05, 50.0, ["mg/dl", "umol/l", "mg%"])
            if val is not None: extracted["Bilirubin"] = val

        if any(k in line for k in ["vitamin d", "vit d", "25-oh"]) and extracted.get("Vitamin_D") is None:
            val = extract_smart_biomarker(line, 1.0, 300.0, ["ng/ml", "nmol/l", "ug/l"])
            if val is not None: extracted["Vitamin_D"] = val

        if any(k in line for k in ["vitamin b12", "vit b12", "b12"]) and extracted.get("Vitamin_B12") is None:
            val = extract_smart_biomarker(line, 10.0, 3000.0, ["pg/ml", "pmol/l", "ng/l"])
            if val is not None: extracted["Vitamin_B12"] = int(val)

        if any(k in line for k in ["ferritin"]) and extracted.get("Ferritin") is None:
            val = extract_smart_biomarker(line, 1.0, 5000.0, ["ng/ml", "ug/l", "pmol/l"])
            if val is not None: extracted["Ferritin"] = int(val)

        if any(k in line for k in ["esr", "erythrocyte sedimentation"]) and extracted.get("ESR") is None:
            val = extract_smart_biomarker(line, 1.0, 200.0, ["mm/hr", "mm/1st hr", "mm"])
            if val is not None: extracted["ESR"] = int(val)

    defaults = {
        "Age": 45, "Gender": "Female", "Hemoglobin": 13.5, "WBC": 6.8, "Platelets": 260,
        "Fasting_Blood_Sugar": 85, "Creatinine": 0.85, "TSH": 1.8, "Cholesterol": 165
    }
    extended_markers = [
        "Vitamin_D", "Vitamin_B12", "Calcium", "Ionized_Calcium", "GGT",
        "Albumin", "Globulin", "AG_Ratio", "Total_Protein", "ALT", "AST",
        "ALP", "Bilirubin", "Ferritin", "ESR", "eGFR"
    ]
    final_values = {}
    extracted_flags = {}
    for key, def_val in defaults.items():
        if extracted.get(key) is not None:
            final_values[key] = extracted[key]
            extracted_flags[key] = True
        else:
            final_values[key] = def_val
            extracted_flags[key] = False
    for em in extended_markers:
        if extracted.get(em) is not None:
            final_values[em] = extracted[em]
            extracted_flags[em] = True
        else:
            final_values[em] = None
            extracted_flags[em] = False
            
    return final_values, extracted_flags

CLINICAL_RANGES = {
    "Hemoglobin":         {"low": 12.0, "high": 17.5, "unit": "g/dL",      "critical_low": 7.0,  "critical_high": 20.0},
    "WBC":                {"low": 4.5,  "high": 11.0, "unit": "×10³/µL",   "critical_low": 2.0,  "critical_high": 30.0},
    "Platelets":          {"low": 150,  "high": 450,  "unit": "×10³/µL",   "critical_low": 50,   "critical_high": 1000},
    "Fasting_Blood_Sugar":{"low": 70,   "high": 100,  "unit": "mg/dL",     "critical_low": 50,   "critical_high": 400},
    "Creatinine":         {"low": 0.6,  "high": 1.2,  "unit": "mg/dL",     "critical_low": 0.2,  "critical_high": 10.0},
    "TSH":                {"low": 0.4,  "high": 4.5,  "unit": "mIU/L",     "critical_low": 0.01, "critical_high": 50.0},
    "Cholesterol":        {"low": 120,  "high": 200,  "unit": "mg/dL",     "critical_low": 80,   "critical_high": 400},
    "Ionized_Calcium":    {"low": 4.6,  "high": 5.3,  "unit": "mg/dL",     "critical_low": 3.8,  "critical_high": 6.5},
    "Calcium":            {"low": 8.5,  "high": 10.2,  "unit": "mg/dL",     "critical_low": 7.0,  "critical_high": 12.0},
    "GGT":                {"low": 9.0,  "high": 50.0,  "unit": "U/L",       "critical_low": 0.0,  "critical_high": 300.0},
    "Albumin":            {"low": 3.5,  "high": 5.5,  "unit": "g/dL",      "critical_low": 2.0,  "critical_high": 6.5},
    "AG_Ratio":           {"low": 1.2,  "high": 2.2,  "unit": "ratio",     "critical_low": 0.6,  "critical_high": 3.5},
    "Total_Protein":      {"low": 6.0,  "high": 8.3,  "unit": "g/dL",      "critical_low": 4.5,  "critical_high": 10.0},
    "Vitamin_D":          {"low": 30.0, "high": 100.0, "unit": "ng/mL",     "critical_low": 10.0, "critical_high": 150.0},
    "Vitamin_B12":        {"low": 200.0,"high": 900.0, "unit": "pg/mL",     "critical_low": 150.0,"critical_high": 2000.0},
    "Ferritin":           {"low": 30.0, "high": 300.0, "unit": "ng/mL",     "critical_low": 10.0, "critical_high": 1000.0},
    "ESR":                {"low": 0.0,  "high": 20.0,  "unit": "mm/hr",     "critical_low": 0.0,  "critical_high": 60.0},
}

def _is_borderline(val, low, high):
    margin_low  = (high - low) * 0.1
    margin_high = (high - low) * 0.1
    if low <= val < low + margin_low: return "borderline_low"
    if high - margin_high < val <= high: return "borderline_high"
    return None

def run_rule_based_engine(values):
    primary_findings = []
    borderline = []
    cautions = []
    
    age = values.get("Age", 45)
    gender = values.get("Gender", "Unknown")
    is_female = (gender or "").lower().startswith("f")

    hb = values.get("Hemoglobin")
    wbc = values.get("WBC")
    plt_val = values.get("Platelets")
    fbs = values.get("Fasting_Blood_Sugar")
    creatinine = values.get("Creatinine")
    tsh = values.get("TSH")
    chol = values.get("Cholesterol")

    ion_ca = values.get("Ionized_Calcium")
    tot_ca = values.get("Calcium")
    ggt = values.get("GGT")
    alb = values.get("Albumin")
    glob = values.get("Globulin")
    ag_ratio = values.get("AG_Ratio")
    tp = values.get("Total_Protein")
    alt = values.get("ALT")
    ast = values.get("AST")
    alp = values.get("ALP")
    bili = values.get("Bilirubin")
    egfr_val = values.get("eGFR")

    # Hemoglobin
    if hb is not None:
        hb_low = 12.0 if is_female else 13.0
        hb_high = 15.5 if is_female else 17.5
        if hb < 7.0:
            primary_findings.append(f"Severe Anaemia (Hb {hb} g/dL) — urgent clinical intervention required")
        elif hb < hb_low:
            primary_findings.append(f"Low Haemoglobin ({hb} g/dL) — consistent with Anaemia")
        elif hb > hb_high:
            primary_findings.append(f"Elevated Haemoglobin ({hb} g/dL) — consider polycythaemia or dehydration")
        else:
            bl = _is_borderline(hb, hb_low, hb_high)
            if bl == "borderline_low":
                borderline.append(f"Haemoglobin ({hb} g/dL) at lower edge of normal — monitor trend")

    # WBC
    if wbc is not None:
        if wbc > 15.0:
            primary_findings.append(f"Marked Leukocytosis ({wbc} ×10³/µL) — active infection or systemic inflammation")
        elif wbc > 11.0:
            primary_findings.append(f"Elevated WBC ({wbc} ×10³/µL) — possible infection or inflammation")
        elif wbc < 3.5:
            primary_findings.append(f"Low WBC ({wbc} ×10³/µL) — Leukopenia, evaluate for immunosuppression")
        elif wbc < 4.5:
            borderline.append(f"WBC ({wbc} ×10³/µL) at lower edge of normal")

    # Platelets
    if plt_val is not None:
        if plt_val < 50:
            primary_findings.append(f"Critical Thrombocytopenia (Platelets {plt_val} ×10³/µL) — high bleeding risk")
        elif plt_val < 150:
            primary_findings.append(f"Low Platelets ({plt_val} ×10³/µL) — Thrombocytopenia")
        elif plt_val > 450:
            primary_findings.append(f"Elevated Platelets ({plt_val} ×10³/µL) — Thrombocytosis")
        else:
            bl = _is_borderline(plt_val, 150, 450)
            if bl == "borderline_low":
                borderline.append(f"Platelets ({plt_val} ×10³/µL) near lower limit of normal")

    # Fasting Blood Sugar
    if fbs is not None:
        if fbs >= 126:
            primary_findings.append(f"Fasting Glucose {fbs} mg/dL — meets diagnostic threshold for Diabetes Mellitus")
        elif fbs >= 100:
            primary_findings.append(f"Impaired Fasting Glucose ({fbs} mg/dL) — Prediabetes range")
        elif fbs < 60:
            primary_findings.append(f"Low Glucose ({fbs} mg/dL) — Hypoglycaemia risk")
        else:
            bl = _is_borderline(fbs, 70, 100)
            if bl == "borderline_high":
                borderline.append(f"Fasting glucose ({fbs} mg/dL) approaching prediabetes threshold")

    # Creatinine & Renal Status (CKD Logic)
    creat_limit = 1.1 if is_female else 1.3
    if creatinine is not None:
        if creatinine > 2.0 or (egfr_val is not None and egfr_val < 45):
            primary_findings.append(f"Creatinine elevated ({creatinine} mg/dL) — impaired renal function")
        elif creatinine > creat_limit or (egfr_val is not None and egfr_val < 60):
            borderline.append(f"Mildly elevated Creatinine ({creatinine} mg/dL) — monitor renal trajectory")
        elif creatinine < 0.5:
            borderline.append(f"Creatinine low ({creatinine} mg/dL) — likely reflects lower muscle mass")
        else:
            cautions.append("No evidence of chronic kidney disease based on current renal markers.")

    # TSH
    if tsh is not None:
        if tsh > 10.0:
            primary_findings.append(f"TSH markedly elevated ({tsh} mIU/L) — overt Hypothyroidism")
        elif tsh > 4.5:
            primary_findings.append(f"TSH elevated ({tsh} mIU/L) — subclinical or overt Hypothyroidism")
        elif tsh < 0.1:
            primary_findings.append(f"TSH suppressed ({tsh} mIU/L) — Hyperthyroidism")
        elif tsh < 0.4:
            primary_findings.append(f"TSH low ({tsh} mIU/L) — subclinical Hyperthyroidism")
        else:
            bl = _is_borderline(tsh, 0.4, 4.5)
            if bl == "borderline_high":
                borderline.append(f"TSH ({tsh} mIU/L) near upper limit — monitor thyroid function")

    # Cholesterol
    if chol is not None:
        if chol >= 240:
            primary_findings.append(f"Total Cholesterol significantly elevated ({chol} mg/dL) — Hypercholesterolaemia")
        elif chol >= 200:
            primary_findings.append(f"Total Cholesterol borderline high ({chol} mg/dL)")

    # Calcium (Ionized Ca++ & Total Ca Tiers)
    if ion_ca is not None:
        if ion_ca >= 4.6:
            pass
        elif ion_ca >= 4.3:
            primary_findings.append(f"Borderline hypocalcemia (Ionized Ca {ion_ca} mg/dL)—monitor and correlate clinically.")
        elif ion_ca >= 4.0:
            primary_findings.append(f"Hypocalcemia (Ionized Ca {ion_ca} mg/dL)—requires clinical correlation.")
        else:
            primary_findings.append(f"Significant hypocalcemia (Ionized Ca {ion_ca} mg/dL)—urgent clinical review.")
    elif tot_ca is not None:
        if tot_ca >= 8.5 and tot_ca <= 10.2:
            pass
        elif tot_ca >= 8.0:
            primary_findings.append(f"Borderline hypocalcemia (Serum Ca {tot_ca} mg/dL)—monitor and correlate clinically.")
        elif tot_ca >= 7.0:
            primary_findings.append(f"Hypocalcemia (Serum Ca {tot_ca} mg/dL)—requires clinical correlation.")
        elif tot_ca < 7.0:
            primary_findings.append(f"Significant hypocalcemia (Serum Ca {tot_ca} mg/dL)—urgent clinical review.")
        elif tot_ca > 10.2:
            primary_findings.append(f"Elevated Serum Calcium ({tot_ca} mg/dL) — Hypercalcaemia")

    # Gamma GT (GGT Tiers)
    if ggt is not None:
        if ggt <= 50.0:
            pass
        elif ggt <= 75.0:
            primary_findings.append(f"Mild GGT elevation ({ggt} U/L)—possible hepatic enzyme stress; correlate with other liver markers and history.")
        elif ggt <= 150.0:
            primary_findings.append(f"Moderate GGT elevation ({ggt} U/L)—hepatic enzyme elevation; correlate clinically.")
        else:
            primary_findings.append(f"Marked GGT elevation ({ggt} U/L)—significant hepatic enzyme elevation; clinical review advised.")

    # Liver or Bile Duct Disease Multi-Marker Rule
    abnormal_liver_count = 0
    if alt is not None and alt > 55.0: abnormal_liver_count += 1
    if ast is not None and ast > 45.0: abnormal_liver_count += 1
    if bili is not None and bili > 1.2: abnormal_liver_count += 1
    if alp is not None and alp > 140.0: abnormal_liver_count += 1
    if ggt is not None and ggt > 150.0: abnormal_liver_count += 1

    if abnormal_liver_count >= 2:
        primary_findings.append("Hepatic Enzyme Pattern — multiple abnormal liver function markers present; clinical evaluation indicated.")
    elif abnormal_liver_count == 1 and (alt or ast or bili or alp):
        borderline.append("Mild liver enzyme variation—non-specific; correlate clinically.")

    # A/G Ratio & Protein Deficiency Logic
    if alb is not None and tp is not None:
        if alb < 3.5 and tp < 6.0:
            primary_findings.append(f"Protein Deficiency (Albumin {alb} g/dL, Total Protein {tp} g/dL)—evaluate nutritional intake and renal/hepatic losses.")
        elif ag_ratio is not None and ag_ratio < 1.2 and alb >= 3.5:
            primary_findings.append(f"Low A/G ratio ({ag_ratio})—may reflect relative globulin increase; no clear evidence of protein deficiency.")
    elif ag_ratio is not None and ag_ratio < 1.2:
        primary_findings.append(f"Low A/G ratio ({ag_ratio})—pattern is non-specific and not diagnostic of protein deficiency alone.")

    # Build safe primary diagnosis and findings
    hard_disease_labels = []
    for f in primary_findings:
        f_low = f.lower()
        if "severe anaemia" in f_low or "anaemia" in f_low:
            hard_disease_labels.append("Anaemia")
        elif "diabetes mellitus" in f_low:
            hard_disease_labels.append("Diabetes")
        elif "overt hypothyroidism" in f_low or "hypothyroidism" in f_low:
            hard_disease_labels.append("Hypothyroidism")
        elif "hyperthyroidism" in f_low:
            hard_disease_labels.append("Hyperthyroidism")
        elif "hypercholesterolaemia" in f_low:
            hard_disease_labels.append("Hypercholesterolaemia")
        elif "critical thrombocytopenia" in f_low or "thrombocytopenia" in f_low:
            hard_disease_labels.append("Thrombocytopenia")
        elif "impaired renal function" in f_low:
            hard_disease_labels.append("Impaired Renal Function")
        elif "protein deficiency (" in f_low:
            hard_disease_labels.append("Protein Deficiency")
        elif "hepatic enzyme pattern" in f_low:
            hard_disease_labels.append("Hepatic Enzyme Elevation")

    hard_disease_labels = list(dict.fromkeys(hard_disease_labels))

    if hard_disease_labels:
        diagnosis = ", ".join(hard_disease_labels)
    elif primary_findings:
        short_clauses = []
        for pf in primary_findings:
            first_clause = pf.split("—")[0].split("(")[0].strip()
            if first_clause and first_clause not in short_clauses:
                short_clauses.append(first_clause)
        diagnosis = ", ".join(short_clauses) if short_clauses else "Minor Variations Noted"
    elif borderline:
        diagnosis = "No significant abnormalities — minor borderline variations noted"
    else:
        diagnosis = "No significant abnormalities detected"

    summary_parts = []
    if primary_findings:
        summary_parts.append("PRIMARY FINDINGS: " + "; ".join(primary_findings))
    if borderline:
        summary_parts.append("BORDERLINE / VARIATIONS: " + "; ".join(borderline))
    if cautions:
        summary_parts.append("CLINICAL NOTES: " + "; ".join(cautions))
    if not primary_findings and not borderline:
        summary_parts.append("All assessed biomarkers are within established clinical reference ranges. No pathological features detected.")
    summary_parts.append("Findings require correlation with patient clinical history and physical examination.")

    return {
        "diagnosis": diagnosis,
        "findings": primary_findings,
        "borderline_flags": borderline,
        "cautions": cautions,
        "rules_triggered": primary_findings,
        "summary": " ".join(summary_parts)
    }

def run_ml_classifier(values):
    model, le, feature_cols = load_local_classifier()
    if model is None:
        return {"diagnosis": "ML model files missing. Retrain models.", "probabilities": {}}
        
    gender_enc = 1 if values["Gender"] == "Male" else 0
    vector = [
        values["Age"], gender_enc, values["Hemoglobin"], values["WBC"], values["Platelets"],
        values["Fasting_Blood_Sugar"], values["Creatinine"], values["TSH"], values["Cholesterol"]
    ]
    vector_np = np.array([vector])
    probs = model.predict_proba(vector_np)[0]
    preds = {}
    for idx, class_name in enumerate(le.classes_):
        preds[class_name] = round(float(probs[idx]), 4)
        
    best_class = max(preds, key=preds.get)
    best_prob = preds[best_class]
    
    if best_prob < 0.5:
        diagnosis = f"Inconclusive — {best_class} most likely ({best_prob*100:.1f}%) but below confidence threshold"
    else:
        diagnosis = f"{best_class} ({best_prob*100:.1f}% Confidence)"
    
    return {
        "diagnosis": diagnosis,
        "probabilities": preds,
        "summary": f"Classifier predicted {best_class} with {best_prob*100:.1f}% probability based on diagnostic profile."
    }

def normalize_image_bytes(image_bytes):
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        max_dim = 1600
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, "image/png"

def parse_vlm_json_response(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r'(\{[\s\S]*\})', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return {
        "Patient_Name": None, "Age": None, "Gender": None, "Hemoglobin": None, "WBC": None, "Platelets": None,
        "Fasting_Blood_Sugar": None, "Creatinine": None, "TSH": None, "Cholesterol": None,
        "Ionized_Calcium": None, "Calcium": None, "GGT": None, "Albumin": None,
        "Globulin": None, "AG_Ratio": None, "Total_Protein": None,
        "Clinical_Interpretation": raw_text[:600] if raw_text else "Analysis completed.",
        "Predicted_Diseases": []
    }

def run_nvidia_nim_vlm(image_bytes, api_key):
    start_time = time.time()
    norm_bytes, mime_type = normalize_image_bytes(image_bytes)
    base64_image = base64.b64encode(norm_bytes).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "You are an expert clinical pathologist analyzing a medical laboratory report.\n"
        "Extract patient demographics (Patient_Name, Age, Gender) and all numeric biomarker values accurately.\n"
        "Multilingual Patient Demographics Extraction Guidelines:\n"
        "- Extract Patient_Name accurately in its original script (English, Sinhala: සිංහල, Tamil: தமிழ், etc.) or Romanized script.\n"
        "- Look for patient name headers in English ('Patient Name', 'Pt Name'), Sinhala ('රෝගියාගේ නම', 'රෝගී නම', 'නම'), or Tamil ('நோயாளி பெயர்', 'பெயர்').\n"
        "- Do NOT confuse Doctor's name (Dr., Consultant, Ref By) or Hospital/Lab name with the Patient's Name.\n"
        "Clinical Calibration Rules:\n"
        "- Ionized Calcium: 4.6–5.3 mg/dL is normal. 4.3–4.5 mg/dL is mild/borderline hypocalcemia (do NOT label as severe disease).\n"
        "- Gamma GT (GGT): Mild elevation (up to 75 U/L) is non-specific hepatic stress, NOT liver disease. Only diagnose liver disease if multiple liver markers (ALT, AST, Bilirubin, ALP) are abnormal.\n"
        "- A/G Ratio: Low A/G with normal Albumin is non-specific; do NOT label as Protein Deficiency unless Albumin < 3.5 g/dL and Total Protein < 6.0 g/dL.\n"
        "- Chronic Kidney Disease: Do NOT diagnose CKD if Creatinine is normal.\n"
        "- If values are healthy or show only mild isolated borderline variations, Predicted_Diseases MUST be empty [].\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        "  \"Patient_Name\": \"string or null\",\n"
        "  \"Age\": int or null,\n"
        "  \"Gender\": \"Male\" | \"Female\" | null,\n"
        "  \"Hemoglobin\": float or null,\n"
        "  \"WBC\": float or null,\n"
        "  \"Platelets\": int or null,\n"
        "  \"Fasting_Blood_Sugar\": int or null,\n"
        "  \"Creatinine\": float or null,\n"
        "  \"TSH\": float or null,\n"
        "  \"Cholesterol\": int or null,\n"
        "  \"Ionized_Calcium\": float or null,\n"
        "  \"Calcium\": float or null,\n"
        "  \"GGT\": float or null,\n"
        "  \"Albumin\": float or null,\n"
        "  \"AG_Ratio\": float or null,\n"
        "  \"Total_Protein\": float or null,\n"
        "  \"Clinical_Interpretation\": \"concise calibrated clinical summary\",\n"
        "  \"Predicted_Diseases\": []\n"
        "}"
    )
    
    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        latency = time.time() - start_time
        if response.status_code != 200:
            return {"error": f"Nvidia NIM API returned status {response.status_code}: {response.text[:200]}", "latency": latency}
        res_data = response.json()
        raw_text = res_data['choices'][0]['message']['content'].strip()
        structured_data = parse_vlm_json_response(raw_text)
        structured_data["latency"] = latency
        return structured_data
    except Exception as e:
        return {"error": f"Nvidia NIM error: {str(e)}", "latency": time.time() - start_time}

def run_gemini_vlm(image_bytes, api_key):
    start_time = time.time()
    norm_bytes, mime_type = normalize_image_bytes(image_bytes)
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(data=norm_bytes, mime_type=mime_type)
        
        prompt = (
            "Analyze the medical laboratory report image.\n"
            "Extract patient demographics (Patient_Name, Age, Gender) and all visible biomarker values.\n"
            "Multilingual Patient Demographics Extraction Guidelines:\n"
            "- Extract Patient_Name accurately in its original script (English, Sinhala: සිංහල, Tamil: தமிழ், etc.) or Romanized script.\n"
            "- Look for patient name headers in English ('Patient Name', 'Pt Name'), Sinhala ('රෝගියාගේ නම', 'රෝගී නම', 'නම'), or Tamil ('நோயாளி பெயர்', 'பெயர்').\n"
            "- Do NOT confuse Doctor's name (Dr., Consultant, Ref By) or Hospital/Lab name with the Patient's Name.\n"
            "Clinical Calibration Guidelines:\n"
            "- Ionized Calcium: 4.6–5.3 mg/dL is normal. 4.3–4.5 mg/dL is borderline hypocalcemia (not severe).\n"
            "- Gamma GT (GGT): Mild elevation (< 75 U/L) is non-specific hepatic stress, NOT liver disease. Only flag liver disease if >= 2 liver markers (ALT, AST, Bilirubin, ALP) are abnormal.\n"
            "- A/G Ratio: Low A/G with normal Albumin is non-specific; do NOT diagnose Protein Deficiency unless Albumin < 3.5 g/dL and Total Protein < 6.0 g/dL.\n"
            "- Chronic Kidney Disease: Do NOT diagnose CKD if Creatinine is within normal range.\n"
            "- Only list confirmed pathological diseases in Predicted_Diseases. If normal or only mild isolated variation, return Predicted_Diseases as [].\n\n"
            "Return ONLY a clean JSON object matching:\n"
            "{\n"
            "  \"Patient_Name\": \"string or null\",\n"
            "  \"Age\": integer or null,\n"
            "  \"Gender\": \"Male\" | \"Female\" | null,\n"
            "  \"Hemoglobin\": float or null,\n"
            "  \"WBC\": float or null,\n"
            "  \"Platelets\": int or null,\n"
            "  \"Fasting_Blood_Sugar\": int or null,\n"
            "  \"Creatinine\": float or null,\n"
            "  \"TSH\": float or null,\n"
            "  \"Cholesterol\": int or null,\n"
            "  \"Ionized_Calcium\": float or null,\n"
            "  \"Calcium\": float or null,\n"
            "  \"GGT\": float or null,\n"
            "  \"Albumin\": float or null,\n"
            "  \"AG_Ratio\": float or null,\n"
            "  \"Total_Protein\": float or null,\n"
            "  \"Clinical_Interpretation\": \"clinical summary\",\n"
            "  \"Predicted_Diseases\": []\n"
            "}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        
        latency = time.time() - start_time
        raw_text = response.text.strip()
        structured_data = parse_vlm_json_response(raw_text)
        structured_data["latency"] = latency
        return structured_data
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}", "latency": time.time() - start_time}
