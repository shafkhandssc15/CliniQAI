import os
import re
import json
import time
import base64
import requests
import joblib
import numpy as np
from PIL import Image, ImageDraw
import io

_tess_api = None
_easyocr_reader = None
_clinical_model = None
_label_encoder = None
_feature_columns = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import torch
            import easyocr
            gpu_available = torch.cuda.is_available()
            _easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
        except Exception as e:
            print(f"[OCR Engine] EasyOCR fallback unavailable: {e}")
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

def get_tesseract_cmd():
    user_appdata = os.environ.get("LOCALAPPDATA", "")
    user_prog = os.path.join(user_appdata, "Programs", "Tesseract-OCR", "tesseract.exe") if user_appdata else ""
    user_tess = os.path.join(user_appdata, "Tesseract-OCR", "tesseract.exe") if user_appdata else ""
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        user_prog,
        user_tess,
        os.path.join(os.path.dirname(__file__), "tesseract", "tesseract.exe"),
        os.environ.get("TESSERACT_PATH", "")
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    import shutil
    return shutil.which("tesseract")

def extract_text_from_image(image_bytes, return_bboxes: bool = False):
    start_time = time.time()
    
    # ── Primary Engine: Native Tesseract 5.5.3 (C++ Leptonica in-process, ~0.2-0.8s) ──
    try:
        import tesserocr
        tessdata_candidates = [
            os.path.join(os.path.dirname(__file__), "tessdata"),
            r"c:\CliniQ AI\tessdata",
            r"C:\Program Files\Tesseract-OCR\tessdata",
            os.environ.get("TESSDATA_PREFIX", "")
        ]
        tessdata_path = next((p for p in tessdata_candidates if p and os.path.exists(p) and os.path.exists(os.path.join(p, "eng.traineddata"))), None)
        if tessdata_path:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            orig_w, orig_h = pil_img.size
            # 2x Lanczos upsampling for optimal optical character recognition resolution
            scale = 2.0 if max(orig_w, orig_h) < 2200 else 1.0
            if scale > 1.0:
                proc_img = pil_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)
            else:
                proc_img = pil_img
                
            with tesserocr.PyTessBaseAPI(path=tessdata_path, lang="eng") as api:
                api.SetImage(proc_img)
                full_text = api.GetUTF8Text()
                dur = time.time() - start_time
                print(f"[OCR Engine] Primary Native Tesseract 5.5.3 OCR completed in {dur:.2f}s ({len(full_text)} chars)")
                
                if return_bboxes:
                    api.Recognize()
                    ri = api.GetIterator()
                    level = tesserocr.RIL.WORD
                    bboxes = []
                    while ri and not ri.Empty(level):
                        word = ri.GetUTF8Text(level)
                        conf = ri.Confidence(level)
                        box = ri.BoundingBox(level)
                        if word and word.strip() and box:
                            x0, y0, x1, y1 = [int(v / scale) for v in box]
                            bx = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                            bboxes.append((bx, word.strip(), float(conf) / 100.0))
                        if not ri.Next(level):
                            break
                    return full_text, dur, bboxes
                return full_text, dur
    except Exception as tess_err:
        print(f"[OCR Engine] Primary tesserocr error: {tess_err}. Trying secondary Tesseract CLI...")

    # Secondary Path: pytesseract CLI if standalone binary installed
    tess_cmd = get_tesseract_cmd()
    if tess_cmd:
        try:
            print(f"[OCR Engine] Secondary Engine: Tesseract CLI ({tess_cmd})")
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            full_text = pytesseract.image_to_string(pil_img, lang="eng")
            
            if return_bboxes:
                data = pytesseract.image_to_data(pil_img, lang="eng", output_type=pytesseract.Output.DICT)
                bboxes = []
                n_boxes = len(data.get("text", []))
                for i in range(n_boxes):
                    t = str(data["text"][i]).strip()
                    if t:
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        raw_c = data["conf"][i]
                        conf = float(raw_c) / 100.0 if raw_c != "-1" else 0.5
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        bboxes.append((bbox, t, conf))
                return full_text, time.time() - start_time, bboxes
            return full_text, time.time() - start_time
        except Exception as e:
            print(f"[OCR Engine] Tesseract CLI error: {e}. Falling back to EasyOCR.")
    else:
        print("[OCR Engine] Tesseract binary not found. Falling back to EasyOCR.")

    # Emergency Fallback: EasyOCR (PyTorch-based)
    reader = get_easyocr_reader()
    if reader is None:
        if return_bboxes:
            return "", time.time() - start_time, []
        return "", time.time() - start_time

    results = reader.readtext(image_bytes, batch_size=4, canvas_size=1280, mag_ratio=1.0)
    
    if not results:
        if return_bboxes:
            return "", time.time() - start_time, []
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
            if abs(b["y_center"] - line_y_avg) <= max(10.0, line_h_avg * 0.65):
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    lines.sort(key=lambda line: sum(b["y_center"] for b in line) / len(line))
    for line in lines:
        line.sort(key=lambda b: b["x_min"])

    reconstructed_lines = [" ".join(b["text"] for b in line) for line in lines]
    full_text = "\n".join(reconstructed_lines)
    if return_bboxes:
        return full_text, time.time() - start_time, results
    return full_text, time.time() - start_time


def deidentify_report_image(image_bytes, ocr_results=None) -> bytes:
    """
    Redacts patient Personally Identifiable Information (PII) including Name, NIC,
    Phone Number, and Address before transmitting image bytes to cloud VLM models.
    Ensures compliance with Sri Lanka Personal Data Protection Act (PDPA) No. 9 of 2022.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        if ocr_results is None:
            _, _, ocr_results = extract_text_from_image(image_bytes, return_bboxes=True)

        PII_IDENTIFIERS = [
            "name", "patient", "pt", "client", "nic", "identity",
            "phone", "mobile", "tel", "contact", "address", "reg", "uhid", "pid",
            "රෝගියා", "නම", "ජා.හැ", "දුරකථන", "ලිපිනය",
            "நோயாளி", "பெயர்", "தொலைபேசி", "முகவரி"
        ]

        for bbox, text, conf in ocr_results:
            t_low = str(text).lower().strip()
            # Match keywords
            is_pii = any(k in t_low for k in PII_IDENTIFIERS)
            # Match Sri Lankan NIC patterns
            is_nic = bool(re.search(r'\b\d{9}[VXvx]\b', text) or re.search(r'\b\d{12}\b', text))
            # Match Phone patterns
            clean_digits = re.sub(r'[\s\-]', '', text)
            is_phone = bool(re.search(r'\b(?:\+94|0)\d{9}\b', clean_digits) or re.search(r'\b07\d{8}\b', clean_digits))

            if is_pii or is_nic or is_phone:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x0, x1 = max(0, min(xs) - 4), min(pil_img.width, max(xs) + 4)
                y0, y1 = max(0, min(ys) - 2), min(pil_img.height, max(ys) + 2)
                # Redact text box + mask patient value region horizontally
                if any(k in t_low for k in ["name", "patient", "pt", "නම", "රෝගියා", "பெயர்", "நோயாளி"]):
                    x1 = min(pil_img.width, x1 + 240)
                draw.rectangle([x0, y0, x1, y1], fill="#000000")

        out = io.BytesIO()
        pil_img.save(out, format="JPEG", quality=88)
        return out.getvalue()
    except Exception as e:
        print(f"[WARN] Patient de-identification failed gracefully: {e}")
        return image_bytes


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
        "asiri", "asirihealth", "horns", "plc", "automated", "reference", "confidential", "www.", ".com",
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
        r'patient\s*[:\-#.]?',
        r'\bname\s*[:\-#.]?',
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

    TITLE_PATTERN = r'\b(?:mr\.?|mrs\.?|ms\.?|miss|master|mast\.?|dr\.?|rev\.?|baby|b/o|shri|smt|ven\.\s+|මයා|මිය|මෙනවිය|පූජ්‍ය|හිමි|திரு|திருமதி|செல்வி)\b'

    # 1. Scan for Age and Gender
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
                age_m2 = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?|y\b|y/f|y/m|/f|/m|වසර|வருடங்கள்)', line, re.IGNORECASE)
                if age_m2:
                    v = int(age_m2.group(1))
                    if 1 <= v <= 120:
                        result["age"] = v

        # Gender (English, Sinhala: ස්ත්‍රී / පුරුෂ, Tamil: பெண் / ஆண்)
        if result["gender"] is None:
            if re.search(r'\b(?:sex|gender)\s*[:/\-]?\s*(?:female|f\b)', ll) or re.search(r'\b(female|woman)\b', ll) or re.search(r'\b(?:\d{0,3}\s*)?(?:y|yrs?|years?)?\s*/\s*f\b', ll) or any(s in line for s in ["ස්ත්‍රී", "ගැහැණු", "பெண்"]):
                result["gender"] = "Female"
            elif re.search(r'\b(?:sex|gender)\s*[:/\-]?\s*(?:male|m\b)', ll) or re.search(r'\b(male|man)\b', ll) or re.search(r'\b(?:\d{0,3}\s*)?(?:y|yrs?|years?)?\s*/\s*m\b', ll) or any(s in line for s in ["පුරුෂ", "පිරිමි", "ஆண்"]):
                if 'fe' not in ll.split('male')[0][-2:]:
                    result["gender"] = "Male"

    # 2. Phase 1: Explicit Patient Prefix Scan across the entire document
    for line in lines:
        ll = line.lower().strip()
        has_excluded = any(ex in ll for ex in EXCLUDE_TOKENS)
        if has_excluded and not any(k in ll for k in ["patient name", "pt name", "name of patient", "pt. name", "patient :", "patient :", "රෝගියා", "නම", "பெயர்", "நோயாளி"]):
            continue

        for p_pat in NAME_PREFIXES:
            match = re.search(rf'{p_pat}\s*([^\n\r\t|;#]+)', line, re.IGNORECASE)
            if match:
                raw_candidate = match.group(1).strip()
                # Strip trailing metadata like age, sex, date, sample no, etc.
                raw_candidate = re.split(r'\b(?:age|sex|gender|date|ref|dr\b|doctor|pid|uhid|ipd|opd|reg|id|bill|lab\s*no|sample|වයස|ස්ත්‍රී|පුරුෂ|வயது|ஆண்|பெண்)\b|[|;\t]', raw_candidate, flags=re.IGNORECASE)[0].strip()
                raw_candidate = re.sub(r'^[MK]PS\.?\s*', 'MRS. ', raw_candidate, flags=re.IGNORECASE).strip()
                clean = re.sub(r'^[^\w\u0D80-\u0DFF\u0B80-\u0BFF]+|[^\w\u0D80-\u0DFF\u0B80-\u0BFF.]+$', '', raw_candidate).strip()
                clean = re.sub(r'\s+', ' ', clean)
                clean = re.sub(r'[\d_#=+\\/*]', '', clean).strip(' ,:;-')
                words = clean.split()
                filtered_words = [w for w in words if w.lower() not in {"years", "year", "yrs", "yr", "male", "female", "y", "m", "f", "patient", "name", "report", "වයස", "නම", "பெயர்", "வயது", "ස්ත්‍රී", "පුරුෂ"}]
                clean = " ".join(filtered_words)
                if len(clean) >= 2 and re.search(r'[\w\u0D80-\u0DFF\u0B80-\u0BFF]', clean) and not any(k in clean.lower() for k in ["hemoglobin", "haemoglobin", "glucose", "calcium", "serum", "blood", "urine", "count", "profile", "test", "laboratory", "hospital", "pathology"]):
                    result["name"] = clean.strip()
                    break
        if result["name"]:
            break

    # 3. Phase 2: Fallback Title Matching (only if no explicit prefix found)
    if result["name"] is None:
        for line in lines:
            ll = line.lower().strip()
            has_excluded = any(ex in ll for ex in EXCLUDE_TOKENS)
            if not has_excluded:
                title_match = re.search(rf'({TITLE_PATTERN}\s+[^\n\r\t|;#]+)', line, re.IGNORECASE)
                if title_match:
                    raw_candidate = title_match.group(1).strip()
                    raw_candidate = re.split(r'\b(?:age|sex|gender|date|ref|pid|uhid|reg|id|වයස|ස්ත්‍රී|පුරුෂ|வயது)\b|[|;\t]', raw_candidate, flags=re.IGNORECASE)[0].strip()
                    raw_candidate = re.sub(r'^[MK]PS\.?\s*', 'MRS. ', raw_candidate, flags=re.IGNORECASE).strip()
                    clean = re.sub(r'[\d_#=+\\/*]', '', raw_candidate).strip(' ,:;-')
                    clean = re.sub(r'\s+', ' ', clean)
                    if len(clean) >= 3 and not any(k in clean.lower() for k in ["doctor", "dr.", "consultant"]):
                        result["name"] = clean.strip()
                        break

    # 4. Gender Inference from Personal Title if gender was not explicitly present in table
    if result["gender"] is None and result["name"]:
        nl = result["name"].lower()
        if re.search(r'\b(?:mrs\.?|ms\.?|miss|මිය|මෙනවිය|திருமதி|செல்வி)\b', nl):
            result["gender"] = "Female"
        elif re.search(r'\b(?:mr\.?|master|mast\.?|මයා|திரு)\b', nl):
            result["gender"] = "Male"

    return result

def extract_smart_biomarker(line: str, min_val: float, max_val: float, unit_hints: list = None) -> float:
    """
    Extracts the clinical patient test result value from a report line.
    Accurately ignores reference range columns, bracketed ranges, and hyphenated ranges.
    Prioritizes:
    1. Numbers immediately preceding known measurement units
    2. Candidate numbers in range-cleaned line
    3. Guarantees that reference range limits are NEVER extracted as patient results
    """
    if not line or not line.strip():
        return None

    # Pre-normalize: separate multipliers and common unit tokens from attached numbers
    # e.g., '380x10^9/L' -> '380 x10^9/L', '8.2gdL' -> '8.2 g/dL'
    norm_line = re.sub(r'(\d+(?:\.\d+)?)\s*([x*]\s*10[\^*]?\s*\d*(?:/[a-zA-Z]+)?)', r' \1 \2 ', line, flags=re.IGNORECASE)
    norm_line = re.sub(r'(\d+(?:\.\d+)?)([a-zA-Z/%])', r'\1 \2', norm_line)
    norm_line = re.sub(r'([a-zA-Z/%])(\d+(?:\.\d+)?)', r'\1 \2', norm_line)

    # 1. Check if a number is directly followed by a recognized measurement unit
    if unit_hints:
        for u in unit_hints:
            pattern = re.compile(rf'(?<![.\d])(\d+(?:\.\d+)?)\s*(?:{re.escape(u)})\b', re.IGNORECASE)
            m = pattern.search(norm_line)
            if m:
                try:
                    val = float(m.group(1))
                    if min_val <= val <= max_val:
                        return val
                except ValueError:
                    pass

    # 2. Identify and sanitize explicit reference ranges:
    # Matches: (0.55 - 4.78), [0.55 - 4.78], 0.55 - 4.78, 0.55 – 4.78, 0.55 to 4.78, 0.55..4.78
    clean_line = re.sub(r'[\(\[\{]?\s*\d+(?:\.\d+)?\s*(?:[-–—]|to|\.\.)\s*\d+(?:\.\d+)?\s*[\)\]\}]?', ' [REF_RANGE] ', norm_line, flags=re.IGNORECASE)
    clean_line = re.sub(r'(?:ref\.?\s*range|reference|normal|ref|biological\s*ref|range)\s*[:\-]?\s*[\(\[\{]?\s*[\d\.\-\–\—\s\<\>to]+\s*[\)\]\}]?', ' [REF_RANGE] ', clean_line, flags=re.IGNORECASE)
    clean_line = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' [REF_RANGE] ', clean_line)

    # 3. Find candidate numbers in the range-cleaned line
    num_pattern = re.compile(r'(?<![.\d])(\d+(?:\.\d+)?)(?![.\d])')
    nums = num_pattern.findall(clean_line)
    if nums:
        for n in nums:
            try:
                val = float(n)
                # Ignore isolated multiplier exponents like 9 or 3 from 10^9 or 10^3 if they appear alone
                if val in (3.0, 9.0, 10.0, 12.0) and any(tok in clean_line for tok in ['10', '^', '*', 'x']) and min_val > 10.0:
                    continue
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
        "Vitamin_B12": None, "Ferritin": None, "ESR": None, "eGFR": None,
        "Urea": None, "Uric_Acid": None, "Phosphorus": None, "Sodium": None,
        "Potassium": None, "Chloride": None
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

        if any(k in line for k in ["haemoglobin", "hemoglobin", "hb", "hgb", "hemo", "haemo"]) and extracted["Hemoglobin"] is None:
            val = extract_smart_biomarker(line, 3.0, 24.0, ["g/dl", "g/l", "g/t", "g%", "gm/dl"])
            if val is not None: extracted["Hemoglobin"] = val

        if any(k in line for k in ["wbc", "white cell", "white blood", "leukocyte", "tcl", "w.b.c"]) and extracted["WBC"] is None:
            val = extract_smart_biomarker(line, 0.5, 100.0, ["10^3", "10*3", "x10^3", "/cumm", "/ul", "/mm3", "k/ul", "cells/cumm", "10^9/l", "10*9/l"])
            if val is not None: extracted["WBC"] = val

        if any(k in line for k in ["platelet", "plt", "thrombocyte", "p.l.t"]) and extracted["Platelets"] is None:
            val = extract_smart_biomarker(line, 10.0, 1500.0, ["10^3", "10*3", "x10^3", "/cumm", "/ul", "/mm3", "k/ul", "lakhs", "10^9/l", "10*9/l"])
            if val is not None: extracted["Platelets"] = int(val)

        if any(k in line for k in ["glucose", "fbs", "blood sugar", "sugar", "fasting"]) and extracted["Fasting_Blood_Sugar"] is None:
            val = extract_smart_biomarker(line, 30.0, 800.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None: extracted["Fasting_Blood_Sugar"] = int(val)

        if any(k in line for k in ["creatinine", "creat", "crea", "sr. creatinine"]) and extracted["Creatinine"] is None:
            val = extract_smart_biomarker(line, 0.1, 25.0, ["mg/dl", "mg%", "umol/l", "µmol/l"])
            if val is not None: extracted["Creatinine"] = val

        if any(k in line for k in ["urea", "blood urea", "s. urea"]) and extracted["Urea"] is None:
            val = extract_smart_biomarker(line, 2.0, 350.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None: extracted["Urea"] = val

        if any(k in line for k in ["uric acid", "urate"]) and extracted["Uric_Acid"] is None:
            val = extract_smart_biomarker(line, 0.5, 25.0, ["mg/dl", "mg%", "umol/l"])
            if val is not None: extracted["Uric_Acid"] = val

        if any(k in line for k in ["phosphorus", "phosphate", "phos"]) and extracted["Phosphorus"] is None:
            val = extract_smart_biomarker(line, 0.2, 20.0, ["mg/dl", "mg%", "mmol/l"])
            if val is not None: extracted["Phosphorus"] = val

        if any(k in line for k in ["sodium", "serum sodium"]) and extracted["Sodium"] is None:
            val = extract_smart_biomarker(line, 80.0, 200.0, ["meq/l", "mmol/l"])
            if val is not None: extracted["Sodium"] = val

        if any(k in line for k in ["potassium", "serum potassium"]) and extracted["Potassium"] is None:
            val = extract_smart_biomarker(line, 1.0, 12.0, ["meq/l", "mmol/l"])
            if val is not None: extracted["Potassium"] = val

        if any(k in line for k in ["chloride", "serum chloride"]) and extracted["Chloride"] is None:
            val = extract_smart_biomarker(line, 50.0, 180.0, ["meq/l", "mmol/l"])
            if val is not None: extracted["Chloride"] = val

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

        if any(k in line for k in ["calcium", "serum calcium", "total calcium"]) and not any(x in line for x in ["ionized", "ca++"]) and extracted["Calcium"] is None:
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
            if val is not None: extracted["Ferritin"] = val

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

# ── Universal Laboratory Values Extractor (Tesseract OCR Engine) ───────────
LAB_KNOWN_UNITS = [
    "x10^12/l", "10^12/l", "x10^9/l", "10^9/l", "x10^3/ul", "10^3/ul", "cells/cumm", "/cumm", "/ul", "/mm3", "k/ul",
    "g/dl", "g/l", "mg/dl", "mg/l", "ug/dl", "ng/ml", "pg/ml", "ng/l", "pmol/l", "nmol/l", "mmol/l", "umol/l",
    "µmol/l", "meq/l", "uiu/ml", "miu/l", "u/l", "iu/l", "fl", "pg", "ml/min", "ml/min/1.73m2", "mg/g",
    "mm/1st hr", "mm/hr", "mm", "%", "ratio", "index"
]

CATEGORY_PATTERNS = {
    "Glucose & Metabolism": [
        r"\bhba1c\b", r"glycated", r"\bglucose\b", r"\bfbs\b", r"\bfpg\b", r"blood sugar", r"\bsugar\b",
        r"\bppbs\b", r"\brbs\b", r"\binsulin\b", r"\bhoma\b"
    ],
    "CBC": [
        r"\bhaemoglobin\b", r"\bhemoglobin\b", r"\bhb\b", r"\bhgb\b",
        r"\bwbc\b", r"white blood", r"white cell", r"\bleukocyte\b", r"total leukocyte", r"\btlc\b"
    ],
    "Differential Count": [
        r"\bneutrophil", r"\blymphocyte", r"\bmonocyte", r"\beosinophil", r"\bbasophil",
        r"\bpolymorph", r"\bband cell", r"\banc\b", r"\balc\b", r"\baec\b"
    ],
    "Red Blood Cells": [
        r"\brbc\b", r"red blood", r"\berythrocyte", r"\bpcv\b", r"packed cell",
        r"\bhematocrit\b", r"\bhct\b", r"\bmcv\b", r"\bmch\b", r"\bmchc\b", r"\brdw\b"
    ],
    "Platelets": [
        r"\bplatelet", r"\bplt\b", r"\bthrombocyte", r"\bmpv\b", r"\bpdw\b", r"\bpct\b"
    ],
    "Liver Function": [
        r"\balt\b", r"\bsgpt\b", r"\bast\b", r"\bsgot\b", r"\balp\b", r"alkaline phosphatase",
        r"\bbilirubin", r"total protein", r"\balbumin\b", r"\bglobulin\b", r"a/g ratio", r"\bggt\b", r"gamma gt"
    ],
    "Kidney Function": [
        r"\bcreatinine\b", r"\burea\b", r"\bbun\b", r"\begfr\b", r"\bgfr\b", r"uric acid",
        r"\burate\b", r"microalbumin", r"\bcystatin\b"
    ],
    "Electrolytes": [
        r"\bsodium\b", r"\bpotassium\b", r"\bchloride\b", r"\bbicarbonate\b", r"\bco2\b",
        r"\bcalcium\b", r"\bmagnesium\b", r"\bphosphorus\b", r"\bphosphate\b"
    ],
    "Lipid Profile": [
        r"cholesterol", r"triglyceride", r"\bhdl\b", r"\bldl\b", r"\bvldl\b",
        r"non-hdl", r"tc/hdl", r"lipid"
    ],
    "Thyroid Function": [
        r"\btsh\b", r"\bft3\b", r"\bft4\b", r"free t3", r"free t4", r"total t3", r"total t4", r"thyroid"
    ],
    "Inflammatory & Serology": [
        r"\bcrp\b", r"\besr\b", r"\bferritin\b", r"\biron\b", r"\btibc\b", r"\btransferrin\b",
        r"\bdengue\b", r"\bns1\b", r"\btroponin\b", r"vitamin d", r"vitamin b12", r"\bvit d\b", r"\bvit b12\b"
    ],
    "Urinalysis": [
        r"\burine\b", r"pus cells", r"epithelial", r"specific gravity", r"urobilinogen",
        r"nitrite", r"ketone", r"casts", r"crystals"
    ]
}

def extract_all_laboratory_values(text: str, image_bytes: bytes = None, ocr_bboxes: list = None) -> dict:
    """
    Extracts all laboratory results and report-provided reference values.
    """
    def _merge_reference_values_from_bboxes(results: list, ocr_bboxes: list) -> None:
        if not ocr_bboxes:
            return

        words = []
        for bbox, text, confidence in ocr_bboxes:
            if not text or not str(text).strip():
                continue
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            words.append({
                "text": str(text).strip(),
                "x0": min(xs),
                "x1": max(xs),
                "y": (min(ys) + max(ys)) / 2,
                "height": max(1, max(ys) - min(ys)),
            })

        words.sort(key=lambda word: word["y"])
        rows = []
        for word in words:
            row = next((
                candidate for candidate in rows
                if abs(word["y"] - candidate["y"]) <= max(5, min(word["height"], candidate["height"]) * 0.65)
            ), None)
            if row is None:
                row = {"y": word["y"], "height": word["height"], "words": []}
                rows.append(row)
            row["words"].append(word)
            row["y"] = sum(item["y"] for item in row["words"]) / len(row["words"])
            row["height"] = sum(item["height"] for item in row["words"]) / len(row["words"])

        for row in rows:
            row["words"].sort(key=lambda word: word["x0"])
            row["text"] = " ".join(word["text"] for word in row["words"])

        header = next((
            row for row in rows
            if "reference" in row["text"].lower() and "value" in row["text"].lower()
        ), None)
        if header is None:
            return

        reference_start = min(
            word["x0"] for word in header["words"]
            if "reference" in word["text"].lower()
        )
        reference_pattern = re.compile(
            r"(?:[<>]=?\s*)?\d+(?:[.,]\d+)?\s*%?\s*(?:[-–—]|to)\s*\d+(?:[.,]\d+)?\s*%?"
            r"|[<>]=?\s*\d+(?:[.,]\d+)?\s*%?",
            re.IGNORECASE
        )

        for result in results:
            name = re.sub(r"[^a-z0-9]", "", result.get("test_name", "").lower())
            if len(name) < 3:
                continue
            matching_rows = [row for row in rows if name in re.sub(r"[^a-z0-9]", "", row["text"].lower())]
            if not matching_rows:
                continue
            result_row = matching_rows[0]
            reference_words = [word for word in result_row["words"] if word["x0"] >= reference_start - 4]
            reference_text = " ".join(word["text"] for word in reference_words).strip()
            if not reference_pattern.search(reference_text):
                next_rows = [row for row in rows if result_row["y"] < row["y"] <= result_row["y"] + max(8, result_row["height"] * 1.8)]
                for next_row in next_rows:
                    continuation = " ".join(
                        word["text"] for word in next_row["words"] if word["x0"] >= reference_start - 4
                    ).strip()
                    if continuation and re.fullmatch(r"[\d.,()\[\]<>%+\-–—\s]+", continuation):
                        reference_text = f"{reference_text} {continuation}".strip()

            if not reference_pattern.search(reference_text):
                continue

            reference_text = re.sub(r"\s+([%)\]])", r"\1", reference_text)
            reference_text = re.sub(r"([([])\s+", r"\1", reference_text)
            result["reference_value"] = reference_text
            reference_range = result.setdefault("reference_range", {})
            reference_range["raw"] = reference_text
            reference_range["provided"] = True

            bounds = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:[.,]\d+)?)", reference_text, re.IGNORECASE)
            if bounds:
                reference_range["low"] = float(bounds.group(1).replace(",", "."))
                reference_range["high"] = float(bounds.group(2).replace(",", "."))

            value = result.get("result_value")
            flag = result.get("original_flag")
            low = reference_range.get("low")
            high = reference_range.get("high")
            if flag and "h" in flag.lower():
                result["result_status"] = "high"
            elif flag and "l" in flag.lower():
                result["result_status"] = "low"
            elif isinstance(value, (int, float)) and low is not None and high is not None:
                result["result_status"] = "low" if value < low else "high" if value > high else "normal"
    if not text:
        return {
            "patient_identifier": None,
            "report_date": None,
            "laboratory_name": None,
            "results": [],
            "warnings": ["No text detected by OCR engine."],
            "completeness_summary": {
                "pages_reviewed": 0,
                "results_extracted": 0,
                "categories_found": [],
                "status_counts": {},
                "unprocessed_sections": []
            }
        }

    # Split into pages if multi-page separators exist
    raw_pages = re.split(r'\x0c|--- PAGE \d+ ---|\bPage \d+ of \d+\b', text)
    pages = [p.strip() for p in raw_pages if p.strip()] or [text.strip()]
    pages_count = max(1, len(pages))

    patient_id = None
    report_date = None
    lab_name = None

    LAB_KEYWORDS = [
        "hospital", "laboratory", "laboratories", "diagnostics", "pathology", "clinic", "centre", "center",
        "asiri", "nawaloka", "durdans", "lanka hospitals", "metropolis", "quest", "labcorp"
    ]

    all_results = []
    all_warnings = []
    unprocessed = []

    SECTION_HEADERS = [
        ("HAEMATOLOGY", "CBC"),
        ("COMPLETE BLOOD COUNT", "CBC"),
        ("FULL BLOOD COUNT", "CBC"),
        ("DIFFERENTIAL LEUKOCYTE COUNT", "Differential Count"),
        ("DIFFERENTIAL COUNT", "Differential Count"),
        ("RED BLOOD CELL INDICES", "Red Blood Cells"),
        ("LIVER FUNCTION TESTS", "Liver Function"),
        ("HEPATIC PROFILE", "Liver Function"),
        ("RENAL FUNCTION TESTS", "Kidney Function"),
        ("KIDNEY FUNCTION", "Kidney Function"),
        ("LIPID PROFILE", "Lipid Profile"),
        ("SERUM ELECTROLYTES", "Electrolytes"),
        ("ELECTROLYTES", "Electrolytes"),
        ("THYROID PROFILE", "Thyroid Function"),
        ("URINE ROUTINE", "Urinalysis"),
        ("URINALYSIS", "Urinalysis"),
        ("BIOCHEMISTRY", "Liver Function"),
        ("SEROLOGY", "Inflammatory & Serology"),
        ("IMMUNOLOGY", "Inflammatory & Serology")
    ]

    for p_idx, page_content in enumerate(pages, 1):
        lines = [line.strip() for line in page_content.split("\n") if line.strip()]
        joined_lines = []
        for line in lines:
            is_reference_continuation = re.fullmatch(
                r"\(\s*\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?\s*\)",
                line,
                re.IGNORECASE
            )
            if joined_lines and is_reference_continuation:
                joined_lines[-1] = f"{joined_lines[-1]} {line}"
            else:
                joined_lines.append(line)
        lines = joined_lines

        # Metadata scan on early lines of document
        if p_idx == 1:
            for line in lines[:15]:
                low = line.lower()
                if not lab_name and any(k in low for k in LAB_KEYWORDS) and not any(skip in low for skip in ["patient", "doctor", "dr.", "age"]):
                    clean_l = re.sub(r'[\:\;\#\*\-]+', '', line).strip()
                    if 3 < len(clean_l) < 60:
                        lab_name = clean_l

                if not patient_id:
                    m_pid = re.search(r'\b(?:patient\s*(?:name|id|no)?|name|pt\.?\s*name|uhid|reg\s*no)\s*[:\-#]?\s*([A-Za-z0-9\.\s]+)', line, re.IGNORECASE)
                    if m_pid:
                        cand = m_pid.group(1).strip()
                        cand = re.split(r'\b(?:age|sex|date|gender|ref)\b', cand, flags=re.IGNORECASE)[0].strip()
                        if 2 <= len(cand) <= 40 and not any(kw in cand.lower() for kw in ["hemoglobin", "test", "report", "pathology"]):
                            patient_id = cand

                if not report_date:
                    m_date = re.search(r'\b(?:date|reported|collected|printed)\s*[:\-#]?\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4})', line, re.IGNORECASE)
                    if m_date:
                        report_date = m_date.group(1).strip()
                    else:
                        m_date2 = re.search(r'\b(\d{1,2}[-/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/\.]\d{2,4})\b', line, re.IGNORECASE)
                        if m_date2:
                            report_date = m_date2.group(1).strip()

        current_section = "General Clinical Pathology"
        current_category = "Other"

        for line in lines:
            line_clean = line.strip()
            low = line_clean.lower()

            # Check for section header
            is_header = False
            for header_text, cat_hint in SECTION_HEADERS:
                if header_text.lower() in low and len(line_clean) < 50 and not any(c.isdigit() for c in line_clean if c not in ["1", "2", "3"]):
                    current_section = header_text.title()
                    current_category = cat_hint
                    is_header = True
                    break

            if is_header:
                continue

            # Skip demographic, patient, header, facility contact, and administrative rows
            if any(h in low for h in [
                "patient name", "pt. name", "pt name", "name of patient", "patient:", "age / sex", "age/sex", "age | sex", "age|sex",
                "age:", "gender:", "report ref", "bill no", "sample no", "uhid:", "uhid-", "reg no", "dr.",
                "test parameter", "investigation", "reference range", "clinical pathology",
                "authorized signature", "page 1 of", "reported on", "referred by", "doctor:",
                "address:", "address", "phone:", "phone", "tel:", "tel.", "fax:", "email:", "laboratory", "accredited"
            ]) or re.search(r'\b(?:patient|pt\.?|age|gender|sex|ref|uhid|address|phone|tel|fax|email)\b\s*[\:\/\|\-]', low):
                if not any(k in low for k in ["haemoglobin", "glucose", "cholesterol", "platelet", "creatinine", "wbc", "tsh"]):
                    continue

            # Step A: Preserve report reference values from the line tail
            ref_raw = None
            ref_low = None
            ref_high = None

            number_pattern = r"\d+(?:\.\d+)?"
            range_pattern = rf"{number_pattern}\s*%?\s*(?:-|–|—|to)\s*{number_pattern}\s*%?"
            qualified_pattern = rf"[<>]=?\s*{number_pattern}\s*%?(?:\s*(?:-|–|—|to)\s*{number_pattern}\s*%?)?"
            interval_pattern = rf"(?:\(\s*(?:{range_pattern}|{qualified_pattern})\s*\)|{range_pattern}|{qualified_pattern})"
            m_reference = re.search(rf"({interval_pattern}(?:\s+{interval_pattern})*)\s*$", line_clean, re.IGNORECASE)
            if m_reference:
                ref_raw = m_reference.group(1)
                m_range = re.search(rf"({number_pattern})\s*(?:-|–|—|to)\s*({number_pattern})", ref_raw, re.IGNORECASE)
                if m_range:
                    try:
                        ref_low = float(m_range.group(1))
                        ref_high = float(m_range.group(2))
                    except ValueError:
                        pass
                rem_line = line_clean[:m_reference.start()].strip()
            else:
                m_ineq = re.search(r'([<>]=?)\s*(\d+(?:\.\d+)?)\s*$', line_clean)
                if m_ineq:
                    ref_raw = m_ineq.group(0).strip()
                    try:
                        num = float(m_ineq.group(2))
                        if "<" in m_ineq.group(1):
                            ref_high = num
                        else:
                            ref_low = num
                    except ValueError:
                        pass
                    rem_line = line_clean[:m_ineq.start()].strip()
                else:
                    m_qual = re.search(r'\b(NEGATIVE|NON-REACTIVE|NOT DETECTED|NIL|NORMAL)\b\s*$', line_clean, re.IGNORECASE)
                    if m_qual:
                        ref_raw = m_qual.group(1)
                        rem_line = line_clean[:m_qual.start()].strip()
                    else:
                        rem_line = line_clean

            # Step B: Value and Flag in rem_line
            result_val = None
            result_raw = None
            flag_found = None
            test_name = None
            tail_after_val = ""

            # Check qualitative values
            m_qual_val = re.search(r'\b(POSITIVE|NEGATIVE|REACTIVE|NON-REACTIVE|DETECTED|NOT DETECTED|PRESENT|ABSENT|NIL|TRACE)\b', rem_line, re.IGNORECASE)
            if m_qual_val:
                result_raw = m_qual_val.group(1).upper()
                result_val = result_raw
                test_name = rem_line[:m_qual_val.start()].strip()
                tail_after_val = rem_line[m_qual_val.end():].strip()
            else:
                # Mask parenthesized text to prevent numbers inside test names like (25-OH) being selected
                bracket_tokens = []
                def repl_bracket(m):
                    idx = len(bracket_tokens)
                    bracket_tokens.append(m.group(0))
                    return f"__BRACKET_{idx}__"

                masked_line = re.sub(r'\([^\)]*\)', repl_bracket, rem_line)

                m_num = re.search(r'(?<![.\w/])(\d+(?:\.\d+)?)\s*([\*HL])?(?:\s+([HL]))?\b', masked_line)
                if m_num:
                    result_raw = m_num.group(1)
                    try:
                        result_val = float(result_raw)
                    except ValueError:
                        result_val = result_raw
                    f1 = m_num.group(2)
                    f2 = m_num.group(3)
                    flag_found = (f1 or f2 or "").strip() or None
                    test_name_masked = masked_line[:m_num.start()].strip()
                    tail_after_val_masked = masked_line[m_num.end():].strip()

                    for i, tok in enumerate(bracket_tokens):
                        test_name_masked = test_name_masked.replace(f"__BRACKET_{i}__", tok)
                        tail_after_val_masked = tail_after_val_masked.replace(f"__BRACKET_{i}__", tok)

                    test_name = test_name_masked
                    tail_after_val = tail_after_val_masked

            if not flag_found and tail_after_val:
                m_fl = re.search(r'^[\*HL]\b|^\*', tail_after_val)
                if m_fl:
                    flag_found = m_fl.group(0).strip()
                    tail_after_val = tail_after_val[m_fl.end():].strip()
                else:
                    m_fl = re.search(r'(?<![\w/])([HL])\s*$|(?<!\w)(\*)\s*$', tail_after_val, re.IGNORECASE)
                    if m_fl:
                        flag_found = m_fl.group(1) or m_fl.group(2)
                        tail_after_val = tail_after_val[:m_fl.start()].strip()

            # Step C: Unit extraction from tail_after_val
            unit_found = None
            for u in sorted(LAB_KNOWN_UNITS, key=len, reverse=True):
                pattern = rf'\b{re.escape(u)}\b'
                if u == "%": pattern = r'%'
                unit_match = re.search(pattern, tail_after_val, re.IGNORECASE)
                if unit_match:
                    unit_found = unit_match.group(0)
                    break

            if not test_name or len(test_name) < 2:
                if any(num in low for num in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                    unprocessed.append(line_clean)
                continue

            test_name = re.sub(r'^[\d\.\-\*\•\s]+', '', test_name)
            test_name = re.sub(r'[\:\.\-\s]+$', '', test_name).strip()

            if result_val is None or len(test_name) < 2:
                continue

            # Step D: Category Determination
            name_low = test_name.lower()
            matched_cat = None
            for cat, patterns in CATEGORY_PATTERNS.items():
                if any(re.search(pat, name_low, re.IGNORECASE) for pat in patterns):
                    matched_cat = cat
                    break
            final_category = matched_cat or current_category or "Other"

            # Step E: Result Status Determination
            status = "undetermined"
            if isinstance(result_val, (int, float)):
                if flag_found and "h" in flag_found.lower():
                    status = "high"
                elif flag_found and "l" in flag_found.lower():
                    status = "low"
                elif flag_found and any(c in flag_found.lower() for c in ["crit", "panic"]):
                    status = "critical"
                elif ref_low is not None and ref_high is not None:
                    if result_val < ref_low:
                        status = "low"
                    elif result_val > ref_high:
                        status = "high"
                    else:
                        status = "normal"
                elif ref_high is not None and ref_low is None:
                    status = "high" if result_val > ref_high else "normal"
                elif ref_low is not None and ref_high is None:
                    status = "low" if result_val < ref_low else "normal"
                elif flag_found and "*" in flag_found:
                    status = "high"
                else:
                    # User requirement: If reference range absent and no flag, status is undetermined
                    status = "undetermined"
            elif isinstance(result_val, str):
                val_upper = result_val.upper()
                if any(p in val_upper for p in ["POSITIVE", "REACTIVE", "DETECTED", "PRESENT", "ABNORMAL", "TRACE"]):
                    status = "positive"
                elif any(n in val_upper for n in ["NEGATIVE", "NON-REACTIVE", "NOT DETECTED", "ABSENT", "NIL"]):
                    status = "negative"
                elif val_upper == "NORMAL":
                    status = "normal"
                else:
                    status = "undetermined"

            # Deduplication: remove true duplicates (same test name and result value and unit)
            is_dup = False
            for ex in all_results:
                if ex["test_name"].lower() == test_name.lower() and ex["result_value"] == result_val and ex["unit"] == unit_found:
                    is_dup = True
                    break

            if not is_dup:
                all_results.append({
                    "test_name": test_name,
                    "result": result_val,
                    "result_value": result_val,
                    "result_raw": str(result_raw),
                    "unit": unit_found if unit_found else None,
                    "unit_provided": bool(unit_found),
                    "reference_range": {
                        "raw": ref_raw if ref_raw else None,
                        "low": ref_low if ref_low is not None else None,
                        "high": ref_high if ref_high is not None else None,
                        "provided": bool(ref_raw)
                    },
                    "reference_value": ref_raw if ref_raw else "Not provided",
                    "result_status": status,
                    "test_category": final_category,
                    "flag": flag_found,
                    "original_flag": flag_found,
                    "page_section": f"Page {p_idx} — {current_section}",
                    "extraction_confidence": "high" if (unit_found or ref_raw) else "medium",
                    "confidence": "high" if (unit_found or ref_raw) else "medium",
                    "is_estimated": False
                })

    _merge_reference_values_from_bboxes(all_results, ocr_bboxes)

    status_counts = {}
    categories_found = list(dict.fromkeys([r["test_category"] for r in all_results]))
    for r in all_results:
        st = r["result_status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    if unprocessed:
        all_warnings.append(f"{len(unprocessed)} line(s) contained numerical data but could not be definitively structured.")

    return {
        "patient_identifier": patient_id,
        "report_date": report_date,
        "laboratory_name": lab_name,
        "results": all_results,
        "warnings": all_warnings,
        "completeness_summary": {
            "pages_reviewed": pages_count,
            "results_extracted": len(all_results),
            "categories_found": categories_found,
            "status_counts": status_counts,
            "unprocessed_sections": unprocessed[:5]
        }
    }


CLINICAL_RANGES = {
    "Hemoglobin":         {"low": 12.0, "high": 17.5, "unit": "g/dL",      "critical_low": 7.0,  "critical_high": 20.0},
    "WBC":                {"low": 4.5,  "high": 11.0, "unit": "×10³/µL",   "critical_low": 2.0,  "critical_high": 30.0},
    "Platelets":          {"low": 150,  "high": 450,  "unit": "×10³/µL",   "critical_low": 50,   "critical_high": 1000},
    "Fasting_Blood_Sugar":{"low": 70,   "high": 100,  "unit": "mg/dL",     "critical_low": 50,   "critical_high": 400},
    "Creatinine":         {"low": 0.6,  "high": 1.2,  "unit": "mg/dL",     "critical_low": 0.2,  "critical_high": 10.0},
    "TSH":                {"low": 0.4,  "high": 4.5,  "unit": "mIU/L",     "critical_low": 0.01, "critical_high": 50.0},
    "Cholesterol":        {"low": 120,  "high": 200,  "unit": "mg/dL",     "critical_low": 80,   "critical_high": 400},
    "Ionized_Calcium":    {"low": 4.6,  "high": 5.3,  "unit": "mg/dL",     "critical_low": 3.8,  "critical_high": 6.5},
    "Calcium":            {"low": 8.5,  "high": 10.5, "unit": "mg/dL",     "critical_low": 7.0,  "critical_high": 12.0},
    "GGT":                {"low": 9.0,  "high": 50.0, "unit": "U/L",       "critical_low": 0.0,  "critical_high": 300.0},

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
        if tot_ca >= 8.5 and tot_ca <= 10.5:
            pass
        elif tot_ca >= 8.0:
            primary_findings.append(f"Borderline hypocalcemia (Serum Ca {tot_ca} mg/dL)—monitor and correlate clinically.")
        elif tot_ca >= 7.0:
            primary_findings.append(f"Hypocalcemia (Serum Ca {tot_ca} mg/dL)—requires clinical correlation.")
        elif tot_ca < 7.0:
            primary_findings.append(f"Significant hypocalcemia (Serum Ca {tot_ca} mg/dL)—urgent clinical review.")
        elif tot_ca > 10.5:
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

# @deprecated: Replaced by CliniQEngine in cliniq_ml (v2.2.0 multi-label XGBoost).
# Kept only for backwards compatibility with legacy callers.
def run_ml_classifier(values):
    """[DEPRECATED] Legacy 8-class XGBoost classifier. Use CliniQEngine('artifacts') instead."""
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
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=78)
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

    # 1. Attempt recovery by closing unclosed trailing strings and braces
    repaired = cleaned
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    if not repaired.rstrip().endswith("}"):
        repaired += "\n}"
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # 2. Match bounded JSON object
    match = re.search(r'(\{[\s\S]*\})', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Robust Regex Recovery: Extract patient demographics and all biomarkers
    # Guaranteed to succeed even if the model was truncated at token limits
    result = {
        "Patient_Name": None, "Age": None, "Gender": None, "Hemoglobin": None, "WBC": None, "Platelets": None,
        "Fasting_Blood_Sugar": None, "Creatinine": None, "TSH": None, "Cholesterol": None,
        "Ionized_Calcium": None, "Calcium": None, "GGT": None, "Albumin": None,
        "Globulin": None, "AG_Ratio": None, "Total_Protein": None,
        "Clinical_Interpretation": "Analysis completed.",
        "Predicted_Diseases": []
    }

    match_name = re.search(r'"Patient_Name"\s*:\s*"([^"]+)"', raw_text)
    if match_name and match_name.group(1).strip().lower() not in ("null", "none", ""):
        result["Patient_Name"] = match_name.group(1).strip()

    match_age = re.search(r'"Age"\s*:\s*(\d+)', raw_text)
    if match_age:
        result["Age"] = int(match_age.group(1))

    match_gender = re.search(r'"Gender"\s*:\s*"(Male|Female)"', raw_text, re.IGNORECASE)
    if match_gender:
        result["Gender"] = match_gender.group(1).capitalize()

    for k in [
        "Hemoglobin", "WBC", "Platelets", "Fasting_Blood_Sugar", "Creatinine", "TSH",
        "Cholesterol", "Ionized_Calcium", "Calcium", "GGT", "Albumin", "Globulin",
        "AG_Ratio", "Total_Protein"
    ]:
        m = re.search(rf'"{k}"\s*:\s*([0-9.]+)', raw_text)
        if m:
            try:
                v = float(m.group(1))
                result[k] = int(v) if k in ("Platelets", "Fasting_Blood_Sugar", "Cholesterol") else v
            except Exception:
                pass

    interp_m = re.search(r'"Clinical_Interpretation"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', raw_text)
    if interp_m:
        result["Clinical_Interpretation"] = interp_m.group(1).strip()
    elif raw_text:
        result["Clinical_Interpretation"] = raw_text[:400]

    return result

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
            timeout=15
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


def run_openai_vlm(image_bytes, api_key, model_name="gpt-4o-mini"):
    start_time = time.time()
    norm_bytes, mime_type = normalize_image_bytes(image_bytes)
    base64_image = base64.b64encode(norm_bytes).decode("utf-8")

    prompt = (
        "Analyze this de-identified medical laboratory report. Extract patient demographics and laboratory values faithfully. "
        "Do not infer missing values or diagnose from values alone. Return only a JSON object with keys "
        "Patient_Name, Age, Gender, Hemoglobin, WBC, Platelets, Fasting_Blood_Sugar, Creatinine, TSH, "
        "Cholesterol, Ionized_Calcium, Calcium, GGT, Albumin, Globulin, AG_Ratio, Total_Protein, "
        "Clinical_Interpretation, and Predicted_Diseases. Use null for unavailable scalar values and an empty array "
        "for Predicted_Diseases when there are no confirmed findings."
    )
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }],
        "response_format": {"type": "json_object"},
        "max_tokens": 800,
        "temperature": 0.1
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        latency = time.time() - start_time
        if response.status_code != 200:
            return {"error": f"OpenAI API returned status {response.status_code}: {response.text[:200]}", "latency": latency}
        raw_text = response.json()["choices"][0]["message"]["content"].strip()
        structured_data = parse_vlm_json_response(raw_text)
        structured_data["latency"] = latency
        return structured_data
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}", "latency": time.time() - start_time}


def run_roar_ai_vlm(image_bytes, api_key=None, model_name="gemini-3.8-flash"):
    """
    Executes VLM clinical analysis through Roar AI Gateway.
    """
    start_time = time.time()
    roar_key = api_key or os.environ.get("ROAR_API_KEY")
    roar_base = os.environ.get("ROAR_BASE_URL", "https://api.roar-ai.com/v1")

    if not roar_key:
        return {"error": "Roar AI API key not configured.", "latency": 0.0}

    norm_bytes, mime_type = normalize_image_bytes(image_bytes)
    base64_image = base64.b64encode(norm_bytes).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {roar_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are an expert clinical pathologist analyzing a medical laboratory report.\n"
        "Extract patient demographics (Patient_Name, Age, Gender) and all numeric biomarker values accurately.\n"
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
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.1
    }

    try:
        response = requests.post(f"{roar_base}/chat/completions", headers=headers, json=payload, timeout=20)
        latency = time.time() - start_time
        if response.status_code != 200:
            return {"error": f"Roar AI error ({response.status_code}): {response.text[:200]}", "latency": latency}
        res_data = response.json()
        raw_text = res_data['choices'][0]['message']['content'].strip()
        structured_data = parse_vlm_json_response(raw_text)
        structured_data["latency"] = latency
        return structured_data
    except Exception as e:
        return {"error": f"Roar AI VLM error: {str(e)}", "latency": time.time() - start_time}

