from ocr_engine import extract_all_laboratory_values


def test_extracts_percent_and_absolute_reference_ranges_verbatim():
    report = """DIFFERENTIAL COUNT
Neutrophils 35.0 % L 40%-80% (2.0-7.0)"""

    result = extract_all_laboratory_values(report)["results"][0]

    assert result["test_name"] == "Neutrophils"
    assert result["result"] == 35.0
    assert result["result_value"] == 35.0
    assert result["unit"] == "%"
    assert result["original_flag"] == "L"
    assert result["flag"] == "L"
    assert result["reference_value"] == "40%-80% (2.0-7.0)"
    assert result["reference_range"]["raw"] == "40%-80% (2.0-7.0)"


def test_extracts_single_reference_interval_without_reformatting():
    report = "Haemoglobin 13.20 g/dL 12.0–16.0"

    result = extract_all_laboratory_values(report)["results"][0]

    assert result["result_value"] == 13.2
    assert result["unit"] == "g/dL"
    assert result["reference_value"] == "12.0–16.0"


def test_marks_absent_reference_as_not_provided():
    result = extract_all_laboratory_values("Glucose 92 mg/dL")["results"][0]

    assert result["reference_value"] == "Not provided"
    assert result["reference_range"]["raw"] is None


def test_extracts_wrapped_basophil_reference_range():
    report = """TOTAL WHITE CELL COUNT 4.9 10^9/L 4.0 - 11.0
DIFFERENTIAL COUNT
NEUTROPHILS 35.0 % 1.7 L 40%-80% (2.0-7.0)
LYMPHOCYTES 50.4 % 2.5 H 20%-40% (1.0-3.0)
MONOCYTES 13.8 % 0.7 H 02%-10% (0.2-1.0)
EOSINOPHILS 0.2 % 0.0 L 01%-06% (0.02-0.5)
BASOPHILS 0.6 % 0.0 <0.1%-02%
(0.02-0.1)
HAEMOGLOBIN AND RBC PARAMETERS
HAEMOGLOBIN 12.4 g/dL 11.8 - 14.8
RED BLOOD CELLS 4.48 10^12/L 3.8 - 4.8
MEAN CELL VOLUME 82.8 fl 76.0 - 96.0
HAEMATOCRIT 37.1 L/L(%) 36.0 - 44.0
MEAN CELL HAEMOGLOBIN 27.7 pg 27.0 - 33.0
M.C.H. CONCENTRATION 33.4 g/dL 32.0 - 36.0
RED CELLS DISTRIBUTION WIDTH 13.3 % 12.0 - 15.0
PLATELET COUNT 380 10^9/L 150 - 400"""

    results = extract_all_laboratory_values(report)["results"]
    by_name = {result["test_name"].upper(): result for result in results}

    expected_references = {
        "TOTAL WHITE CELL COUNT": "4.0 - 11.0",
        "NEUTROPHILS": "40%-80% (2.0-7.0)",
        "LYMPHOCYTES": "20%-40% (1.0-3.0)",
        "MONOCYTES": "02%-10% (0.2-1.0)",
        "EOSINOPHILS": "01%-06% (0.02-0.5)",
        "BASOPHILS": "<0.1%-02% (0.02-0.1)",
        "HAEMOGLOBIN": "11.8 - 14.8",
        "RED BLOOD CELLS": "3.8 - 4.8",
        "MEAN CELL VOLUME": "76.0 - 96.0",
        "HAEMATOCRIT": "36.0 - 44.0",
        "MEAN CELL HAEMOGLOBIN": "27.0 - 33.0",
        "M.C.H. CONCENTRATION": "32.0 - 36.0",
        "RED CELLS DISTRIBUTION WIDTH": "12.0 - 15.0",
        "PLATELET COUNT": "150 - 400",
    }

    assert len(results) == len(expected_references)
    assert {name: by_name[name]["reference_value"] for name in expected_references} == expected_references
    assert all(by_name[name]["result_status"] != "undetermined" for name in expected_references)


def test_recovers_reference_from_rightmost_ocr_column():
    def word(text, x0, y0, x1, y1):
        return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, 0.99)

    bboxes = [
        word("REFERENCE", 600, 20, 675, 30),
        word("VALUE", 690, 20, 730, 30),
        word("NEUTROPHILS", 50, 50, 150, 60),
        word("35.0", 350, 50, 380, 60),
        word("%", 450, 50, 460, 60),
        word("L", 500, 50, 508, 60),
        word("40%-80%", 600, 50, 655, 60),
        word("(2.0-7.0)", 675, 50, 730, 60),
    ]

    result = extract_all_laboratory_values(
        "NEUTROPHILS 35.0 % L",
        ocr_bboxes=bboxes,
    )["results"][0]

    assert result["reference_value"] == "40%-80% (2.0-7.0)"
    assert result["result_status"] == "low"