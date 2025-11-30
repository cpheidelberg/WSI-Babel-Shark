"""
Unit tests for babel_shark.slide_id_generator

This test focuses on the core identifier logic:
- Normalization of LabID / Year / CaseNumber / Pot / BlockID / Section
- CaseID construction
- DataMatrix reconstruction via build_datamatrix(...)
- SlideID generation from DataMatrix + Stain
- "method of extraction" mapping from ExtractionMethod
"""

import pandas as pd
import pytest

from babel_shark.slide_id_generator import compute_identifiers, build_datamatrix


def test_compute_identifiers_builds_caseid_datamatrix_and_slideid():
    """
    Scenario:
        LabID        = "L"
        Year         = "24"        -> Year4 = "2024"
        CaseNumber   = "123"       -> Case6 = "000123"
        Pot          = "A"
        BlockID      = "2"
        Section      = "1"
        Stain        = "PAS"
        ExtractionMethod = "ROIbase"

    Expectations:
        - Year4 is normalized to "2024"
        - Case6 is normalized/padded to "000123"
        - CaseID = "L2024000123"
        - DataMatrix is rebuilt using build_datamatrix()
        - SlideID = f"{DataMatrix}-PAS"
        - "method of extraction" is "ROIbase"
    """

    # Minimal input frame mimicking the upstream merged Excel skeleton
    df_in = pd.DataFrame(
        [
            {
                "FileName": "sample.svs",
                "DataMatrix": "",          # empty so that compute_identifiers will rebuild it
                "LabID": "L",
                "Year": "24",              # 2-digit year; should become 4-digit
                "CaseNumber": "123",       # should become 6-digit zero-padded
                "Pot": "A",
                "BlockID": "2",
                "Section": "1",
                "ExtractionMethod": "ROIbase",
                "Stain": "PAS",
            }
        ]
    )

    # Run identifier computation
    df_out = compute_identifiers(df_in)
    assert len(df_out) == 1, "Expected exactly one row after compute_identifiers()"
    row = df_out.iloc[0]

    # Year4 and Case6 normalization
    assert row["Year4"] == "2024", "Year4 should be normalized from '24' to '2024'"
    assert row["Case6"] == "000123", "Case6 should be zero-padded to 6 digits"

    # CaseID = LabID + Year4 + Case6
    assert (
        row["CaseID"] == "L2024000123"
    ), "CaseID should be LabID + Year4 + Case6"

    # DataMatrix should be rebuilt according to build_datamatrix template
    expected_dm = build_datamatrix("L", "2024", "000123", "A", "2", "1")
    assert expected_dm, "Expected build_datamatrix to return a non-empty value"
    assert (
        row["DataMatrix"] == expected_dm
    ), "DataMatrix should match the value computed by build_datamatrix()"

    # Stain normalization:
    # for "PAS" input we expect the same label back (no H&E normalization here)
    assert row["Stain"] == "PAS", "Stain should remain 'PAS' for this test case"

    # SlideID must be built only when DataMatrix is valid and non-nan
    expected_slide_id = f"{expected_dm}-PAS"
    assert (
        row["SlideID"] == expected_slide_id
    ), "SlideID should be DataMatrix + '-' + Stain"

    # method of extraction:
    # - If ExtractionMethod == 'roibase' (case-insensitive) → 'ROIbase'
    # - Otherwise → 'Datamatrix'
    assert (
        row["method of extraction"] == "ROIbase"
    ), "Expected 'method of extraction' to map 'ROIbase' correctly from ExtractionMethod"

