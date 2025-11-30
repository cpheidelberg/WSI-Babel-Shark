"""
Unit tests for babel_shark.label_extractor

These tests verify the functional parts of the LabelExtractor that do NOT
depend on OpenSlide, OCR, or external DLLs. Specifically, we test:

1. generate_output_name()        → ensures filenames are standardized to .png
2. PNG extraction path logic     → crop + rotate + atomic save
3. File output correctness       → resulting PNG is created and loadable

This makes the tests:
- fully deterministic
- lightweight
- independent of external libraries (OpenSlide)
- aligned with the REAL API of label_extractor.py
"""

from pathlib import Path
from PIL import Image
from babel_shark.label_extractor import LabelExtractor


def test_generate_output_name():
    """
    generate_output_name() should convert any input path to a .png filename,
    preserving the original stem.

    Example:
        '/tmp/sample.svs' → 'sample.png'
    """
    input_path = Path("/tmp/sample_slide.svs")

    # LabelExtractor requires valid paths, but generate_output_name() is static logic.
    extractor = LabelExtractor(
        input_dir=Path("."),
        output_dir=Path("."),
        log_file_path=Path("log.txt")
    )

    out = extractor.generate_output_name(input_path)
    assert out == "sample_slide.png"


def test_png_extraction(tmp_path):
    """
    This test exercises the *actual PNG processing pipeline* used in LabelExtractor:

    Steps tested:
        1. Load a real PNG image
        2. Apply crop ratio (label_crop_ratio)
        3. Rotate image (rotation_degrees)
        4. Save output using atomic_save_pil_image()
    
    Validation:
        - Output PNG file must be created
        - The file must be readable by Pillow
        - Image dimensions must reflect crop + rotation effects
    """

    # --- 1. Create temporary folders -----------------------------------------
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_file = tmp_path / "logs" / "failures.log"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Create a synthetic PNG image -------------------------------------
    # A simple 200×100 red image.
    img_path = input_dir / "fake_label.png"
    img = Image.new("RGB", (200, 100), color=(255, 0, 0))
    img.save(img_path)

    # Expected crop width = 200 * 0.3 = 60 pixels.

    # --- 3. Create a real LabelExtractor instance -----------------------------
    extractor = LabelExtractor(
        input_dir=input_dir,
        output_dir=output_dir,
        log_file_path=log_file,
        label_crop_ratio=0.3,      # crop left 30% of the width
        rotation_degrees=-90,      # rotate counterclockwise
        macro_tag="macro",
    )

    # --- 4. Run the extraction on the synthetic PNG ---------------------------
    extractor.extract_label()

    # --- 5. Validate that output PNG exists ----------------------------------
    out_file = output_dir / "fake_label.png"
    assert out_file.exists(), "Output cropped PNG was not created."

    # --- 6. Validate output integrity ----------------------------------------
    # Load the resulting image; it must be readable.
    output_image = Image.open(out_file)

    # Because rotation = -90 degrees, the resulting dimensions will be swapped.
    # Original crop size = (60, 100). After rotation → (100, 60).
    expected_w = 100
    expected_h = 60

    assert output_image.size == (expected_w, expected_h), (
        f"Unexpected output size {output_image.size}, "
        f"expected {(expected_w, expected_h)}."
    )
