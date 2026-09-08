"""
SONAR-GUARD — Unit Tests: Preprocessing
Tests actual preprocessing pipeline behaviour.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import cv2

from src.preprocessing.sonar_preprocessor import SonarPreprocessor, PreprocessResult


@pytest.fixture
def sample_grayscale():
    """256x256 grayscale test image with some structure."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (160, 160), 200, -1)
    noise = np.random.randint(0, 30, (256, 256), dtype=np.uint8)
    return cv2.add(img, noise)


@pytest.fixture
def sample_bgr(sample_grayscale):
    return cv2.cvtColor(sample_grayscale, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def preprocessor():
    return SonarPreprocessor(
        target_width=128, target_height=128,
        median_ksize=5, gaussian_ksize=0,
        use_clahe=True, clahe_clip_limit=2.0,
    )


class TestSonarPreprocessor:
    def test_init_valid(self):
        p = SonarPreprocessor(target_width=640, target_height=640, median_ksize=5)
        assert p.target_width == 640

    def test_init_even_kernel_raises(self):
        with pytest.raises(ValueError, match="median_ksize must be odd"):
            SonarPreprocessor(median_ksize=4)

    def test_grayscale_input(self, preprocessor, sample_grayscale):
        result = preprocessor.process(sample_grayscale, image_id="test_gray")
        assert result.success
        assert result.preprocessed is not None
        assert result.preprocessed.shape == (128, 128)

    def test_bgr_input(self, preprocessor, sample_bgr):
        result = preprocessor.process(sample_bgr, image_id="test_bgr")
        assert result.success
        assert result.preprocessed.ndim == 2  # output is grayscale

    def test_output_size(self, preprocessor, sample_grayscale):
        result = preprocessor.process(sample_grayscale, image_id="test_size")
        h, w = result.preprocessed.shape[:2]
        assert w == preprocessor.target_width
        assert h == preprocessor.target_height

    def test_stages_recorded(self, preprocessor, sample_grayscale):
        result = preprocessor.process(sample_grayscale, image_id="test_stages",
                                      record_stages=True)
        assert result.success
        assert "04_clahe" in result.stages
        assert "05_resized" in result.stages

    def test_none_image_returns_error(self, preprocessor):
        result = preprocessor.process(None, image_id="test_none")
        assert not result.success
        assert result.error is not None

    def test_empty_image_returns_error(self, preprocessor):
        result = preprocessor.process(np.array([]), image_id="test_empty")
        assert not result.success

    def test_timing_recorded(self, preprocessor, sample_grayscale):
        result = preprocessor.process(sample_grayscale, image_id="test_timing")
        assert result.preprocessing_time_s > 0

    def test_uint16_input(self, preprocessor):
        img16 = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
        result = preprocessor.process(img16, image_id="test_uint16")
        assert result.success
        assert result.preprocessed.dtype == np.uint8

    def test_clahe_disabled(self):
        p = SonarPreprocessor(target_width=128, target_height=128,
                               median_ksize=3, use_clahe=False)
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = p.process(img, image_id="test_no_clahe")
        assert result.success
        assert "CLAHE disabled" in result.warnings[0]

    def test_params_in_result(self, preprocessor, sample_grayscale):
        result = preprocessor.process(sample_grayscale, image_id="test_params")
        assert "median_ksize" in result.params_used
        assert result.params_used["median_ksize"] == preprocessor.median_ksize

