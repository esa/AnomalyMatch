#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from anomaly_match_ui.widget import shorten_filename


class TestShortenFilename:
    """Tests for the shorten_filename helper function."""

    def test_short_filename_unchanged(self):
        """Filenames within max length should remain unchanged."""
        assert shorten_filename("short.fits", max_length=25) == "short.fits"
        assert shorten_filename("image.jpg", max_length=25) == "image.jpg"

    def test_long_filename_shortened(self):
        """Long filenames should be shortened to max_length."""
        long_name = "very_long_filename_that_exceeds_limit.fits"
        result = shorten_filename(long_name, max_length=25)
        assert len(result) <= 25
        assert result.endswith(".fits")
        assert "..." in result

    def test_filename_with_multiple_dots(self):
        """Filenames with multiple dots should preserve only the extension."""
        name = "image.2024.01.15.observation.fits"
        result = shorten_filename(name, max_length=25)
        assert len(result) <= 25
        assert result.endswith(".fits")
        assert "..." in result

    def test_filename_without_extension(self):
        """Filenames without extension should still be shortened correctly."""
        name = "very_long_filename_without_any_extension"
        result = shorten_filename(name, max_length=25)
        assert len(result) <= 25
        assert "..." in result

    def test_exact_max_length(self):
        """Filename exactly at max_length should be unchanged."""
        name = "exactly_25_chars_long.fit"
        assert len(name) == 25
        assert shorten_filename(name, max_length=25) == name

    def test_very_short_max_length(self):
        """Very short max_length should still produce valid output."""
        name = "some_filename.fits"
        result = shorten_filename(name, max_length=10)
        assert len(result) <= 10
        assert "..." in result

    def test_preserves_start_and_end(self):
        """Shortened name should contain parts of the original start and end."""
        name = "START_middle_content_END.fits"
        result = shorten_filename(name, max_length=20)
        assert result.startswith("START")
        # Should contain some part of the end before the extension
        assert "END" in result or "..." in result
