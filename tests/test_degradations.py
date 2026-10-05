import pytest
from PIL import Image
from src.degradations import degrade


@pytest.mark.parametrize("kind", ["blur", "dark", "bright", "noise", "resolution", "jpeg"])
def test_degradation_preserves_size(kind):
    image = Image.new("RGB", (64, 48), (80, 140, 70))
    assert degrade(image, kind, 2).size == image.size


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        degrade(Image.new("RGB", (16, 16)), "blur", 4)

