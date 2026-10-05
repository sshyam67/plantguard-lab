import io
from PIL import Image
from app import app


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(output, "PNG")
    output.seek(0)
    return output


def test_health():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_demo_prediction_is_explicit():
    client = app.test_client()
    response = client.post("/api/predict", data={"image": (image_bytes(), "leaf.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["mode"] in {"demo", "trained"}
    if response.json["mode"] == "demo":
        assert response.json["prediction"] == "Model not trained"

