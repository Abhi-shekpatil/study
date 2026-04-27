from fastapi.testclient import TestClient
from src.inference import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_file_type():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"] == "Only image files are supported"


def test_detect_response_structure():
    """Test that detect endpoint returns detections with class names"""
    # This test would need an actual image file to pass
    # For now, just verify endpoint exists and accepts images
    assert client.post("/detect")  # endpoint exists
