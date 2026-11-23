import pytest
import httpx
import base64

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=300.0) as c:
        yield c


def check_image(image):
    assert isinstance(image, str)
    assert len(image) > 0
    decoded = base64.b64decode(image)
    assert decoded.startswith(b"\x89PNG")


def test_api_info(client):
    response = client.get("/api/info")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert data["classes"] == 10


def test_generate(client):
    response = client.post(
        "/api/generate", json={"digit": 7, "count": 1, "w": 1.2, "steps": 5, "seed": 42}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    check_image(data["items"][0]["image"])


def test_compare(client):
    response = client.post(
        "/api/compare",
        json={"digit": 7, "count": 1, "ws": [0.0, 1.2], "steps": 5, "seed": 42},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_experiment(client):
    response = client.post(
        "/api/experiment",
        json={
            "experiment_type": "guidance",
            "digit": 7,
            "count": 1,
            "values": [0.0, 1.2],
            "steps": 5,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_interpolate(client):
    response = client.post(
        "/api/interpolate",
        json={
            "digit": 7,
            "w": 1.2,
            "steps": 5,
            "seed_a": 42,
            "seed_b": 123,
            "frames": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["frames"]) == 3
    check_image(data["frames"][0]["image"])


def test_visualize(client):
    response = client.post(
        "/api/visualize",
        json={"digit": 7, "w": 1.2, "steps": 5, "seed": 42, "snapshot_count": 4},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["snapshots"]) >= 4
    check_image(data["snapshots"][0]["image"])


def test_representation(client):
    response = client.post(
        "/api/representation",
        json={"digit": 7, "timestep": 200, "steps": 50, "w": 1.2, "seed": 42},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["channels"] == 256
    assert len(data["heatmaps"]) == 256
    assert len(data["feature_shape"]) == 3
    check_image(data["heatmaps"][0]["image"])


def test_representation_evolution(client):
    response = client.post(
        "/api/representation/evolution",
        json={
            "digit": 7,
            "timesteps": [350, 252, 154, 48],
            "steps": 50,
            "w": 1.2,
            "seed": 42,
            "channel": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["channel"] == 5
    assert len(data["evolution"]) == 4
    check_image(data["evolution"][0]["image"])


def test_evaluate(client):
    response = client.post(
        "/api/evaluate", json={"digit": 7, "count": 2, "w": 1.2, "steps": 5, "seed": 42}
    )

    assert response.status_code == 200
    data = response.json()
    assert "class_consistency" in data
    assert "mean_confidence" in data
    assert len(data["items"]) == 2


def test_invalid_digit(client):
    response = client.post("/api/generate", json={"digit": 10, "count": 1, "steps": 5})

    assert response.status_code == 422
