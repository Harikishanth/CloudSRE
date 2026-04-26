import traceback
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

try:
    response = client.post("/reset", json={"task_id": "cascade"})
    if response.status_code != 200:
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    else:
        print("Success:", response.json())
except Exception as e:
    traceback.print_exc()
