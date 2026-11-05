import json
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import server_windows


class FakeModel:
    def tokenize(self, text, add_bos=True):
        return text.decode("utf-8").split()

    def create_chat_completion(self, **_kwargs):
        yield {"choices": [{"delta": {"content": "Hello"}}]}
        yield {"choices": [{"delta": {"content": " from Windows."}}]}


class WindowsBackendTests(unittest.TestCase):
    def setUp(self):
        self.previous_model = server_windows.model
        server_windows.model = FakeModel()

    def tearDown(self):
        server_windows.model = self.previous_model

    def test_model_registry_exposes_only_the_windows_model(self):
        result = server_windows.get_models()
        self.assertEqual([server_windows.MODEL_ID], [m["id"] for m in result["models"]])
        self.assertEqual(server_windows.MODEL_ID, result["active"])

    def test_stream_returns_text_and_final_stats(self):
        request = server_windows.ChatRequest(
            messages=[server_windows.Message(role="user", content="Say hello")],
            max_tokens=64,
        )
        events = list(server_windows._generate(request))
        self.assertEqual("Hello", events[0]["delta"])
        self.assertEqual(" from Windows.", events[1]["delta"])
        self.assertTrue(events[-1]["done"])
        self.assertEqual("Hello from Windows.", events[-1]["content"])
        self.assertGreater(events[-1]["tokens"], 0)
        self.assertEqual(server_windows.MODEL_NAME, events[-1]["model"])

    def test_select_model_rejects_models_not_supported_by_windows(self):
        with self.assertRaises(server_windows.HTTPException) as error:
            server_windows.select_model(
                server_windows.ModelRequest(model="stablecode")
            )
        self.assertEqual(400, error.exception.status_code)

    def test_chat_stream_endpoint_emits_sse_and_marks_windows_limitations(self):
        model_ready = threading.Event()

        def fake_load_model():
            server_windows.model = FakeModel()
            server_windows.model_status = "ready"
            model_ready.set()

        with patch.object(server_windows, "load_model", fake_load_model):
            with TestClient(server_windows.app) as client:
                self.assertTrue(model_ready.wait(2))
                response = client.post(
                    "/chat/stream",
                    json={
                        "messages": [{"role": "user", "content": "Say hello"}],
                        "max_tokens": 64,
                    },
                )
                self.assertEqual(200, response.status_code)
                events = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                self.assertEqual("Hello", events[0]["delta"])
                self.assertTrue(events[-1]["done"])
                self.assertIn("CPU inference", events[-1]["note"])

                unavailable = client.post(
                    "/chat/agent/stream",
                    json={
                        "messages": [{"role": "user", "content": "Research this"}]
                    },
                )
                self.assertEqual(501, unavailable.status_code)


if __name__ == "__main__":
    unittest.main()
