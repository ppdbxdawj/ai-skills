import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).with_name("generate.py")
SPEC = importlib.util.spec_from_file_location("seedream_generate", MODULE_PATH)
generate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate)


class AtlasPayloadTests(unittest.TestCase):
    def test_text_to_image_keeps_single_image_default(self):
        payload = generate.build_atlas_payload(
            prompt="a cat",
            image_urls=None,
            width=1024,
            height=1536,
            force_single=True,
            max_images=4,
        )

        self.assertEqual(payload["model"], "bytedance/seedream-v4")
        self.assertEqual(payload["size"], "1024*1536")
        self.assertNotIn("max_images", payload)

    def test_edit_sequence_selects_matching_model(self):
        payload = generate.build_atlas_payload(
            prompt="change the background",
            image_urls=["https://example.com/source.png"],
            width=None,
            height=None,
            force_single=False,
            max_images=3,
        )

        self.assertEqual(payload["model"], "bytedance/seedream-v4/edit-sequential")
        self.assertEqual(payload["images"], ["https://example.com/source.png"])
        self.assertEqual(payload["max_images"], 3)


class AtlasRequestTests(unittest.TestCase):
    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"})
    @patch("requests.post")
    def test_submit_sends_one_request(self, post):
        response = Mock()
        response.json.return_value = {"code": 200, "data": {"id": "prediction-123"}}
        post.return_value = response

        task_id = generate.submit_atlas_task({"model": "model", "prompt": "prompt"})

        self.assertEqual(task_id, "prediction-123")
        post.assert_called_once()

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"})
    @patch("requests.get")
    def test_poll_normalizes_completed_outputs(self, get):
        response = Mock()
        response.json.return_value = {
            "code": 200,
            "data": {"status": "completed", "outputs": ["https://example.com/image.png"]},
        }
        get.return_value = response

        result = generate.poll_atlas_until_done("prediction-123", poll_interval=0)

        self.assertEqual(result, {"data": {"image_urls": ["https://example.com/image.png"]}})
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
