"""Checks for the actual bundled source heads; no fitting or model downloads."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from leap import LinearReadout
from leap.cli import main

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


class PretrainedReadoutTests(unittest.TestCase):
    def test_frozen_heads_digests_and_raw_feature_predictions(self):
        manifest = json.loads((CHECKPOINTS / "metadata.json").read_text())
        self.assertEqual(manifest["default_head"], "qwen3-8b-lmsys")
        self.assertEqual(len(manifest["heads"]), 10)
        expected_main = {
            "qwen3-8b-lmsys": ([30], 4096),
            "llama-3.1-8b-instruct-lmsys": ([30, 32], 8192),
            "mistral-7b-instruct-v0.3-lmsys": ([28], 4096),
        }
        for key, (layers, width) in expected_main.items():
            self.assertEqual(manifest["heads"][key]["feature_layers"], layers)
            self.assertEqual(manifest["heads"][key]["feature_dim"], width)
        rng = np.random.default_rng(0)
        for head_id, info in manifest["heads"].items():
            with self.subTest(head=head_id):
                x = rng.normal(size=(8, info["feature_dim"])).astype(np.float16).astype(np.float32)
                path = CHECKPOINTS / info["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info["sha256"])
                self.assertEqual(path.stat().st_size, info["bytes"])
                head = LinearReadout.load(path)
                self.assertEqual(head.metadata["source_domain"], info["source_domain"])
                actual_layers = head.metadata.get("feature_layers", [head.metadata.get("one_based_layer")])
                self.assertEqual(actual_layers, info["feature_layers"])
                self.assertEqual(head.mapping, "sqrt")
                self.assertFalse(head.metadata["test_used_for_selection"])
                self.assertEqual(head.metadata["model_revision"], info["model_revision"])
                with np.load(path, allow_pickle=False) as z:
                    self.assertEqual(z["w"].dtype, np.dtype(info["exported_parameter_dtype"]))
                    self.assertEqual(z["b"].dtype, z["w"].dtype)
                    dtype = "<f4" if z["w"].dtype == np.float32 else "<f8"
                    params = z["w"].astype(dtype).tobytes() + z["b"].astype(dtype).tobytes()
                    self.assertEqual(hashlib.sha256(params).hexdigest(), info["coefficient_sha256"])
                    raw_coordinate = x.astype(np.float64) @ z["w"].astype(np.float64) + float(z["b"])
                expected = np.clip(np.clip(raw_coordinate, 1, np.sqrt(2048)) ** 2, 1, 2048)
                np.testing.assert_array_equal(head.predict(x), expected)

    def test_bundled_head_evaluates_through_public_cli_without_refitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.loads((CHECKPOINTS / "metadata.json").read_text())
            for head_id, info in manifest["heads"].items():
                with self.subTest(head=head_id):
                    checkpoint = CHECKPOINTS / info["file"]
                    data, output = root / (head_id + ".npz"), root / (head_id + ".json")
                    x = np.zeros((4, info["feature_dim"]), dtype=np.float32)
                    np.savez(data, test_x=x, test_y=np.array([64, 128, 256, 512]))
                    main(["evaluate", "--head", str(checkpoint), "--data", str(data),
                          "--split", "test", "--output", str(output)])
                    result = json.loads(output.read_text())
                    self.assertEqual(result["metrics"]["n"], 4)
                    self.assertFalse(result["refit_or_selection"])
                    self.assertEqual(result["head_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_scale_control_is_distinct_from_default_qwen_head(self):
        manifest = json.loads((CHECKPOINTS / "metadata.json").read_text())
        primary = manifest["heads"]["qwen3-8b-lmsys"]
        scale = manifest["heads"]["qwen3-8b-lmsys-round0"]
        self.assertEqual(primary["protocol_group"], "default_readout")
        self.assertEqual(scale["training_label"], "observed round0 only")
        self.assertNotEqual(primary["coefficient_sha256"], scale["coefficient_sha256"])
        self.assertEqual(primary["exported_parameter_dtype"], "float32")
        self.assertEqual(scale["exported_parameter_dtype"], "float64")
        head = LinearReadout.load(CHECKPOINTS / "llama-3.1-8b-instruct-lmsys/head.npz")
        with self.assertRaises(ValueError):
            head.predict(np.zeros((1, 4096)))


if __name__ == "__main__":
    unittest.main()
