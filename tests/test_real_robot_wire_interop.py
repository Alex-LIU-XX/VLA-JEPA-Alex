"""Bidirectional wire-format interop with the real control-side ``VLAProtocol``.

The control repo (``Double_Piper_Teleop/vla_infer``) pins ``numpy>=2.0`` while this repo pins
``1.26.4``, so it cannot be imported as a package here. However
``vla_infer/src/zmq/protocol.py`` only depends on numpy/msgpack/msgpack_numpy/PIL, so we load
that single module by path and cross-check the exact bytes in both directions.

Set ``VLA_INFER_PROTOCOL`` to override the protocol path. Tests are skipped when the control
repo is not present, so this file is safe to run in a standalone checkout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REAL_ROBOT_DIR = Path(__file__).resolve().parents[1] / "examples" / "real-robot"
sys.path.insert(0, str(REAL_ROBOT_DIR))

from wire import pack_payload, unpack_payload  # noqa: E402

DEFAULT_PROTOCOL_PATH = Path(
    "/home/liuxx/repo/Double_Piper_Teleop/vla_infer/src/zmq/protocol.py"
)


def _resolve_protocol_path() -> Path | None:
    candidates = []
    override = os.environ.get("VLA_INFER_PROTOCOL")
    if override:
        candidates.append(Path(override))
    candidates.append(DEFAULT_PROTOCOL_PATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_control_protocol():
    path = _resolve_protocol_path()
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("vla_infer_protocol_under_test", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VLAProtocol


VLAProtocol = _load_control_protocol()


def _client_observation() -> dict:
    rng = np.random.default_rng(0)
    return {
        "image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        # float64 on purpose: both sides must downgrade it to float32 (protocol rule c).
        "state": np.linspace(0.0, 1.0, 7, dtype=np.float64),
        "cmd": "0",
        "return_action_chunk": True,
    }


@unittest.skipIf(VLAProtocol is None, "control-side VLAProtocol not available")
class ControlProtocolInteropTests(unittest.TestCase):
    def test_client_pack_to_server_unpack(self) -> None:
        observation = _client_observation()
        packed = VLAProtocol.pack_payload(observation, jpeg_quality=80)
        decoded = unpack_payload(packed)

        self.assertEqual(decoded["cmd"], observation["cmd"])
        self.assertEqual(decoded["return_action_chunk"], observation["return_action_chunk"])
        self.assertEqual(decoded["state"].dtype, np.float32)
        self.assertEqual(decoded["state"].shape, (7,))
        np.testing.assert_allclose(decoded["state"], observation["state"], atol=1e-5)
        for key in ("image", "wrist_image"):
            self.assertIsInstance(decoded[key], np.ndarray)
            self.assertEqual(decoded[key].shape, (224, 224, 3))
            self.assertEqual(decoded[key].dtype, np.uint8)

    def test_server_pack_to_client_unpack(self) -> None:
        action = np.arange(49, dtype=np.float32).reshape(7, 7) / 10.0
        packed = pack_payload({"action": action}, jpeg_quality=80)
        decoded = VLAProtocol.unpack_payload(packed)

        self.assertIsInstance(decoded["action"], np.ndarray)
        self.assertEqual(decoded["action"].dtype, np.float32)
        self.assertEqual(decoded["action"].shape, (7, 7))
        np.testing.assert_allclose(decoded["action"], action, atol=1e-6)

    def test_non_image_float_arrays_stay_arrays(self) -> None:
        action = np.zeros((7, 7), dtype=np.float32)
        decoded = unpack_payload(pack_payload({"action": action}))
        self.assertIsInstance(decoded["action"], np.ndarray)
        self.assertEqual(decoded["action"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
