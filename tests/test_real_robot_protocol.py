"""Focused regression tests for the Piper ZMQ wire and safety boundaries."""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import msgpack
import numpy as np
import zmq

REAL_ROBOT_DIR = Path(__file__).resolve().parents[1] / "examples" / "real-robot"
sys.path.insert(0, str(REAL_ROBOT_DIR))

from policy import DryRunPolicy, to_pil_image, unnormalize_actions  # noqa: E402
from transport import ZmqRepServer  # noqa: E402
from wire import pack_payload, unpack_payload  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class WireTests(unittest.TestCase):
    def test_image_round_trip_and_payload_limits(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        decoded = unpack_payload(pack_payload({"image": image}))
        self.assertEqual(decoded["image"].shape, image.shape)
        self.assertEqual(decoded["image"].dtype, np.uint8)

        with self.assertRaises(ValueError):
            unpack_payload(msgpack.packb({"image": b"not-a-jpeg"}, use_bin_type=True))

    def test_non_mapping_payload_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            unpack_payload(msgpack.packb([1, 2, 3], use_bin_type=True))


class PolicyTests(unittest.TestCase):
    def test_images_and_state_are_strict(self) -> None:
        with self.assertRaises(TypeError):
            to_pil_image("image", np.zeros((4, 4, 3), dtype=np.int16))
        with self.assertRaises(ValueError):
            to_pil_image("image", np.full((4, 4, 3), np.nan, dtype=np.float32))

        policy = DryRunPolicy()
        obs = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.arange(7, dtype=np.float32),
        }
        action = policy.predict_request(obs)["action"]
        np.testing.assert_array_equal(action, np.repeat(obs["state"][None, :], 7, axis=0))
        with self.assertRaises(ValueError):
            policy.predict_request({**obs, "state": None})

    def test_action_statistics_and_shape_are_checked(self) -> None:
        stats = {"min": np.zeros(7), "max": np.ones(7), "mask": np.ones(7, dtype=bool)}
        result = unnormalize_actions(np.zeros((2, 7)), stats, binarize_gripper=False)
        np.testing.assert_allclose(result, 0.5)
        with self.assertRaises(ValueError):
            unnormalize_actions(np.zeros((7,)), stats)


class TransportTests(unittest.TestCase):
    def test_bad_frame_gets_reply_and_server_recovers(self) -> None:
        port = _free_port()

        def handler(request: dict) -> dict:
            if request == {"ok": True}:
                return {"action": np.ones((1, 7), dtype=np.float32)}
            raise ValueError("bad request")

        def fallback(_request: dict) -> dict:
            return {"action": np.zeros((1, 7), dtype=np.float32)}

        server = ZmqRepServer(handler, host="127.0.0.1", port=port, fallback=fallback, max_requests=2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        try:
            sock.send(b"corrupt")
            first = unpack_payload(sock.recv())
            self.assertIn("action", first)

            sock.send(pack_payload({"ok": True}))
            second = unpack_payload(sock.recv())
            np.testing.assert_array_equal(second["action"], np.ones((1, 7), dtype=np.float32))
        finally:
            sock.close()
            ctx.term()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
