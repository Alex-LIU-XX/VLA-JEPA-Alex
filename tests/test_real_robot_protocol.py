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

import logging  # noqa: E402

import piper_zmq_server  # noqa: E402
import policy as policy_module  # noqa: E402
import transport as transport_module  # noqa: E402
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

    def test_return_action_chunk_is_coerced_like_reference_servers(self) -> None:
        # 控制端参考实现用 bool(...) 强制转换；int / np.bool_ 都必须被接受，而不是报错。
        policy = DryRunPolicy()
        obs = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.arange(7, dtype=np.float32),
        }
        self.assertEqual(policy.predict_request(obs)["action"].shape, (7, 7))
        self.assertEqual(policy.predict_request({**obs, "return_action_chunk": False})["action"].shape, (1, 7))
        self.assertEqual(policy.predict_request({**obs, "return_action_chunk": 0})["action"].shape, (1, 7))
        self.assertEqual(policy.predict_request({**obs, "return_action_chunk": np.True_})["action"].shape, (7, 7))


class LoggingTests(unittest.TestCase):
    def test_reassert_module_logging_reenables_starvla_disabled_loggers(self) -> None:
        # starVLA 的 overwatch 会 dictConfig(disable_existing_loggers=True)，禁用模块 logger。
        # reassert_module_logging 必须把它们重新启用，否则启动/安全日志被静默丢弃。
        policy_module.logger.disabled = True
        transport_module.logger.disabled = True
        try:
            piper_zmq_server.reassert_module_logging("INFO")
            self.assertFalse(policy_module.logger.disabled)
            self.assertFalse(transport_module.logger.disabled)
            self.assertEqual(policy_module.logger.level, logging.INFO)
        finally:
            pass


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

    def test_real_handler_holds_on_failure_and_errors_without_state(self) -> None:
        """真实 piper_zmq_server handler/fallback：推理失败→保持位姿；无 state→error。"""
        port = _free_port()
        policy = DryRunPolicy()
        policy.load()
        state: dict = {"step": 0, "last_cmd": None}
        server = ZmqRepServer(
            piper_zmq_server.build_handler(policy, state),
            host="127.0.0.1",
            port=port,
            fallback=piper_zmq_server.build_fallback(policy, state),
            max_requests=3,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        joints = np.arange(7, dtype=np.float32)
        try:
            # 1) 正常请求：返回 (7,7)，且 step 递增
            sock.send(pack_payload({"image": image, "wrist_image": image, "state": joints, "cmd": "0"}))
            ok = unpack_payload(sock.recv())
            self.assertEqual(ok["action"].shape, (7, 7))
            np.testing.assert_allclose(ok["action"][0], joints)
            self.assertEqual(state["step"], 1)

            # 2) 缺图像但 state 合法：handler 抛错 → fallback 用当前 state 保持位姿
            sock.send(pack_payload({"state": joints, "cmd": "0"}))
            held = unpack_payload(sock.recv())
            self.assertNotIn("error", held)
            np.testing.assert_allclose(held["action"], np.repeat(joints[None, :], 7, axis=0))
            self.assertEqual(state["step"], 1)  # 失败请求不推进 step

            # 3) 缺 state：fallback 也无法安全保持 → 明确 error，不伪造动作
            sock.send(pack_payload({"image": image, "wrist_image": image, "cmd": "0"}))
            errored = unpack_payload(sock.recv())
            self.assertIn("error", errored)
            self.assertNotIn("action", errored)
        finally:
            sock.close()
            ctx.term()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
