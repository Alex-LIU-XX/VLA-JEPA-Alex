"""协议自检客户端：不接机器人，直接按 `vla_infer` 的线格式向 server 发请求并校验回包。

用途：在没有机械臂、没有 GPU 的情况下验证 wire/transport/policy 三段是否自洽：
    # 终端 1
    python examples/real-robot/piper_zmq_server.py --dry-run --port 15555
    # 终端 2
    python examples/real-robot/selftest_client.py --port 15555

它等价于控制端 `VlaZmqClient.get_response()` 的收发方式（见
`vla_infer/src/zmq/zmq_client.py:33-60`），因此通过本自检即说明协议层没问题；
真机侧仍需用 `vla_infer` 的客户端做一次实机联调。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import typing as t

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wire import pack_payload, unpack_payload  # noqa: E402


def build_observation(args: argparse.Namespace) -> t.Dict[str, t.Any]:
    if args.image and os.path.isfile(args.image):
        from PIL import Image

        head = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        head = np.ascontiguousarray(_resize(head, args.image_size))
    else:
        head = np.random.randint(0, 256, (args.image_size, args.image_size, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 256, (args.image_size, args.image_size, 3), dtype=np.uint8)
    state = np.linspace(0.0, 0.6, args.state_dim, dtype=np.float32)
    return {
        "image": head,
        "wrist_image": wrist,
        "state": state,
        "cmd": args.cmd,
    }


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((size, size)), dtype=np.uint8)


def main(argv: t.Optional[t.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ZMQ protocol self-test client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image", type=str, default="", help="可选：真实图片路径（默认随机图）")
    parser.add_argument("--cmd", type=str, default="0", help="建议填该任务在 tasks.jsonl 里的原文")
    parser.add_argument("--send-bad-payload", action="store_true",
                        help="先发一个缺 batch 字段的坏 payload，验证 server 的兜底与存活")
    args = parser.parse_args(argv)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.connect(f"tcp://{args.host}:{args.port}")

    if args.send_bad_payload:
        print("[1/2] 发送坏 payload（只有 cmd，缺图像/状态）…")
        sock.send(pack_payload({"cmd": args.cmd}, jpeg_quality=args.jpeg_quality))
        try:
            reply = unpack_payload(sock.recv())
            print("      server 依然回包 →", _describe(reply))
        except zmq.error.Again:
            print("      ❌ server 超时未回包（兜底逻辑失效）")
            return 1

    ok = True
    for step in range(args.steps):
        obs = build_observation(args)
        started = time.time()
        sock.send(pack_payload(obs, jpeg_quality=args.jpeg_quality))
        try:
            reply = unpack_payload(sock.recv())
        except zmq.error.Again:
            print(f"step {step}: ❌ 超时（>{args.timeout_ms} ms）—— 真机上会触发急停")
            return 1
        latency_ms = (time.time() - started) * 1000.0
        action = reply.get("action")
        print(f"step {step}: latency={latency_ms:.0f}ms resp_keys={sorted(reply)} action={_describe(action)}")
        if not isinstance(action, np.ndarray):
            print(f"      ❌ action 不是 np.ndarray：{type(action)}")
            ok = False
            continue
        if action.dtype != np.float32:
            print(f"      ⚠️ action dtype={action.dtype}（客户端会用 float32 处理，通常无碍）")
        if action.ndim != 2 or action.shape[1] != args.state_dim:
            print(f"      ❌ action 形状应为 (T, {args.state_dim})，实际 {action.shape}")
            ok = False
        elif np.any(~np.isfinite(action)):
            print("      ❌ action 含 NaN/Inf")
            ok = False

    sock.close()
    ctx.term()
    print("✅ 协议自检通过" if ok else "❌ 协议自检失败")
    return 0 if ok else 1


def _describe(value: t.Any) -> str:
    if isinstance(value, np.ndarray):
        head = np.round(value[0], 3).tolist() if value.size else []
        return f"ndarray{value.shape} dtype={value.dtype} first={head}"
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
