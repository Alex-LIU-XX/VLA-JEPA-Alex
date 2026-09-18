"""VLA-JEPA 真机推理服务（Piper 控制端 `vla_infer` 协议）。

在 GPU 机器上以 **VLA 环境**运行，控制端（机器人 PC）用现成的 `VlaZmqClient` 连上来即可：

    # 真机（需要权重）
    python examples/real-robot/piper_zmq_server.py \
        --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt \
        --port 5555 --use-bf16

    # 无模型自检（验证协议/客户端链路，不需要 GPU 与权重）
    python examples/real-robot/piper_zmq_server.py --dry-run --port 5555

协议与实现依据：`doc/robot/piper_server_plan.md`（§1 协议、§5 设计、§5.5 安全策略）。

⚠️ 控制端必须配 `state_type=joint` + `absolute_action=True`（本项目默认，见
`doc/robot/deployment.md` §4.2）；本服务输出**绝对关节角 + 夹爪**。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import typing as t

import numpy as np

# 允许 `python examples/real-robot/piper_zmq_server.py` 直接运行（同目录模块可 import）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy import DryRunPolicy, VLAJepaPiperPolicy
from transport import ZmqRepServer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLA-JEPA policy server speaking the vla_infer (ZMQ REQ/REP + msgpack) protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="run 目录下的 .pt（同目录需有 config.yaml 与 dataset_statistics.json）")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--jpeg-quality", type=int, default=80, help="回包图像压缩质量（与客户端一致）")
    parser.add_argument("--cuda", default="0", help="GPU 序号，传给 cuda:<n>")
    parser.add_argument("--use-bf16", action="store_true", help="模型转 bfloat16")
    parser.add_argument("--num-inference-timesteps", type=int, default=0,
                        help="flow-matching 去噪步数；0 = 用权重自带的 config 值")
    parser.add_argument("--chunk-steps", type=int, default=0,
                        help="返回的动作块长度；0 = 用模型 chunk 长度（future_action_window_size+1）")
    parser.add_argument("--unnorm-key", type=str, default="",
                        help="dataset_statistics.json 的顶层键；权重只有一份统计量时可留空")
    parser.add_argument("--default-instruction", type=str, default="",
                        help="请求里没有 cmd/instruction 时使用；建议填该任务在 tasks.jsonl 里的原文")
    parser.add_argument("--no-binarize-gripper", action="store_true",
                        help="关闭第 6 维（夹爪）0/1 二值化，并按 Piper min/max 统计量输出连续物理值")
    parser.add_argument("--dry-run", action="store_true",
                        help="不载入模型，返回保持位姿的动作块（协议自检用）")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--max-requests", type=int, default=0, help="处理 N 个请求后退出；0 = 一直服务")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def build_policy(args: argparse.Namespace) -> t.Any:
    if args.dry_run:
        logging.warning("dry-run 模式：不载入模型，返回保持位姿的动作块（仅供协议自检）")
        return DryRunPolicy(
            chunk_steps=args.chunk_steps or 7,
        )
    if not args.ckpt_path:
        raise SystemExit("必须提供 --ckpt_path（或使用 --dry-run）")
    return VLAJepaPiperPolicy(
        ckpt_path=args.ckpt_path,
        device=f"cuda:{args.cuda}",
        use_bf16=args.use_bf16,
        num_inference_timesteps=args.num_inference_timesteps,
        chunk_steps=args.chunk_steps,
        unnorm_key=args.unnorm_key,
        default_instruction=args.default_instruction,
        binarize_gripper=not args.no_binarize_gripper,
    )


def main(argv: t.Optional[t.Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    policy = build_policy(args)
    policy.load()          # DryRunPolicy.load 是 no-op
    if args.warmup:
        policy.warmup()

    state: t.Dict[str, t.Any] = {"step": 0, "last_action": None, "last_cmd": None}

    def handle(obs: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        started = time.time()
        response = policy.predict_request(obs)
        action = np.asarray(response["action"])
        state["step"] += 1
        state["last_action"] = action
        state["last_cmd"] = obs.get("cmd", obs.get("instruction"))
        logging.info(
            "step=%d cmd=%r state=%s latency=%.0fms action=%s first=%s last=%s",
            state["step"],
            state["last_cmd"],
            _summarize(obs.get("state")),
            (time.time() - started) * 1000.0,
            action.shape,
            np.round(action[0], 3).tolist() if action.size else [],
            np.round(action[-1], 3).tolist() if action.size else [],
        )
        return response

    def fallback(obs: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """异常兜底：优先回上一块可用动作，否则回保持位姿（避免客户端 2 s 超时急停）。"""
        action = state["last_action"]
        if action is None:
            action = policy.make_hold_action(obs)
            logging.warning("no previous action, returning hold-position chunk")
        else:
            logging.warning("returning last successful action chunk as fallback")
        return {"action": np.ascontiguousarray(action, dtype=np.float32)}

    logging.info("client must be configured with state_type=joint + absolute_action=True (absolute joint targets)")
    server = ZmqRepServer(
        handler=handle,
        host=args.host,
        port=args.port,
        jpeg_quality=args.jpeg_quality,
        fallback=fallback,
        max_requests=args.max_requests,
    )
    served = server.serve_forever()
    logging.info("server exited after %d request(s)", served)
    return 0


def _summarize(value: t.Any) -> t.Any:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.round(arr, 3).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
