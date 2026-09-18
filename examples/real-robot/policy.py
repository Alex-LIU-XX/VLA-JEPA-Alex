"""VLA-JEPA × Piper 策略适配层：把 ZMQ 请求（观测字典）映射成动作块。

请求契约（控制端 `vla_infer` 决定，见 `doc/robot/piper_server_plan.md` §1.4）：

    {
      "image":       np.ndarray(H,W,3) uint8,   # 头部相机（客户端已 resize 到 224）
      "wrist_image": np.ndarray(H,W,3) uint8,   # 腕部相机
      "state":       np.ndarray(7,) float32,    # 6 关节角 + 夹爪，**未归一化**
      "cmd":         str,                       # 任务指令（兼容 instruction / default_instruction）
      "return_action_chunk": bool,              # 可选，False 时只回一帧
    }

响应契约：{"action": np.ndarray(T, 7) float32}，T = `future_action_window_size + 1`（Piper 配置为 7）。

动作语义：本 server 输出**绝对关节角 + 夹爪**（与 Piper 训练配置 `action_type: absolute` 一致），
控制端必须配 `state_type=joint` + `absolute_action=True` 才会原样下发，详见
`doc/robot/deployment.md` §4.2。

模型侧契约（载入、反归一化）见 `doc/robot/model.md` §2、§5。
"""

from __future__ import annotations

import logging
import time
import typing as t

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DIM = 7  # Piper: 6 关节 + 夹爪


# --------------------------------------------------------------------------- #
# 观测校验与转换
# --------------------------------------------------------------------------- #
def to_pil_image(name: str, value: t.Any) -> Image.Image:
    """把协议里的 HWC3 图像（uint8/float∈[0,1] / PIL）统一成 PIL.Image。"""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be np.ndarray or PIL.Image, got {type(value)}")
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be HxWx3 RGB, got shape={value.shape}")
    arr = value
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)  # 负 stride / 非连续时复制
    return Image.fromarray(arr, mode="RGB")


def get_instruction(obs: t.Dict[str, t.Any], default: str = "") -> str:
    """兼容 cmd / instruction / default_instruction（代码以 cmd 为准，文档曾写 instruction）。"""
    for key in ("cmd", "instruction", "task_instruction"):
        value = obs.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def unnormalize_actions(normalized: np.ndarray, stats: t.Dict[str, t.Any], binarize_gripper: bool = True) -> np.ndarray:
    """归一化动作 → 物理动作。

    统计量键名有两套（见 `doc/robot/model.md` §5.2），这里按实际存在的键自动选择：
      * q01/q99（框架 base_framework.unnormalize_actions 用的形式）
      * min/max（LIBERO 客户端用的形式）
    `mask` 为 False 的维度保持归一化值不变（gripper 维通常为 False）。
    Piper 的数据配置仍对 gripper 使用 `min_max`；因此关闭二值化时，显式对
    gripper 做 min/max 反归一化，避免把归一化值直接当作物理动作下发。
    """
    if "q99" in stats and "q01" in stats:
        hi, lo = np.asarray(stats["q99"], dtype=np.float64), np.asarray(stats["q01"], dtype=np.float64)
        probe = hi
    elif "max" in stats and "min" in stats:
        hi, lo = np.asarray(stats["max"], dtype=np.float64), np.asarray(stats["min"], dtype=np.float64)
        probe = hi
    else:
        raise KeyError(f"dataset_statistics.action 里既没有 q01/q99 也没有 min/max，实际键：{sorted(stats)}")

    mask = np.asarray(stats.get("mask", np.ones_like(probe, dtype=bool)), dtype=bool).copy()
    actions = np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)
    if binarize_gripper and actions.shape[-1] > 6:
        # 与框架/仿真客户端一致：第 6 维（夹爪）按 0.5 阈值二值化
        actions[:, 6] = np.where(actions[:, 6] < 0.5, 0.0, 1.0)
    elif actions.shape[-1] > 6 and mask.shape[0] > 6:
        # Piper data_config.py uses min_max for action.gripper even though the
        # generic action mask marks gripper as a non-scaled command dimension.
        mask[6] = True
    return np.where(mask, 0.5 * (actions + 1.0) * (hi - lo) + lo, actions)


# --------------------------------------------------------------------------- #
# 策略实现
# --------------------------------------------------------------------------- #
class DryRunPolicy:
    """无模型自检策略：返回"保持当前位姿"的动作块，用于验证协议与客户端链路。"""

    def __init__(self, chunk_steps: int = 7, state_dim: int = DEFAULT_CHUNK_DIM) -> None:
        self.chunk_steps = chunk_steps
        self.state_dim = state_dim

    def load(self) -> None:
        """与 VLAJepaPiperPolicy 保持同一接口（dry-run 无需载入任何东西）。"""
        logger.info("dry-run policy ready | chunk_steps=%d state_dim=%d", self.chunk_steps, self.state_dim)

    def warmup(self) -> None:
        return None

    def make_hold_action(self, obs: t.Dict[str, t.Any]) -> np.ndarray:
        chunk = np.zeros((self.chunk_steps, self.state_dim), dtype=np.float32)
        state = obs.get("state")
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.shape[0] == self.state_dim:
                chunk[:] = flat
        return chunk

    def predict_request(self, obs: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        # 仍然走一遍图像/指令解析，保证协议字段被真正校验
        to_pil_image("image", obs.get("image"))
        to_pil_image("wrist_image", obs.get("wrist_image"))
        get_instruction(obs)
        chunk = self.make_hold_action(obs)
        if obs.get("return_action_chunk", True) is False:
            chunk = chunk[:1]
        return {"action": chunk}


class VLAJepaPiperPolicy:
    """真实策略：载入 VLA-JEPA 权重，把观测映射成绝对关节动作块。"""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        use_bf16: bool = False,
        num_inference_timesteps: int = 0,
        chunk_steps: int = 0,
        unnorm_key: str = "",
        default_instruction: str = "",
        binarize_gripper: bool = True,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.use_bf16 = use_bf16
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.chunk_steps = int(chunk_steps)  # 0 = 用模型自身 chunk 长度
        self.unnorm_key = unnorm_key
        self.default_instruction = default_instruction
        self.binarize_gripper = binarize_gripper

        self.model: t.Any = None
        self.state_dim: int = DEFAULT_CHUNK_DIM
        self.action_stats: t.Dict[str, t.Any] = {}

    # -------------------------------------------------------------- load #
    def load(self) -> None:
        import torch  # 延迟导入：dry-run 模式不需要 torch/starVLA

        from starVLA.model.framework.base_framework import baseframework
        from starVLA.model.tools import read_mode_config

        started = time.time()
        model = baseframework.from_pretrained(self.ckpt_path)  # 见 doc/robot/model.md §2.2
        if self.use_bf16:
            model = model.to(torch.bfloat16)
        self.model = model.to(torch.device(self.device)).eval()

        _, norm_stats = read_mode_config(self.ckpt_path)  # 读 run 目录的 dataset_statistics.json
        key = self.unnorm_key or (next(iter(norm_stats)) if len(norm_stats) == 1 else "")
        if not key:
            raise ValueError(
                f"权重含多个数据集统计量 {sorted(norm_stats)}，必须用 --unnorm-key 指定其中一个"
            )
        if key not in norm_stats:
            raise KeyError(f"unnorm_key='{key}' 不在 {sorted(norm_stats)} 中")
        self.action_stats = norm_stats[key]["action"]

        cfg = self.model.config.framework.action_model
        self.state_dim = int(getattr(cfg, "state_dim", DEFAULT_CHUNK_DIM) or DEFAULT_CHUNK_DIM)
        chunk_len = int(getattr(cfg, "future_action_window_size", self.chunk_steps or 6)) + 1
        self.chunk_steps = self.chunk_steps or chunk_len

        if self.num_inference_timesteps > 0:
            # 唯一有效的去噪步数开关（传 num_ddim_steps 会被 predict_action 的 **kwargs 吞掉）
            self.model.action_model.num_inference_timesteps = self.num_inference_timesteps

        logger.info(
            "model loaded in %.1fs | device=%s bf16=%s | state_dim=%d chunk_steps=%d "
            "num_inference_timesteps=%d | action stats keys=%s",
            time.time() - started,
            self.device,
            self.use_bf16,
            self.state_dim,
            self.chunk_steps,
            getattr(self.model.action_model, "num_inference_timesteps", -1),
            sorted(self.action_stats),
        )

    def warmup(self) -> None:
        """预热：避免真机第一帧因为 CUDA 初始化超时（客户端 2 s 超时即急停）。"""
        obs = {
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros((1, self.state_dim), dtype=np.float32),
            "cmd": self.default_instruction,
        }
        started = time.time()
        self.predict_request(obs)
        logger.info("warmup done in %.1fs", time.time() - started)

    # ----------------------------------------------------------- predict #
    def make_hold_action(self, obs: t.Dict[str, t.Any]) -> np.ndarray:
        """异常兜底：保持当前位姿（无 state 时回零），避免客户端等超时。"""
        chunk = np.zeros((self.chunk_steps, DEFAULT_CHUNK_DIM), dtype=np.float32)
        state = obs.get("state")
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.shape[0] == chunk.shape[1]:
                chunk[:] = flat
        return chunk

    def predict_request(self, obs: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if self.model is None:
            raise RuntimeError("policy 未载入，请先调用 load()")

        head = to_pil_image("image", obs.get("image"))
        wrist = to_pil_image("wrist_image", obs.get("wrist_image"))
        instruction = get_instruction(obs, self.default_instruction)

        state_in = obs.get("state")
        state_arg = None
        if state_in is not None:
            flat = np.asarray(state_in, dtype=np.float32).reshape(-1)
            if flat.shape[0] != self.state_dim:
                raise ValueError(
                    f"state 维度不匹配：收到 {flat.shape[0]}，期望 {self.state_dim}"
                    "（控制端请确认 state_type=joint，见 doc/robot/deployment.md §4.2）"
                )
            state_arg = [flat.reshape(1, -1)]  # → np.array 后为 (B=1, 1, state_dim)

        out = self.model.predict_action(
            batch_images=[[head, wrist]],  # 顺序固定：头相机在前、腕相机在后
            instructions=[instruction],
            state=state_arg,
        )
        normalized = np.asarray(out["normalized_actions"], dtype=np.float64)  # (B, T, D)
        if normalized.ndim != 3 or normalized.shape[0] < 1:
            raise ValueError(f"normalized_actions 形状异常：{normalized.shape}")

        actions = unnormalize_actions(normalized[0], self.action_stats, self.binarize_gripper)
        if actions.shape[1] != DEFAULT_CHUNK_DIM:
            raise ValueError(
                f"动作维度为 {actions.shape[1]}，控制端 Piper 需要 7 维（6 关节 + 夹爪）"
            )
        if self.chunk_steps > 0:
            actions = actions[: self.chunk_steps]
        if obs.get("return_action_chunk", True) is False:
            actions = actions[:1]
        return {"action": np.ascontiguousarray(actions, dtype=np.float32)}


__all__ = [
    "DryRunPolicy",
    "VLAJepaPiperPolicy",
    "get_instruction",
    "to_pil_image",
    "unnormalize_actions",
]
