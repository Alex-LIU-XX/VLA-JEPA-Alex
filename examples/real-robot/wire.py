"""Piper 控制端（vla_infer）ZMQ 线格式编解码。

与 `vla_infer/src/zmq/protocol.py` **逐条对齐**，两侧必须使用同一套规则才能互通：

1. import 时 `msgpack_numpy.patch()` 全局生效 —— 非图像的 numpy 数组走 msgpack-numpy 的
   ``b'nd'`` 扩展格式（dict: ``nd/type/kind/shape/data``）。
   ⚠️ 本仓库 `deployment/model_server/tools/msgpack_numpy.py` 用的是 ``b'__ndarray__'`` 格式，
   **与本协议不兼容**，不要在这里复用。
2. 打包（pack）：3 维且 uint8 的数组**一律**当图像，压成 JPEG 字节（与 key 名无关）；
   float64 一律降为 float32；其余原样交给 msgpack。
3. 解包（unpack）：只有 **key 名含 'img'/'image'** 的 bytes 才会被还原成 numpy 数组；
   其它 bytes 保持 bytes（所以服务端不要返回 key 名不含 image 的 3D uint8 数组）。

参考：`vla_infer/src/zmq/protocol.py:20,28-46,49-69,72-85`
"""

from __future__ import annotations

import io
import typing as t

import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image

# 与对端一致：启用 msgpack 的 numpy 支持（对 packb/unpackb 全局生效）
m.patch()

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8 * 1024 * 1024


def encode_image(image: np.ndarray, quality: int = 80) -> bytes:
    """numpy (H,W,3) uint8 → JPEG 字节。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 ndarray, got {getattr(image, 'shape', type(image))}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels: {image.shape[:2]}")
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
    with io.BytesIO() as buffer:
        Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def decode_image(image_bytes: bytes) -> np.ndarray:
    """JPEG 字节 → numpy 数组（与对端 decode_image 行为一致）。"""
    if len(image_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"encoded image exceeds {MAX_PAYLOAD_BYTES} bytes")
    with io.BytesIO(image_bytes) as buffer:
        try:
            img = Image.open(buffer)
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"decoded image exceeds {MAX_IMAGE_PIXELS} pixels: {(img.width, img.height)}")
            img.load()
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid encoded image") from exc
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def pack_payload(payload: t.Dict[str, t.Any], jpeg_quality: int = 80) -> bytes:
    """按数据类型自动优化的通用打包（对应 `VLAProtocol.pack_payload`）。"""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be dict, got {type(payload)}")
    processed: t.Dict[str, t.Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"payload keys must be str, got {type(key)}")
        if isinstance(value, np.ndarray):
            if value.ndim == 3 and value.dtype == np.uint8:
                processed[key] = encode_image(value, quality=jpeg_quality)
            elif value.dtype == np.float64:
                processed[key] = value.astype(np.float32)
            else:
                processed[key] = value
        else:
            processed[key] = value
    packed = msgpack.packb(processed, use_bin_type=True)
    if len(packed) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"packed payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    return packed


def unpack_payload(payload_bytes: bytes) -> t.Dict[str, t.Any]:
    """通用解包：把 key 名含 img/image 的 bytes 还原成 numpy 数组。"""
    if not isinstance(payload_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like, got {type(payload_bytes)}")
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    unpacked = msgpack.unpackb(payload_bytes, raw=False)
    if not isinstance(unpacked, dict):
        raise TypeError(f"decoded payload must be dict, got {type(unpacked)}")
    for key, value in unpacked.items():
        if not isinstance(key, str):
            raise TypeError(f"payload keys must be str, got {type(key)}")
        if isinstance(value, bytes) and ("img" in key.lower() or "image" in key.lower()):
            unpacked[key] = decode_image(value)
    return unpacked


__all__ = ["encode_image", "decode_image", "pack_payload", "unpack_payload"]
