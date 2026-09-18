"""ZMQ REP 传输层：一问一答，异常不得破坏 REP 状态机。

对应控制端的 `VlaZmqClient`（REQ，`RCVTIMEO` 默认 2000 ms，超时即急停）。
服务端一旦抛异常而不回包，客户端只能等超时；因此这里 **捕获一切异常** 并调用
`fallback`。具体策略必须由上层提供；Piper server 只允许使用当前有限 state 生成保持位姿，
无法安全保持时返回 error 而不是伪造动作。

参考：`vla_infer/src/zmq/zmq_server.py:26-55`、`src/inference/server.py:27-52`
"""

from __future__ import annotations

import logging
import typing as t

import zmq

from wire import pack_payload, unpack_payload

logger = logging.getLogger(__name__)

Handler = t.Callable[[t.Dict[str, t.Any]], t.Dict[str, t.Any]]


class ZmqRepServer:
    """把 `handler(request_dict) -> response_dict` 暴露成 ZMQ REP 服务。"""

    def __init__(
        self,
        handler: Handler,
        host: str = "0.0.0.0",
        port: int = 5555,
        jpeg_quality: int = 80,
        fallback: t.Optional[t.Callable[[t.Dict[str, t.Any]], t.Dict[str, t.Any]]] = None,
        max_requests: int = 0,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {port}")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
        if max_requests < 0:
            raise ValueError(f"max_requests must be >= 0, got {max_requests}")
        self.handler = handler
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.fallback = fallback
        self.max_requests = max_requests  # 0 = 无限循环（仅调试用有限值）

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)  # 与对端一致：close() 立即销毁
        try:
            self._sock.bind(f"tcp://{host}:{port}")
        except Exception:
            self._sock.close()
            self._ctx.term()
            raise
        self._closed = False
        logger.info("ZMQ REP server ready on tcp://%s:%s", host, port)

    # ------------------------------------------------------------------ #
    def _safe_handle(self, request: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        try:
            response = self.handler(request)
            if not isinstance(response, dict):
                raise TypeError(f"handler must return dict, got {type(response)}")
            return response
        except Exception:
            logger.exception("handler failed, falling back to safe response")
            if self.fallback is None:
                return {"error": "handler failed and no fallback is configured"}
            try:
                return self.fallback(request)
            except Exception:
                logger.exception("fallback also failed")
                return {"error": "handler and safe fallback failed"}

    # ------------------------------------------------------------------ #
    def serve_forever(self, poll_timeout_ms: int = 200) -> int:
        """阻塞服务循环；返回已处理请求数。`Ctrl-C` 可优雅退出。"""
        if poll_timeout_ms < 0:
            raise ValueError(f"poll_timeout_ms must be >= 0, got {poll_timeout_ms}")
        served = 0
        try:
            while self.max_requests <= 0 or served < self.max_requests:
                if not self._sock.poll(poll_timeout_ms):
                    continue  # 让 Ctrl-C / 退出逻辑有机会执行
                try:
                    frames = self._sock.recv_multipart()
                    if len(frames) != 1:
                        raise ValueError(f"expected one request frame, got {len(frames)}")
                    request = unpack_payload(frames[0])
                except Exception:
                    logger.exception("request decode failed; returning error response")
                    request = {}
                    response = self._safe_handle(request)
                else:
                    response = self._safe_handle(request)

                try:
                    packed = pack_payload(response, jpeg_quality=self.jpeg_quality)
                except Exception:
                    logger.exception("response encode failed; returning minimal error response")
                    packed = pack_payload({"error": "response encoding failed"}, jpeg_quality=self.jpeg_quality)
                self._sock.send(packed)
                served += 1
        except KeyboardInterrupt:
            logger.info("interrupted by user after %d request(s)", served)
        except zmq.ZMQError:
            logger.exception("ZMQ server loop failed after %d request(s)", served)
        finally:
            self.close()
        return served

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sock.close()
        self._ctx.term()
        logger.info("ZMQ server closed")


__all__ = ["ZmqRepServer", "Handler"]
