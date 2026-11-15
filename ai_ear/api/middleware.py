"""Bound complete HTTP request bodies before multipart parsing or spooling."""

from __future__ import annotations

import json


class _RequestBodyTooLarge(Exception):
    """Raised when a streamed request exceeds the configured body limit."""


class RequestBodyLimitMiddleware:
    """Limit the complete POST /analyse body, including multipart overhead.

    Content-Length is rejected before the downstream application reads the
    body. Requests without Content-Length are counted as ASGI chunks arrive,
    so chunked uploads cannot bypass the same limit.
    """

    def __init__(self, app, max_body_bytes: int, path: str = "/analyse"):
        if type(max_body_bytes) is not int or max_body_bytes < 1:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def _reject(self, send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("method") != "POST"
                or scope.get("path") != self.path):
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value for key, value in scope.get("headers", ())
            if key.lower() == b"content-length"
        ]
        if len(content_lengths) > 1:
            await self._reject(send, 400, "Invalid Content-Length header")
            return
        if content_lengths:
            try:
                raw_length = content_lengths[0].decode("ascii").strip()
                if not raw_length.isdecimal():
                    raise ValueError("Content-Length must contain decimal digits")
                declared_length = int(raw_length)
            except (UnicodeDecodeError, ValueError):
                await self._reject(send, 400, "Invalid Content-Length header")
                return
            if declared_length > self.max_body_bytes:
                await self._reject(
                    send,
                    413,
                    f"Request body exceeds configured limit of {self.max_body_bytes} bytes",
                )
                return

        received_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            # This endpoint parses multipart input before starting its response.
            # If that contract changes, fail loudly rather than emit a second
            # response or silently truncate an already-started response.
            if response_started:
                raise RuntimeError("request body limit exceeded after response start")
            await self._reject(
                send,
                413,
                f"Request body exceeds configured limit of {self.max_body_bytes} bytes",
            )
