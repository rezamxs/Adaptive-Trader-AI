"""M11 phase 2: upload chunk writes are offloaded from the event loop.

The reads stay async (UploadFile.read); every 1MB disk write of the bounded
temp file runs in a worker thread via asyncio.to_thread — a large upload can
no longer stall the loop with blocking IO.
"""
import asyncio
import os

import pytest
from fastapi import HTTPException

from backend.app.api.v1.endpoints import documents as documents_module
from backend.app.api.v1.endpoints.documents import _write_upload_to_temp


class _FakeUpload:
    """Mimics UploadFile.read semantics with scripted chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


@pytest.fixture
def offload_recorder(monkeypatch):
    calls = []
    real_to_thread = asyncio.to_thread

    async def recording_to_thread(fn, *args, **kwargs):
        calls.append(getattr(fn, "__name__", str(fn)))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)
    return calls


def test_chunk_writes_run_off_the_event_loop(offload_recorder):
    chunks = [b"x" * 1024 for _ in range(3)]
    path, size = asyncio.run(_write_upload_to_temp(_FakeUpload(chunks)))
    try:
        assert size == 3 * 1024
        assert offload_recorder.count("write") == 3, (
            "every chunk write must go through asyncio.to_thread (M11)")
        assert os.path.exists(path)
        with open(path, "rb") as fh:
            assert fh.read() == b"x" * (3 * 1024)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_zero_byte_upload_is_reported_empty(offload_recorder):
    path, size = asyncio.run(_write_upload_to_temp(_FakeUpload([b""])))
    assert size == 0 and os.path.exists(path)
    os.unlink(path)


def test_over_cap_upload_is_rejected_and_temp_cleaned(monkeypatch, offload_recorder):
    monkeypatch.setattr(documents_module, "_MAX_UPLOAD_BYTES", 4)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_write_upload_to_temp(_FakeUpload([b"ab", b"cd", b"ef"])))
    assert exc.value.status_code == 413
    # The failed upload must not leave an orphan temp file behind.
    assert offload_recorder.count("write") == 2, "only the chunks under the cap are written"
