# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only tests for verl.utils.net_utils.get_free_port.

These cover the race-safe reservation mechanism that vllm_async_server.py now
reuses for the single-node (nnodes == 1) MultiprocExecutor path, forwarded to
vLLM via the VLLM_PORT env var, to close verl-project/verl#6677: two rollout
engines starting on the same node could both land in the TOCTOU gap of
vLLM's own get_open_port() (bind -> read assigned port -> release -> bind
again later) and collide on the same port with EADDRINUSE.

No vLLM/GPU dependency is needed: everything here is plain `socket` usage,
mirroring exactly what vLLM's own port-picking code does internally.
"""

import socket
import threading

import pytest

from verl.utils.net_utils import get_free_port


def _plain_bind_like_vllm(port: int) -> bool:
    """Mimic vllm.utils.network_utils._get_open_port's check-bind: no SO_REUSEADDR.

    Returns True if the bind succeeds (and immediately releases the port again,
    exactly like vLLM's own TOCTOU-prone allocator does).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
        return True
    except OSError:
        return False


class TestGetFreePortConcurrentReservations:
    """The core race verl-project/verl#6677 is about: two concurrent callers must
    never be handed the same port while both hold a live reservation."""

    def test_concurrent_reservations_never_collide(self):
        """Launch many concurrent get_free_port(with_alive_sock=True) reservations
        (simulating multiple rollout engines starting on the same node) and assert
        no two of them ever get the same port while all are held open."""
        num_workers = 32
        results: list[tuple[int, socket.socket]] = [None] * num_workers
        errors: list[Exception] = []
        barrier = threading.Barrier(num_workers)

        def reserve(idx: int):
            try:
                barrier.wait(timeout=5)  # maximize concurrent overlap, like a real race
                port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
                results[idx] = (port, sock)
            except Exception as e:  # pragma: no cover - surfaced via `errors`
                errors.append(e)

        threads = [threading.Thread(target=reserve, args=(i,)) for i in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            assert not errors, f"unexpected errors while reserving ports: {errors}"
            ports = [port for port, _sock in results]
            assert len(ports) == len(set(ports)), f"duplicate ports reserved concurrently: {ports}"
        finally:
            for port_sock in results:
                if port_sock is not None:
                    port_sock[1].close()

    def test_without_alive_sock_reservation_is_not_held(self):
        """with_alive_sock=False (verl's default 'immediate use' mode) closes the
        socket internally, so the port is free again right away -- this is the same
        TOCTOU shape as vLLM's own get_open_port(), included here to document why
        with_alive_sock=True is required to actually prevent a race."""
        port, sock = get_free_port("127.0.0.1", with_alive_sock=False)
        assert sock is None
        # The port must be immediately re-bindable, proving nothing is held.
        assert _plain_bind_like_vllm(port)


class TestGetFreePortHandoffToRealBinder:
    """Validates the exact handoff sequence the vllm_async_server.py fix relies on:
    reserve with an alive socket -> forward the port -> close the reservation right
    before the real service (vLLM, via VLLM_PORT) binds it."""

    def test_open_reservation_blocks_a_plain_bind_on_same_port(self):
        """While the reservation socket is alive, nothing else -- in particular a
        vLLM-style plain bind() with no SO_REUSEADDR -- can grab that exact port.
        This is why closing the socket is a required step of the fix, not optional."""
        port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
        try:
            assert not _plain_bind_like_vllm(port), (
                "a plain bind() should fail while the reservation socket is open"
            )
        finally:
            sock.close()

    def test_closing_reservation_immediately_frees_the_exact_port(self):
        """Once released, the target service (vLLM's own bind, forwarded the port
        via VLLM_PORT) must be able to bind that exact port right away."""
        port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
        sock.close()
        assert _plain_bind_like_vllm(port), "closing the reservation socket must free the exact port"

    def test_reservation_prevents_ephemeral_port_from_stealing_it(self):
        """While a port is reserved with an alive socket, the OS ephemeral
        allocator (bind(("", 0))) used by both get_free_port(with_alive_sock=False)
        and vLLM's own get_open_port() must never hand out that exact port to a
        concurrent caller."""
        port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
        try:
            for _ in range(200):
                other_port, other_sock = get_free_port("127.0.0.1", with_alive_sock=False)
                assert other_port != port
        finally:
            sock.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
