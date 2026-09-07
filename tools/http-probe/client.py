"""Socket-level acceptance checks using Python's standard HTTP client.

HTTP/chunked parsing is provided by http.client; this code parses only SSE.
The server and all endpoints are synthetic, local, and independent of Aila/GPU.
"""

import argparse
import concurrent.futures
import http.client
import json
import socket
import sys
import time
import traceback
import urllib.parse
from pathlib import Path


class SseClient:
    def __init__(self, port, request_id, *, count=8, padding=0, interval_ms=40, receive_buffer=None):
        self.id = request_id
        self.started = time.perf_counter()
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        self.connection.connect()
        self.socket = self.connection.sock
        if receive_buffer is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        self.receive_buffer = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        query = urllib.parse.urlencode(dict(id=request_id, count=count, padding=padding, interval_ms=interval_ms))
        self.connection.request("GET", "/sse?" + query, headers={"Accept": "text/event-stream"})
        self.response = self.connection.getresponse()
        self.headers_at = time.perf_counter()
        assert self.response.status == 200, self.response.read()
        assert self.response.getheader("Content-Type").lower().startswith("text/event-stream")
        assert self.response.getheader("Transfer-Encoding").lower() == "chunked"
        assert self.response.getheader("Content-Encoding") is None
        self.events = []

    def event(self):
        fields = {}
        raw = bytearray()
        deadline = time.perf_counter() + 3
        while True:
            assert time.perf_counter() < deadline, "SSE event deadline exceeded"
            # Read through HTTPResponse (which decodes chunk framing) one byte at
            # a time. This deliberately splits multibyte UTF-8 for SSE assembly.
            byte = self.response.read(1)
            if not byte:
                assert not raw, "EOF split an SSE event"
                return None
            raw.extend(byte)
            assert len(raw) <= 20000, "SSE event exceeded the configured size bound"
            if raw.endswith(b"\n\n"):
                break
        for line in raw.decode("utf-8", errors="strict").splitlines():
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            fields[name] = value.lstrip(" ")
        fields["data"] = json.loads(fields["data"])
        fields["at"] = time.perf_counter()
        fields["wire_payload_bytes"] = len(raw)
        assert fields["data"]["id"] == self.id, "Crossed client/request data"
        if fields["event"] == "token":
            assert fields["data"]["text"] == "汉字🙂", "UTF-8 text was changed"
        self.events.append(fields)
        return fields

    def complete(self, expected_count):
        while True:
            event = self.event()
            assert event is not None, "EOF before terminal SSE"
            if event["event"] == "done":
                break
        assert self.event() is None, "Data followed terminal SSE"
        tokens = [event for event in self.events if event["event"] == "token"]
        assert [event["data"]["index"] for event in tokens] == list(range(expected_count))
        assert sum(event["event"] == "done" for event in self.events) == 1
        assert self.events[-1]["data"]["count"] == expected_count
        return {
            "id": self.id,
            "headers_seconds": self.headers_at - self.started,
            "first_event_seconds": tokens[0]["at"] - self.started,
            "first_event_at": tokens[0]["at"],
            "done_at": self.events[-1]["at"],
            "duration_seconds": self.events[-1]["at"] - self.started,
            "tokens": len(tokens),
        }

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.response.close()
        finally:
            self.connection.close()


def request_json(port, path, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        payload = response.read()
        assert response.status == 200, (response.status, payload)
        return json.loads(payload)
    finally:
        connection.close()


def wait_for(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.perf_counter() < deadline, "Condition did not become true before deadline"
        time.sleep(0.02)


def run(port, server_report, report):
    clients = []
    tests = report["tests"]
    shutdown_requested = False

    def open_client(*args, **kwargs):
        client = SseClient(port, *args, **kwargs)
        clients.append(client)
        return client

    def record(request_id):
        return request_json(port, "/stats")["records"][request_id]

    def released(request_id):
        result = record(request_id)
        return result if result["terminal_transitions"] == 1 and result["timer_released"] and result["producer_destroyed"] else None

    try:
        deadline = time.perf_counter() + 10
        while True:
            try:
                request_json(port, "/health")
                break
            except (ConnectionError, OSError, http.client.HTTPException):
                assert time.perf_counter() < deadline, "Server startup deadline exceeded"
                time.sleep(0.05)

        client = open_client("incremental", count=8, interval_ms=40)
        incremental = client.complete(8)
        assert incremental["first_event_seconds"] - incremental["headers_seconds"] >= 0.04
        assert incremental["done_at"] - incremental["first_event_at"] >= 0.2
        client.close()
        wait_for(lambda: released("incremental"))
        tests.append({"name": "incremental_utf8_sse", "passed": True, **incremental})

        def concurrent_client(request_id, count, interval):
            own_client = SseClient(port, request_id, count=count, interval_ms=interval)
            try:
                return own_client.complete(count)
            finally:
                own_client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(concurrent_client, "concurrent_a", 7, 35)
            second = pool.submit(concurrent_client, "concurrent_b", 11, 24)
            results = [first.result(timeout=5), second.result(timeout=5)]
        assert max(item["first_event_at"] for item in results) < min(item["done_at"] for item in results)
        tests.append({"name": "two_clients", "passed": True, "clients": results})

        client = open_client("disconnect", count=1000, interval_ms=50)
        survivor = open_client("disconnect_survivor", count=6, interval_ms=35)
        assert client.event()["event"] == "token"
        closed_at = time.perf_counter()
        client.close()
        disconnected = wait_for(lambda: released("disconnect"))
        assert disconnected["status"] == "disconnected"
        assert disconnected["disconnect_callback_seen"], "Disconnect did not reach the request through the connection callback"
        cancellation_delay = time.perf_counter() - closed_at
        assert cancellation_delay < 2
        tokens_after_cancel = disconnected["tokens"]
        time.sleep(0.12)
        assert record("disconnect")["tokens"] == tokens_after_cancel
        survivor_result = survivor.complete(6)
        survivor.close()
        assert wait_for(lambda: released("disconnect_survivor"))["status"] == "completed"
        tests.append({"name": "disconnect_cancels_producer", "passed": True,
                      "cancel_seconds": cancellation_delay, "record": disconnected,
                      "unaffected_peer": survivor_result})

        slow = open_client("slow_reader", count=100000, padding=16000, interval_ms=1, receive_buffer=4096)
        paused = wait_for(lambda: (value if value["pause_ticks"] >= 10 else None)
                          if (value := record("slow_reader")) else None)
        assert paused["status"] == "active"
        assert 0 < paused["queued_upper_bound"] <= 65536
        assert paused["peak_queued_upper_bound"] <= 65536
        latencies = []
        for _ in range(8):
            started = time.perf_counter()
            health = request_json(port, "/health")
            latencies.append(time.perf_counter() - started)
            assert health["active_producers"] >= 1
        assert max(latencies) < 1
        healthy = open_client("healthy_with_slow_peer", count=5, interval_ms=20)
        healthy_result = healthy.complete(5)
        healthy.close()
        later = record("slow_reader")
        assert later["pause_ticks"] > paused["pause_ticks"]
        assert later["peak_queued_upper_bound"] <= 65536
        assert later["tokens"] <= paused["tokens"] + 2, "Stalled peer continued receiving unbounded submissions"
        # Reopening its receive window must allow the same stream to advance.
        for _ in range(6):
            assert slow.event()["event"] == "token"
        wait_for(lambda: record("slow_reader")["tokens"] > later["tokens"])
        slow.close()
        slow_record = wait_for(lambda: released("slow_reader"))
        tests.append({"name": "slow_reader_transport_bound", "passed": True,
                      "client_receive_buffer": slow.receive_buffer,
                      "health_max_seconds": max(latencies), "record": slow_record,
                      "healthy_peer": healthy_result})

        race_results = []
        for index in range(8):
            request_id = f"completion_race_{index}"
            client = open_client(request_id, count=2, interval_ms=8)
            assert client.event()["event"] == "token"
            assert client.event()["event"] == "token"
            time.sleep((index % 4) * 0.005)
            client.close()
            outcome = wait_for(lambda: released(request_id))
            assert outcome["status"] in {"completed", "disconnected"}
            assert outcome["terminal_transitions"] == 1 and outcome["done_events"] <= 1
            race_results.append({"id": request_id, "status": outcome["status"]})
        tests.append({"name": "completion_disconnect_race", "passed": True, "outcomes": race_results})

        shutdown_a = open_client("shutdown_a", count=100000, interval_ms=50)
        shutdown_b = open_client("shutdown_slow", count=100000, padding=16000, interval_ms=1, receive_buffer=4096)
        assert shutdown_a.event()["event"] == "token"
        shutdown_at = time.perf_counter()
        request_json(port, "/shutdown", "POST")
        shutdown_requested = True
        wait_for(lambda: server_report.exists(), timeout=3)
        stopped = json.loads(server_report.read_text(encoding="utf-8"))
        assert stopped["active_producers"] == 0 and not stopped["forced_shutdown"]
        assert stopped["socket_option_failures"] == 0
        assert stopped["socket_option_callbacks"] > 0 and stopped["connection_close_callbacks"] > 0
        assert 0 < stopped["max_socket_send_buffer"] <= 16384
        for request_id in ("shutdown_a", "shutdown_slow"):
            assert stopped["records"][request_id]["status"] == "shutdown"
        for value in stopped["records"].values():
            assert value["terminal_transitions"] == 1
            assert value["timer_released"] and value["producer_destroyed"]
            assert value["peak_queued_upper_bound"] <= stopped["queue_limit"]
        tests.append({"name": "bounded_shutdown", "passed": True,
                      "server_report_seconds": time.perf_counter() - shutdown_at,
                      "record_count": len(stopped["records"]),
                      "socket_send_buffer": stopped["max_socket_send_buffer"]})
    finally:
        for client in clients:
            try:
                client.close()
            except (OSError, http.client.HTTPException):
                pass
        if not shutdown_requested:
            try:
                request_json(port, "/shutdown", "POST")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = {"host": "127.0.0.1", "port": arguments.port, "passed": False, "tests": []}
    started = time.perf_counter()
    try:
        run(arguments.port, arguments.server_report, report)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        report["duration_seconds"] = time.perf_counter() - started
        arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"HTTP/SSE probe: {len(report['tests'])}/6 scenarios passed; report: {arguments.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
