"""Exercise actual HTTP lifecycle against a local stand-in, without inference."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

from XPolicyLab.policy.WorldScape_RSI.harness.provider import ResponsesClient


class CleanupTests(unittest.TestCase):
    def test_ambiguous_inference_requires_confirmed_release(self):
        requests = []
        release_ok = [False]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, self.headers["session_id"]))
                status, value = ((503, {"error": "upstream unavailable"}) if
                                 self.path == "/responses" else (200, {"released": release_ok[0]}))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "TEST_BASE": f"http://127.0.0.1:{server.server_port}",
                "TEST_KEY": "test-only",
                "R2SI_SESSION_RELEASE_PATH": "/sessions/{session_id}/release",
            }):
                client = ResponsesClient({"model": "fixture", "effort": "high",
                                          "api_base_env": "TEST_BASE", "api_key_env": "TEST_KEY"})
                identity = client.session_id
                with self.assertRaises(Exception):
                    client.complete([{"role": "user", "content": "fixture"}], [])
                with self.assertRaisesRegex(RuntimeError, "Close"):
                    client.reset()
                with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                    client.close()
                self.assertFalse(client.closed)
                release_ok[0] = True
                client.close()
                client.close()
                self.assertTrue(client.closed)
                self.assertEqual(client.token, "")
                self.assertEqual(requests, [
                    ("/responses", identity),
                    (f"/sessions/{identity}/release", identity),
                    (f"/sessions/{identity}/release", identity),
                ])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
