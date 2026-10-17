"""API HTTP minimale (bibliothèque standard) exposant le ClinicalPredictor.

  GET  /health    -> {"status": "ok"}
  GET  /features  -> {"features": [...]}
  POST /diagnose  -> corps JSON {feature: valeur, ...}
                     -> {"diagnosis": "Infecté"|"Sain", "probability": 0.87}
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class DiagnosisAPI:
    def __init__(self, predictor):
        self.predictor = predictor

    def handle_diagnose(self, payload):
        """Retourne (code_http, réponse_dict)."""
        try:
            proba = self.predictor.predict_proba(payload)
        except (ValueError, TypeError, KeyError) as exc:
            return 400, {"error": str(exc)}
        return 200, {
            "diagnosis": self.predictor.label(proba),
            "probability": round(proba, 4),
        }

    def make_server(self, host="127.0.0.1", port=8000):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"status": "ok"})
                elif self.path == "/features":
                    self._send(200, {"features": api.predictor.feature_names})
                else:
                    self._send(404, {"error": "Route inconnue"})

            def do_POST(self):
                if self.path != "/diagnose":
                    self._send(404, {"error": "Route inconnue"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, {"error": "JSON invalide"})
                    return
                status, body = api.handle_diagnose(payload)
                self._send(status, body)

            def log_message(self, *args):
                pass  # silence les logs HTTP

        return ThreadingHTTPServer((host, port), Handler)

    def serve(self, host="127.0.0.1", port=8000):
        server = self.make_server(host, port)
        print(f"API virus_diag sur http://{host}:{port}  (Ctrl+C pour arrêter)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nArrêt du serveur.")
        finally:
            server.server_close()
