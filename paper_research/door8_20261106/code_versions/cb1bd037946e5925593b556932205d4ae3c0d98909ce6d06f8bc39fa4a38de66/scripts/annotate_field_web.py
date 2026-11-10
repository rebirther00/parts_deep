"""Loopback-only human point annotation; no display server or new packages needed."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.annotations import AnnotationStore, RevisionConflict

ASSETS = Path(__file__).with_name('annotation_web')


def make_server(store, port=8765, token=None):
    token = token or secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, kind='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
                             "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self, auth=True):
            host = self.headers.get('Host', '')
            try:
                host_ok = urlsplit('http://' + host).hostname in ('127.0.0.1', 'localhost', '::1')
            except ValueError:
                host_ok = False
            origin = self.headers.get('Origin')
            if not host_ok or (origin and origin != 'http://' + host):
                self.reply(403, {'error': '로컬 포트 전달 주소로 접속하세요.'})
                return False
            if auth and not secrets.compare_digest(self.headers.get('X-Annotation-Token', ''), token):
                self.reply(403, {'error': '터미널에 출력된 전체 접속 주소를 열어주세요.'})
                return False
            return True

        def do_GET(self):
            url = urlsplit(self.path)
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/boot.js': ('boot.js', 'text/javascript; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if not self.allowed(auth=url.path not in assets):
                return
            try:
                if url.path in assets:
                    filename, kind = assets[url.path]
                    data = (ASSETS / filename).read_bytes()
                    if url.path == '/':
                        # A same-origin page receives its CSRF/session token. The
                        # loopback Host/Origin checks and API header remain required.
                        meta = f'<meta name="annotation-token" content="{escape(token, quote=True)}">'
                        data = data.replace(b'<!--ANNOTATION_SESSION-->', meta.encode('utf8'))
                    self.reply(200, data, kind)
                elif url.path == '/api/items':
                    self.reply(200, {'items': store.items()})
                elif url.path in ('/api/item', '/api/image'):
                    key = parse_qs(url.query).get('id', [''])[0]
                    if url.path == '/api/item':
                        self.reply(200, store.item(key))
                    else:
                        path = Path(store.image_path(key))
                        kind = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
                        self.reply(200, path.read_bytes(), kind)
                else:
                    self.reply(404, {'error': '없는 주소입니다.'})
            except (KeyError, FileNotFoundError):
                self.reply(404, {'error': '영상 또는 주석을 찾을 수 없습니다.'})
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != '/api/save':
                self.reply(404, {'error': '없는 주소입니다.'})
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('JSON 저장 요청이 필요합니다.')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('저장 요청 크기가 올바르지 않습니다.')
                data = json.loads(self.rfile.read(size))
                self.reply(200, store.save(data))
            except RevisionConflict as exc:
                self.reply(409, {'error': str(exc)})
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                self.reply(400, {'error': str(exc)})
            except OSError:
                self.reply(500, {'error': '파일을 읽거나 저장하지 못했습니다. 터미널과 저장 공간을 확인하세요.'})

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.annotation_token = token
    server.daemon_threads = True
    return server


def serve(port=8765):
    try:
        server = make_server(AnnotationStore(), port)
    except OSError as exc:
        raise SystemExit(f'주석 서버를 시작할 수 없습니다: {exc}\n다른 포트: --web --port 8766')
    actual_port = server.server_address[1]
    print(f'\n브라우저 주석 주소: http://127.0.0.1:{actual_port}/', flush=True)
    print(f'원격 접속 중이면 VS Code의 포트(Ports)에서 {actual_port} 포트를 전달하세요.', flush=True)
    print('이 터미널을 유지하세요. 종료: Ctrl+C. 저장: data/annotations/field40_v2/\n', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n주석 서버를 종료했습니다. 저장한 라벨은 유지됩니다.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
