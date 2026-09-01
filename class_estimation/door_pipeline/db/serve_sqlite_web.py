#!/usr/bin/env python
"""sqlite_web을 waitress(운영용 WSGI 서버)로 구동.

Flask 개발 서버를 tailscale funnel 프록시 뒤에 두면 간헐적으로 연결이 리셋되는
문제가 있어(ERR_CONNECTION_RESET, 재시도하면 성공) 운영용 서버로 감싼다.
옵션은 sqlite_web CLI와 동일하게 전달된다. 예:

  SQLITE_WEB_PASSWORD='...' python db/serve_sqlite_web.py \
      -r -P -H 127.0.0.1 -p 8080 db/door_pipeline.db
"""
from waitress import serve
from sqlite_web.sqlite_web import app, configure_app

kwargs = configure_app()  # sys.argv 파싱 + 앱 초기화 (sqlite_web main과 동일)

_inner = app.wsgi_app
def _log_requests(environ, start_response):  # 접근 로그 (stdout → nohup 로그)
    def sr(status, headers, exc_info=None):
        loc = next((v for k, v in headers if k.lower() == "location"), "")
        print(f"[req] {environ.get('REQUEST_METHOD')} {environ.get('PATH_INFO')}"
              f" scheme={environ.get('wsgi.url_scheme')} host={environ.get('HTTP_HOST')}"
              f" -> {status}" + (f" Location={loc}" if loc else ""), flush=True)
        return start_response(status, headers, exc_info)
    return _inner(environ, sr)
app.wsgi_app = _log_requests

print(f"sqlite_web (waitress): http://{kwargs['host']}:{kwargs['port']}")
# trusted_proxy: waitress 는 기본적으로 X-Forwarded-* 를 제거하므로, funnel(127.0.0.1)을
# 신뢰 프록시로 지정해 X-Forwarded-Proto: https 가 환경에 반영되게 한다.
# (없으면 세션 next_url/리다이렉트가 절대 http URL 로 생성 → 브라우저가 http 로 빠져 실패)
serve(app, host=kwargs["host"], port=kwargs["port"], threads=8,
      trusted_proxy="127.0.0.1", trusted_proxy_count=1,
      trusted_proxy_headers={"x-forwarded-for", "x-forwarded-proto",
                             "x-forwarded-host", "x-forwarded-port"})
