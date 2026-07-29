#!/usr/bin/env python3
"""전체화면 웹 키오스크 (WebKit2GTK).

이 장비(Jetson L4T)에서는 snap 기반 Firefox가 snap-confine 커널 호환 문제로
실행되지 않아, 시스템에 설치된 webkit2gtk로 직접 전체화면 창을 띄운다.
사용: python3 kiosk_webview.py [URL]
"""
import sys

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.0")
from gi.repository import Gtk, WebKit2, GLib  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
RELOAD_DELAY_SEC = 3


def main():
    win = Gtk.Window(title="Door Capture Kiosk")
    view = WebKit2.WebView()
    win.add(view)

    view.connect("context-menu", lambda *a: True)  # 우클릭 메뉴 차단

    def reload_later(*_):
        GLib.timeout_add_seconds(
            RELOAD_DELAY_SEC, lambda: (view.load_uri(URL), False)[1]
        )

    # 렌더 프로세스 크래시·로드 실패 시 자동 재시도 (서비스 재시작 중 등)
    view.connect("web-process-terminated", reload_later)
    view.connect(
        "load-failed", lambda v, ev, uri, err: (reload_later(), True)[1]
    )

    win.connect("destroy", Gtk.main_quit)
    win.fullscreen()
    win.show_all()
    view.load_uri(URL)
    Gtk.main()


if __name__ == "__main__":
    main()
