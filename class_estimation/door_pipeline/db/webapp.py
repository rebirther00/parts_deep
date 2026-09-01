"""door_pipeline.db 열람·관리 웹 도구 (SQL 없이 브라우저로).

  python db/webapp.py                  # http://<학습PC>:5050
  python db/webapp.py --readonly       # 열람 전용 (정정·split 버튼 비활성)
  python db/webapp.py --port 5051 --db 다른경로.db

화면:
  /          대시보드 — 규모 요약, 데이터셋 현황, 현장 수집 클래스별 세션·수집량, 최근 세션·평가
  /sessions  세션 목록 — build_dataset.py status 와 같은 내용을 표로
  /sessions/<id>  세션 상세 — 메타·이력(notes)·썸네일 + 라벨정정/split/무효화 버튼
  /images    이미지 브라우저 — 데이터셋·클래스·split 필터 + 썸네일
  /training  학습·평가 이력 — 학습 곡선(SVG), 평가 결과 표

관리 동작은 build_dataset.py 의 함수를 그대로 호출한다(규칙 동일: 파일 이동 없음,
세션 단위 split, notes 에 이력). 썸네일은 db/.thumb_cache/ 에 캐시된다.
"""
import argparse
import ast
import json
import sqlite3
import statistics
import time
from pathlib import Path

from flask import (Flask, abort, flash, g, redirect, render_template_string,
                   request, send_file, url_for)
from PIL import Image, ImageDraw

import build_dataset as bd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = BASE_DIR / "db" / "door_pipeline.db"
MIRROR = BASE_DIR / "datasets_factory_collect"
THUMB_DIR = BASE_DIR / "db" / ".thumb_cache"
PAGE_SIZE = 60

app = Flask(__name__)
app.secret_key = "door-db-webapp-local"
app.config["DB_PATH"] = str(DEFAULT_DB)
app.config["READONLY"] = False


# ── DB ──────────────────────────────────────────────────

def db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    con = g.pop("db", None)
    if con:
        con.close()


def image_abs_path(base_path, rel_path):
    """images.rgb_path(상대) → 로컬 절대경로. NAS 데이터셋은 로컬 미러에서 찾는다."""
    if not rel_path:
        return None
    root = MIRROR if base_path.startswith("nas:") else BASE_DIR / base_path
    return root / rel_path


def guard_write():
    if app.config["READONLY"]:
        abort(403, "열람 전용 모드로 실행 중입니다 (--readonly 없이 재시작하면 수정 가능).")


# ── 공통 레이아웃 ────────────────────────────────────────

LAYOUT = """<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🗄️</text></svg>">
<title>{{ title }} · door_pipeline DB</title>
<style>
:root { --paper:#F4F6F8; --card:#FFF; --ink:#1B2430; --muted:#5A6676; --line:#DCE2E9;
        --accent:#2E5E8C; --accent-soft:#E8EFF6; --flag:#B45D0E; --flag-soft:#FAF1E5;
        --good:#2F7D4F; color-scheme: light dark; }
@media (prefers-color-scheme: dark) { :root {
  --paper:#12161C; --card:#1A2028; --ink:#E6EAF0; --muted:#94A1B1; --line:#2A323D;
  --accent:#74A8D8; --accent-soft:#1F2C3A; --flag:#E09A4A; --flag-soft:#2A2118; --good:#6BBD8E; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:14px/1.65 'Apple SD Gothic Neo','Malgun Gothic',sans-serif; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
nav { background:var(--card); border-bottom:1px solid var(--line); padding:0 20px;
      display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
nav .brand { font-weight:700; margin-right:16px; padding:12px 0; }
nav a.tab { padding:12px 12px; color:var(--muted); border-bottom:2px solid transparent; }
nav a.tab.on { color:var(--ink); border-bottom-color:var(--accent); font-weight:500; }
nav .ro { margin-left:auto; font-size:12px; color:var(--flag); }
main { max-width:1200px; margin:0 auto; padding:22px 20px 60px; }
h1 { font-size:20px; margin:6px 0 16px; } h2 { font-size:16px; margin:26px 0 10px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:8px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 14px; }
.card .num { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
.card .lbl { font-size:12px; color:var(--muted); }
.tbl { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:6px; }
.tbl.tall { max-height:72vh; overflow-y:auto; }   /* 가로 스크롤바가 화면 안에 보이도록 */
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:7px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-size:12px; color:var(--muted); font-weight:500; background:var(--paper); position:sticky; top:0; }
tr:last-child td { border-bottom:none; } tr.click:hover { background:var(--accent-soft); cursor:pointer; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; background:var(--accent-soft); color:var(--accent); }
.pill.train { background:var(--accent-soft); color:var(--accent); }
.pill.test { background:var(--flag-soft); color:var(--flag); }
.pill.val { background:color-mix(in srgb, var(--good) 15%, transparent); color:var(--good); }
.pill.none { background:transparent; border:1px dashed var(--line); color:var(--muted); }
.warn { color:var(--flag); font-size:12px; } .ok { color:var(--good); }
.muted { color:var(--muted); } .small { font-size:12px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
.thumb { background:var(--card); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.thumb img { width:100%; height:110px; object-fit:cover; display:block; background:#888; }
.thumb .cap { padding:5px 8px; font-size:11.5px; color:var(--muted); line-height:1.45; }
form.inline { display:inline-flex; gap:6px; align-items:center; flex-wrap:wrap; }
select,input[type=text] { padding:5px 8px; border:1px solid var(--line); border-radius:4px; background:var(--card); color:var(--ink); font-size:13px; }
button { padding:5px 12px; border:1px solid var(--accent); border-radius:4px; background:var(--accent); color:#fff; font-size:13px; cursor:pointer; }
button.ghost { background:var(--card); color:var(--accent); }
button.danger { background:var(--card); border-color:var(--flag); color:var(--flag); }
button:disabled { opacity:.45; cursor:not-allowed; }
.flash { background:var(--accent-soft); border:1px solid var(--accent); border-radius:6px; padding:8px 14px; margin-bottom:14px; }
.flash.err { background:var(--flag-soft); border-color:var(--flag); }
.filters { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:14px; }
.box { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
pre.notes { background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:10px 12px; font-size:12.5px; overflow-x:auto; white-space:pre-wrap; }
.pager { display:flex; gap:10px; align-items:center; margin-top:14px; }
svg text { fill:var(--muted); font-size:10px; }
</style></head><body>
<nav>
  <span class="brand">🗄️ door_pipeline DB</span>
  {% for ep, label in [('dashboard','대시보드'), ('sessions','세션'), ('images','이미지'), ('labels','라벨'), ('training','학습·평가')] %}
    <a class="tab {{ 'on' if active == ep }}" href="{{ url_for(ep) }}">{{ label }}</a>
  {% endfor %}
  {% if readonly %}<span class="ro">열람 전용 모드</span>{% endif %}
</nav>
<main>
{% with msgs = get_flashed_messages(with_categories=true) %}
  {% for cat, m in msgs %}<div class="flash {{ 'err' if cat == 'error' }}">{{ m }}</div>{% endfor %}
{% endwith %}
{{ body | safe }}
</main></body></html>"""


def page(title, active, body_tpl, **ctx):
    body = render_template_string(body_tpl, **ctx)
    return render_template_string(LAYOUT, title=title, active=active, body=body,
                                  readonly=app.config["READONLY"])


# ── 대시보드 ────────────────────────────────────────────

DASH = """
<h1>대시보드</h1>
<div class="cards">
  <div class="card"><div class="num">{{ c.datasets }}</div><div class="lbl">데이터셋</div></div>
  <div class="card"><div class="num">{{ c.sessions }}</div><div class="lbl">촬영 세션</div></div>
  <div class="card"><div class="num">{{ '{:,}'.format(c.images) }}</div><div class="lbl">이미지 메타 행</div></div>
  <div class="card"><div class="num">{{ '{:,}'.format(c.local) }}</div><div class="lbl">로컬 보유 (실촬영)</div></div>
  <div class="card"><div class="num">{{ c.models }}</div><div class="lbl">모델</div></div>
  <div class="card"><div class="num">{{ c.evals }}</div><div class="lbl">평가 기록</div></div>
</div>

<h2>데이터셋</h2>
<div class="tbl"><table>
<tr><th>이름</th><th>설명</th><th class="n">클래스</th><th class="n">이미지</th><th class="n">로컬</th><th>저장 위치</th></tr>
{% for d in datasets %}
<tr><td><a href="{{ url_for('images', dataset_id=d['id']) }}">{{ d['name'] }}</a></td>
    <td class="muted small" style="white-space:normal;max-width:340px">{{ d['description'] or '' }}</td>
    <td class="n">{{ d['n_cls'] }}</td><td class="n">{{ '{:,}'.format(d['n_img']) }}</td>
    <td class="n">{{ '{:,}'.format(d['n_local']) }}</td>
    <td class="muted small">{{ d['base_path'] }}</td></tr>
{% endfor %}
</table></div>

<h2>현장 수집(door_factory_collect) — 클래스별 수집 현황 <span class="muted small">쌍 = NAS 수집량(메타 기준), 로컬 = 유효·동기화(평가에 가용)</span></h2>
<div class="tbl"><table>
<tr><th>클래스</th><th class="n">세션</th><th class="n">수집 쌍</th><th class="n">로컬 보유</th><th>최근 수집</th></tr>
{% for r in field_rows %}
<tr><td>{{ r.cls }}
  {%- if r.pairs == 0 %} <span class="warn">미수집</span>
  {%- elif r.cls == 'Unknown' %} <span class="warn">라벨 미정</span>{% endif %}</td>
  <td class="n">{{ r.sessions or '·' }}</td>
  <td class="n">{{ '{:,}'.format(r.pairs) if r.pairs else '·' }}</td>
  <td class="n">{{ r.local or '·' }}</td>
  <td class="muted">{{ r.latest or '—' }}</td></tr>
{% endfor %}
<tr><td><b>계</b></td><td class="n"><b>{{ field_total.sessions }}</b></td>
  <td class="n"><b>{{ '{:,}'.format(field_total.pairs) }}</b></td>
  <td class="n"><b>{{ '{:,}'.format(field_total.local) }}</b></td><td></td></tr>
</table></div>

<h2>최근 세션</h2>
<div class="tbl"><table>
<tr><th>세션</th><th>클래스(DB)</th><th>현장 입력</th><th class="n">장수</th><th>split</th><th>시작</th></tr>
{% for s in recent_sessions %}
<tr class="click" onclick="location='{{ url_for('session_detail', sid=s['id']) }}'">
  <td>{{ s['session_dir'] }}</td><td>{{ s['cls'] or '—' }}</td>
  <td>{{ s['class_name'] }}{% if s['cls'] and s['class_name'] != s['cls'] %} <span class="warn">*정정됨</span>{% endif %}</td>
  <td class="n">{{ s['n'] }}</td>
  <td><span class="pill {{ s['split'] or 'none' }}">{{ s['split'] or '미지정' }}</span></td>
  <td class="muted">{{ s['started_at'] }}</td></tr>
{% endfor %}
</table></div>

<h2>최근 평가</h2>
<div class="tbl"><table>
<tr><th>모델</th><th>학습세션</th><th>데이터셋</th><th>유형</th><th class="n">정확도</th><th class="n">표본</th><th>일시</th></tr>
{% for e in recent_evals %}
<tr><td>{{ e['model'] }}</td>
    <td>{% if e['session_id'] %}<span class="pill">#{{ e['session_id'] }}</span>{% else %}<span class="muted small">—</span>{% endif %}</td>
    <td>{{ e['dataset'] }}</td><td class="muted">{{ e['eval_type'] }}</td>
    <td class="n"><b>{{ '%.1f'|format(e['accuracy']) }}%</b></td>
    <td class="n">{{ e['correct'] }}/{{ e['total_samples'] }}</td>
    <td class="muted">{{ e['evaluated_at'] }}</td></tr>
{% endfor %}
</table></div>
"""


@app.route("/")
def dashboard():
    con = db()
    c = {k: con.execute(q).fetchone()[0] for k, q in {
        "datasets": "SELECT COUNT(*) FROM datasets",
        "sessions": "SELECT COUNT(*) FROM capture_sessions",
        "images": "SELECT COUNT(*) FROM images",
        "local": "SELECT COUNT(*) FROM images WHERE data_source='camera' AND synced_local",
        "models": "SELECT COUNT(*) FROM models",
        "evals": "SELECT COUNT(*) FROM evaluation_results",
    }.items()}
    datasets = con.execute(
        """SELECT d.*, (SELECT COUNT(*) FROM classes c WHERE c.dataset_id=d.id AND c.image_count>0) n_cls,
                  (SELECT COUNT(*) FROM images i JOIN classes c ON c.id=i.class_id WHERE c.dataset_id=d.id) n_img,
                  (SELECT COUNT(*) FROM images i JOIN classes c ON c.id=i.class_id
                    WHERE c.dataset_id=d.id AND i.synced_local) n_local
           FROM datasets d ORDER BY d.id""").fetchall()
    collected = {r["cls"]: r for r in con.execute(
        """SELECT c.name cls, COUNT(DISTINCT i.session_id) sessions, COUNT(*) pairs,
                  SUM(i.synced_local AND i.is_valid) local, MAX(s.started_at) latest
           FROM images i JOIN classes c ON c.id=i.class_id
           JOIN datasets d ON d.id=c.dataset_id
           LEFT JOIN capture_sessions s ON s.id=i.session_id
           WHERE d.name=? GROUP BY c.name""", (bd.DATASET_NAME,))}
    # 실험실 8종을 기준 목록으로 삼아 미수집 클래스도 0으로 노출 (Unknown은 맨 뒤)
    all_cls = [r[0] for r in con.execute(
        """SELECT c.name FROM classes c JOIN datasets d ON d.id=c.dataset_id
           WHERE d.name='door_real' AND c.name!='Unknown' ORDER BY c.name""")]
    field_rows, field_total = [], {"sessions": 0, "pairs": 0, "local": 0}
    for cls in all_cls + sorted(c for c in collected if c not in all_cls):
        r = collected.get(cls)
        row = {"cls": cls,
               "sessions": r["sessions"] if r else 0,
               "pairs": r["pairs"] if r else 0,
               "local": (r["local"] or 0) if r else 0,
               "latest": (r["latest"] or "")[:16] if r else ""}
        field_rows.append(row)
        for k in field_total:
            field_total[k] += row[k]
    # 라벨 정정으로 한 세션이 두 클래스에 걸치면 클래스별 합이 과대 — 계는 고유 세션 수로
    field_total["sessions"] = con.execute(
        """SELECT COUNT(DISTINCT i.session_id) FROM images i JOIN classes c ON c.id=i.class_id
           JOIN datasets d ON d.id=c.dataset_id WHERE d.name=?""", (bd.DATASET_NAME,)).fetchone()[0]
    recent_sessions = con.execute(
        """SELECT s.id, s.session_dir, s.class_name, s.started_at,
                  (SELECT c.name FROM images i JOIN classes c ON c.id=i.class_id
                    WHERE i.session_id=s.id LIMIT 1) cls,
                  (SELECT COUNT(*) FROM images i WHERE i.session_id=s.id) n,
                  (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split
           FROM capture_sessions s ORDER BY s.started_at DESC LIMIT 6""").fetchall()
    recent_evals = con.execute(
        """SELECT m.name model, d.name dataset, e.* FROM evaluation_results e
           JOIN models m ON m.id=e.model_id JOIN datasets d ON d.id=e.dataset_id
           ORDER BY e.evaluated_at DESC LIMIT 6""").fetchall()
    return page("대시보드", "dashboard", DASH, c=type("C", (), c), datasets=datasets,
                field_rows=field_rows, field_total=field_total,
                recent_sessions=recent_sessions, recent_evals=recent_evals)


# ── 세션 목록·상세 ───────────────────────────────────────

SESSIONS = """
<h1>촬영 세션 <span class="muted small">{{ rows|length }}개 — 행을 누르면 상세·이미지·관리</span></h1>
<p class="small muted" style="margin:-8px 0 12px">
  정렬:
  {% if order == 'desc' %}<b>최신순</b> · <a href="{{ url_for('sessions', order='asc', show_invalid=1 if show_invalid else none) }}">오래된순</a>
  {% else %}<a href="{{ url_for('sessions', show_invalid=1 if show_invalid else none) }}">최신순</a> · <b>오래된순</b>{% endif %}
  &nbsp;|&nbsp;
  {% if n_hidden %}무효 세션 {{ n_hidden }}개 숨김 — <a href="{{ url_for('sessions', show_invalid=1, order=order) }}">표시하기</a>
  {% elif show_invalid %}무효 세션 포함 — <a href="{{ url_for('sessions', order=order) }}">숨기기</a>{% endif %}
</p>
<div class="tbl"><table>
<tr><th>세션 (날짜/현장라벨/시각)</th><th>클래스(DB 확정)</th><th class="n">NAS</th><th class="n">로컬</th>
    <th>split</th><th>상태</th><th>이력(notes)</th></tr>
{% for r in rows %}
<tr class="click" onclick="location='{{ url_for('session_detail', sid=r['id']) }}'">
  <td>{{ r['session_dir'] }}</td>
  <td>{{ r['cls'] or '—' }}{% if r['cls'] and r['class_name'] != r['cls'] %} <span class="warn">*현장입력: {{ r['class_name'] }}</span>{% endif %}</td>
  <td class="n">{{ r['n'] }}</td><td class="n">{{ r['n_local'] }}</td>
  <td><span class="pill {{ r['split'] or 'none' }}">{{ r['split'] or '미지정' }}</span></td>
  <td>{% if not r['valid'] %}<span class="warn">무효</span>{% elif r['cls'] == 'Unknown' %}<span class="warn">라벨 미확정</span>{% else %}<span class="ok">유효</span>{% endif %}</td>
  <td class="muted small" style="white-space:normal;max-width:300px">{{ (r['notes'] or '').split('\\n')[-1] }}</td></tr>
{% endfor %}
</table></div>
{% if not readonly %}
<p style="margin-top:16px">
<form class="inline" method="post" action="{{ url_for('rebuild_view') }}"
      onsubmit="return confirm('datasets_factory_v2/ 링크 뷰를 지우고 다시 만듭니다. 진행할까요?')">
  <button class="ghost">🔄 학습·평가 뷰 재생성 (build)</button>
  <span class="muted small">라벨·split 변경 후 datasets_factory_v2/ 를 최신 상태로 다시 만듭니다</span>
</form></p>
{% endif %}
"""


def session_list_rows(con, order="desc"):
    direction = "DESC" if order == "desc" else "ASC"
    return con.execute(
        f"""SELECT s.id, s.session_dir, s.class_name, s.notes, s.started_at,
                  (SELECT c.name FROM images i JOIN classes c ON c.id=i.class_id
                    WHERE i.session_id=s.id LIMIT 1) cls,
                  COUNT(i.id) n, COALESCE(SUM(i.synced_local),0) n_local,
                  COALESCE(MIN(i.is_valid),1) valid,
                  (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split
           FROM capture_sessions s LEFT JOIN images i ON i.session_id=s.id
           GROUP BY s.id ORDER BY s.session_dir {direction}""").fetchall()


@app.route("/sessions")
def sessions():
    order = "asc" if request.args.get("order") == "asc" else "desc"
    rows = session_list_rows(db(), order)
    show_invalid = request.args.get("show_invalid") == "1"
    if not show_invalid:
        visible = [r for r in rows if r["valid"]]
        n_hidden = len(rows) - len(visible)
        rows = visible
    else:
        n_hidden = 0
    return page("세션", "sessions", SESSIONS, rows=rows, n_hidden=n_hidden, order=order,
                show_invalid=show_invalid, readonly=app.config["READONLY"])


DETAIL = """
<p class="small"><a href="{{ url_for('sessions') }}">← 세션 목록</a></p>
<h1>{{ s['session_dir'] }}
  <span class="pill {{ split or 'none' }}">{{ split or 'split 미지정' }}</span>
  {% if not valid %}<span class="warn">무효 처리됨</span>{% endif %}</h1>

<div class="cards">
  <div class="card"><div class="num">{{ cls or '—' }}</div><div class="lbl">클래스 (DB 확정)</div></div>
  <div class="card"><div class="num">{{ s['class_name'] }}</div><div class="lbl">현장 입력 원본</div></div>
  <div class="card"><div class="num">{{ n }}</div><div class="lbl">NAS 등록 장수</div></div>
  <div class="card"><div class="num">{{ n_local }}</div><div class="lbl">로컬 샘플</div></div>
</div>

<div class="box small" style="margin-bottom:14px">
  카메라 {{ s['camera_type'] }} · {{ s['resolution'] or '?' }} · {{ s['capture_method'] or '?' }}
  {% if s['capture_interval_s'] %}· {{ s['capture_interval_s'] }}s 간격{% endif %}
  · 시작 {{ s['started_at'] }}{% if s['ended_at'] %} · 종료 {{ s['ended_at'] }}{% endif %}
  {% if s['stop_reason'] %}· 종료 사유: {{ s['stop_reason'] }}{% endif %}
</div>

<h2>이력 (notes)</h2>
{% if s['notes'] %}<pre class="notes">{{ s['notes'] }}</pre>
{% else %}<p class="muted small">기록 없음</p>{% endif %}
{% if not readonly %}
<form class="inline" method="post" action="{{ url_for('act_note', sid=s['id']) }}" style="margin-bottom:8px">
  <input type="text" name="text" placeholder="예: 지그 재정렬 후 재촬영, 조명 교체" required style="min-width:340px">
  <button>이력 추가</button>
</form>
{% endif %}

{% if manageable and not readonly %}
<h2>관리</h2>
<div class="box">
  <form class="inline" method="post" action="{{ url_for('act_relabel', sid=s['id']) }}"
        onsubmit="return confirm('이 세션 전체({{ n }}장)의 라벨을 바꿉니다. 파일은 이동하지 않고 DB만 수정되며 notes에 이력이 남습니다.')">
    <b>라벨 정정</b>
    <select name="new_class">{% for c in class_choices %}<option {{ 'selected' if c == cls }}>{{ c }}</option>{% endfor %}</select>
    <input type="text" name="custom" placeholder="직접 입력 (선택)">
    <button>정정</button>
  </form>
  <hr style="border:none;border-top:1px solid var(--line);margin:10px 0">
  <form class="inline" method="post" action="{{ url_for('act_split', sid=s['id']) }}">
    <b>split 지정</b> <span class="muted small">(세션 단위 — 프레임 단위 분할 금지)</span>
    {% for sp in ['train','val','test','none'] %}
      <button class="ghost" name="split" value="{{ sp }}" {{ 'disabled' if split == (None if sp=='none' else sp) }}>{{ sp }}</button>
    {% endfor %}
  </form>
  <hr style="border:none;border-top:1px solid var(--line);margin:10px 0">
  <form class="inline" method="post" action="{{ url_for('act_validity', sid=s['id']) }}"
        onsubmit="return confirm('세션 유효 상태를 변경합니다.')">
    <b>{{ '복귀(validate)' if not valid else '무효화(invalidate)' }}</b>
    <input type="text" name="reason" placeholder="사유 (예: 시험 촬영, 빈 지그)" required>
    <input type="hidden" name="action" value="{{ 'validate' if not valid else 'invalidate' }}">
    <button class="danger">{{ '유효로 복귀' if not valid else '무효 처리' }}</button>
  </form>
</div>
{% endif %}

<h2>로컬 샘플 이미지 ({{ imgs|length }}장)</h2>
{% if imgs %}
<div class="grid">
{% for i in imgs %}
  <div class="thumb"><a href="{{ url_for('raw_image', img_id=i['id']) }}" target="_blank">
    <img src="{{ url_for('thumb', img_id=i['id']) }}" loading="lazy" alt="{{ i['rgb_filename'] }}"></a>
    <div class="cap">{{ i['rgb_filename'] }}{% if i['blur_score'] %} · blur {{ '%.0f'|format(i['blur_score']) }}{% endif %}</div></div>
{% endfor %}
</div>
{% else %}<p class="muted">로컬에 동기화된 이미지가 없습니다 (NAS에만 존재 — <code>python db/pull_nas.py</code>로 샘플을 받으세요).</p>{% endif %}
"""


@app.route("/sessions/<int:sid>")
def session_detail(sid):
    con = db()
    s = con.execute("SELECT * FROM capture_sessions WHERE id=?", (sid,)).fetchone()
    if not s:
        abort(404)
    info = con.execute(
        """SELECT (SELECT c.name FROM images i JOIN classes c ON c.id=i.class_id
                    WHERE i.session_id=? LIMIT 1) cls,
                  (SELECT COUNT(*) FROM images WHERE session_id=?) n,
                  (SELECT COUNT(*) FROM images WHERE session_id=? AND synced_local) n_local,
                  (SELECT COALESCE(MIN(is_valid),1) FROM images WHERE session_id=?) valid,
                  (SELECT split FROM images WHERE session_id=? LIMIT 1) split""",
        (sid,) * 5).fetchone()
    imgs = con.execute(
        """SELECT id, rgb_filename, blur_score FROM images
           WHERE session_id=? AND synced_local ORDER BY rgb_path""", (sid,)).fetchall()
    ds_name = con.execute("SELECT name FROM datasets WHERE id=?", (s["dataset_id"],)).fetchone()[0]
    choices = [r[0] for r in con.execute(
        """SELECT DISTINCT name FROM classes WHERE dataset_id IN
             (SELECT id FROM datasets WHERE name IN (?, 'door_real'))
           ORDER BY name""", (bd.DATASET_NAME,))]
    if "Unknown" not in choices:
        choices.append("Unknown")
    return page(s["session_dir"], "sessions", DETAIL, s=s, cls=info["cls"], n=info["n"],
                n_local=info["n_local"], valid=info["valid"], split=info["split"], imgs=imgs,
                manageable=(ds_name == bd.DATASET_NAME), class_choices=choices,
                readonly=app.config["READONLY"])


# ── 관리 동작 (build_dataset.py 재사용) ──────────────────

def _session_dir(sid):
    r = db().execute("SELECT session_dir FROM capture_sessions WHERE id=?", (sid,)).fetchone()
    if not r:
        abort(404)
    return r["session_dir"]


@app.post("/sessions/<int:sid>/relabel")
def act_relabel(sid):
    guard_write()
    new_cls = (request.form.get("custom") or "").strip() or request.form["new_class"]
    sdir = _session_dir(sid)
    try:
        bd.cmd_relabel(db(), sdir, new_cls)
        flash(f"라벨 정정 완료: {sdir} → {new_cls} (notes에 이력 기록)")
    except Exception as e:  # noqa: BLE001
        flash(f"정정 실패: {e}", "error")
    return redirect(url_for("session_detail", sid=sid))


@app.post("/sessions/<int:sid>/split")
def act_split(sid):
    guard_write()
    sp = request.form["split"]
    sdir = _session_dir(sid)
    try:
        bd.cmd_split(db(), sdir, sp)
        flash(f"split 변경: {sdir} → {sp}")
    except Exception as e:  # noqa: BLE001
        flash(f"변경 실패: {e}", "error")
    return redirect(url_for("session_detail", sid=sid))


@app.post("/sessions/<int:sid>/validity")
def act_validity(sid):
    guard_write()
    valid = request.form["action"] == "validate"
    sdir = _session_dir(sid)
    try:
        bd.cmd_validity(db(), sdir, valid, request.form["reason"])
        flash(f"{'유효 복귀' if valid else '무효 처리'} 완료: {sdir}")
    except Exception as e:  # noqa: BLE001
        flash(f"실패: {e}", "error")
    return redirect(url_for("session_detail", sid=sid))


@app.post("/sessions/<int:sid>/note")
def act_note(sid):
    guard_write()
    text = request.form.get("text", "").strip()
    _session_dir(sid)                              # 존재 확인 (404)
    if text:
        note = f"[{time.strftime('%Y-%m-%d %H:%M')}] 메모: {text}"
        con = db()
        con.execute("UPDATE capture_sessions SET notes = COALESCE(notes || '\n', '') || ? "
                    "WHERE id = ?", (note, sid))
        con.commit()
        flash("이력 추가됨")
    return redirect(url_for("session_detail", sid=sid))


@app.post("/build")
def rebuild_view():
    guard_write()
    try:
        bd.cmd_build(db())
        flash(f"{bd.VIEW.name}/ 뷰 재생성 완료")
    except Exception as e:  # noqa: BLE001
        flash(f"뷰 생성 실패: {e}", "error")
    return redirect(url_for("sessions"))


# ── 이미지 브라우저 ──────────────────────────────────────

IMAGES = """
<h1>이미지 브라우저 <span class="muted small">{{ '{:,}'.format(total) }}장 중 {{ rows|length }}장 표시</span></h1>
<form class="filters" method="get">
  <select name="dataset_id" onchange="this.form.submit()">
    <option value="">데이터셋 전체</option>
    {% for d in dsets %}<option value="{{ d['id'] }}" {{ 'selected' if d['id']|string == cur.dataset_id }}>{{ d['name'] }}</option>{% endfor %}
  </select>
  <select name="cls" onchange="this.form.submit()">
    <option value="">클래스 전체</option>
    {% for c in clss %}<option {{ 'selected' if c == cur.cls }}>{{ c }}</option>{% endfor %}
  </select>
  <select name="split" onchange="this.form.submit()">
    <option value="">split 전체</option>
    {% for sp in ['train','val','test','(미지정)'] %}<option {{ 'selected' if sp == cur.split }}>{{ sp }}</option>{% endfor %}
  </select>
  <select name="synced" onchange="this.form.submit()">
    <option value="">로컬+NAS</option>
    <option value="1" {{ 'selected' if cur.synced == '1' }}>로컬 보유만</option>
  </select>
  <a class="small" href="{{ url_for('images') }}">필터 초기화</a>
</form>
<div class="grid">
{% for i in rows %}
  <div class="thumb">
    {% if i['synced_local'] %}<a href="{{ url_for('raw_image', img_id=i['id']) }}" target="_blank">
      <img src="{{ url_for('thumb', img_id=i['id']) }}" loading="lazy" alt=""></a>
    {% else %}<div style="height:110px;display:flex;align-items:center;justify-content:center;background:var(--paper)" class="muted small">NAS에만 있음</div>{% endif %}
    <div class="cap">{{ i['cls'] }}<br>{{ i['rgb_filename'] }}
      {% if i['split'] %}<span class="pill {{ i['split'] }}">{{ i['split'] }}</span>{% endif %}
      {% if not i['is_valid'] %}<span class="warn">무효</span>{% endif %}</div></div>
{% endfor %}
</div>
<div class="pager">
  {% if pg > 1 %}<a href="{{ prev_url }}">← 이전</a>{% endif %}
  <span class="muted small">{{ pg }} / {{ pages }} 페이지</span>
  {% if pg < pages %}<a href="{{ next_url }}">다음 →</a>{% endif %}
</div>
"""


@app.route("/images")
def images():
    con = db()
    cur = {k: request.args.get(k, "") for k in ("dataset_id", "cls", "split", "synced")}
    pg = max(1, request.args.get("page", 1, type=int))
    where, params = ["1=1"], []
    if cur["dataset_id"]:
        where.append("d.id=?"); params.append(cur["dataset_id"])
    if cur["cls"]:
        where.append("c.name=?"); params.append(cur["cls"])
    if cur["split"] == "(미지정)":
        where.append("i.split IS NULL")
    elif cur["split"]:
        where.append("i.split=?"); params.append(cur["split"])
    if cur["synced"] == "1":
        where.append("i.synced_local")
    base = f"""FROM images i JOIN classes c ON c.id=i.class_id
               JOIN datasets d ON d.id=c.dataset_id WHERE {' AND '.join(where)}"""
    total = con.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
    pages = max(1, -(-total // PAGE_SIZE))
    rows = con.execute(
        f"""SELECT i.id, i.rgb_filename, i.split, i.is_valid, i.synced_local, c.name cls
            {base} ORDER BY d.id, c.name, i.rgb_path LIMIT ? OFFSET ?""",
        params + [PAGE_SIZE, (pg - 1) * PAGE_SIZE]).fetchall()
    dsets = con.execute("SELECT id, name FROM datasets ORDER BY id").fetchall()
    clss = [r[0] for r in con.execute("SELECT DISTINCT name FROM classes WHERE image_count>0 ORDER BY name")]

    def url_at(p):
        args = {k: v for k, v in cur.items() if v}
        return url_for("images", page=p, **args)

    return page("이미지", "images", IMAGES, rows=rows, total=total, pg=pg, pages=pages,
                cur=type("F", (), cur), dsets=dsets, clss=clss,
                prev_url=url_at(pg - 1), next_url=url_at(pg + 1))


# ── 홀 랜드마크 라벨 (labels/holes/*.json) ────────────────

LABELS_DIR = BASE_DIR / "labels" / "holes"
POINT_COLORS = {"corner_hinge": (217, 90, 30), "corner_latch": (200, 40, 40),
                "bolt_tl": (46, 94, 140), "bolt_tr": (46, 94, 140),
                "bolt_bl": (46, 94, 140), "bolt_br": (46, 94, 140)}

LABELS = """
<h1>홀 랜드마크 라벨 <span class="muted small">{{ rows|length }}건 표시 / 전체 {{ total }}건 — 15_label_holes.py로 수동 라벨링, 랜드마크 CNN 학습 데이터</span></h1>
<p class="small muted" style="margin:-8px 0 12px">
  점 색: <span style="color:#D95A1E">■ corner_hinge</span> ·
  <span style="color:#C82828">■ corner_latch</span> ·
  <span style="color:#2E5E8C">■ bolt ×4</span>
  — 썸네일 클릭 시 원본 크기로 라벨 확인 · 라벨링 기간 {{ first_at }} ~ {{ last_at }}
</p>
<form class="filters" method="get">
  <select name="cls" onchange="this.form.submit()">
    <option value="">클래스 전체</option>
    {% for c in clss %}<option {{ 'selected' if c == cur_cls }}>{{ c }}</option>{% endfor %}
  </select>
  <select name="src" onchange="this.form.submit()">
    <option value="">소스 전체</option>
    {% for s in srcs %}<option {{ 'selected' if s == cur_src }}>{{ s }}</option>{% endfor %}
  </select>
  <a class="small" href="{{ url_for('labels') }}">필터 초기화</a>
</form>
<div class="tbl" style="margin-bottom:16px"><table>
<tr><th>클래스</th><th class="n">라벨 수</th><th>첫 라벨</th><th>마지막 라벨</th></tr>
{% for c in per_cls %}
<tr><td>{{ c['cls'] }}</td><td class="n">{{ c['n'] }}</td>
    <td class="muted">{{ c['first'] }}</td><td class="muted">{{ c['last'] }}</td></tr>
{% endfor %}
</table></div>
<div class="grid">
{% for r in rows %}
  <div class="thumb"><a href="{{ url_for('label_image', stem=r['stem']) }}" target="_blank">
    <img src="{{ url_for('label_thumb', stem=r['stem']) }}" loading="lazy" alt="{{ r['stem'] }}"></a>
    <div class="cap">{{ r['cls'] }} · {{ r['idx'] }} <span class="muted">({{ r['src'] }})</span><br>
      {{ r['labeled_at'] }}{% if r['hidden'] %}<br><span class="warn">비가시: {{ r['hidden'] }}</span>{% endif %}{% if r['missing'] %}<br><span class="warn">미기입: {{ r['missing'] }}</span>{% endif %}</div></div>
{% endfor %}
</div>
"""


def load_labels():
    out = []
    for p in sorted(LABELS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        hidden = [k for k, v in (d.get("visible") or {}).items() if not v]
        missing = [k for k, v in (d.get("points") or {}).items() if not v]
        out.append({"stem": p.stem, "cls": d.get("cls", "?"), "src": d.get("src", "?"),
                    "idx": d.get("idx", ""), "image": d.get("image", ""),
                    "labeled_at": d.get("labeled_at", ""), "points": d.get("points", {}),
                    "visible": d.get("visible", {}), "hidden": ", ".join(hidden),
                    "missing": ", ".join(missing)})
    return out


@app.route("/labels")
def labels():
    rows = load_labels()
    total = len(rows)
    clss = sorted({r["cls"] for r in rows})
    srcs = sorted({r["src"] for r in rows})
    per_cls = []
    for c in clss:
        ts = sorted(r["labeled_at"] for r in rows if r["cls"] == c)
        per_cls.append({"cls": c, "n": len(ts), "first": ts[0], "last": ts[-1]})
    times = sorted(r["labeled_at"] for r in rows if r["labeled_at"])
    cur_cls = request.args.get("cls", "")
    cur_src = request.args.get("src", "")
    if cur_cls:
        rows = [r for r in rows if r["cls"] == cur_cls]
    if cur_src:
        rows = [r for r in rows if r["src"] == cur_src]
    return page("라벨", "labels", LABELS, rows=rows, total=total, clss=clss, srcs=srcs,
                cur_cls=cur_cls, cur_src=cur_src, per_cls=per_cls,
                first_at=times[0] if times else "—", last_at=times[-1] if times else "—")


def _label_record(stem):
    if not stem.replace("_", "").replace("-", "").isalnum():
        abort(404)
    p = LABELS_DIR / f"{stem}.json"
    if not p.exists():
        abort(404)
    d = json.loads(p.read_text())
    src_img = BASE_DIR / d.get("image", "")
    if not src_img.exists():
        abort(404, "라벨의 원본 이미지가 로컬에 없습니다.")
    return d, src_img


def _draw_label(d, src_img, max_side=None):
    """라벨 점을 이미지 위에 그려서 PIL Image 반환. 비가시 점은 테두리만."""
    im = Image.open(src_img).convert("RGB")
    w0 = im.size[0]
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    scale = im.size[0] / w0
    dr = ImageDraw.Draw(im)
    r = max(4, im.size[0] // 130)
    for name, xy in (d.get("points") or {}).items():
        if not xy:                       # 좌표 없음(null) = 안 찍은 점 — 건너뜀
            continue
        x, y = xy[0] * scale, xy[1] * scale
        color = POINT_COLORS.get(name, (100, 100, 100))
        vis = (d.get("visible") or {}).get(name, True)
        if vis:
            dr.ellipse([x - r, y - r, x + r, y + r], outline=color, width=max(2, r // 3))
            dr.line([x - r * 2, y, x + r * 2, y], fill=color, width=1)
            dr.line([x, y - r * 2, x, y + r * 2], fill=color, width=1)
        else:
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(150, 150, 150), width=1)
    return im


@app.route("/labels/thumb/<stem>")
def label_thumb(stem):
    THUMB_DIR.mkdir(exist_ok=True)
    cached = THUMB_DIR / f"label_{stem}.jpg"
    if not cached.exists():
        d, src_img = _label_record(stem)
        _draw_label(d, src_img, max_side=420).save(cached, "JPEG", quality=82)
    return send_file(cached, max_age=86400)


@app.route("/labels/img/<stem>")
def label_image(stem):
    THUMB_DIR.mkdir(exist_ok=True)
    cached = THUMB_DIR / f"label_full_{stem}.jpg"
    if not cached.exists():
        d, src_img = _label_record(stem)
        _draw_label(d, src_img).save(cached, "JPEG", quality=90)
    return send_file(cached, max_age=86400)


# ── 학습·평가 이력 ───────────────────────────────────────

TRAINING = """
<h1>학습·평가 이력</h1>

<h2>학습 세션 {{ '(' ~ trains|length ~ '건)' }}</h2>
{% for t in trains %}
<div class="box" style="margin-bottom:12px">
  <b>#{{ t['id'] }} {{ t['model'] }}</b> <span class="pill">{{ t['status'] }}</span>
  <div class="small" style="margin:4px 0 2px">
    학습 데이터: <b>{{ t['dataset'] }}</b> — train {{ '{:,}'.format(t['train_count']) if t['train_count'] else '?' }}장
    / 검증 {{ '{:,}'.format(t['test_count']) if t['test_count'] else '?' }}장
    {% if t['train_ratio'] %}(분할 비율 {{ t['train_ratio'] }}){% endif %}
    {% if t['split_indices_path'] %}· 분할 기록: <span class="muted">{{ t['split_indices_path'] }}</span>{% endif %}
  </div>
  <div class="small muted" style="margin:0 0 8px">
    {{ t['optimizer'] }} lr={{ t['learning_rate'] }} batch={{ t['batch_size'] }}
    · {{ t['actual_epochs'] or '?' }}/{{ t['max_epochs'] }} epochs
    {% if t['best_val_accuracy'] %}· best val {{ '%.2f'|format(t['best_val_accuracy']) }}% (epoch {{ t['best_epoch'] }}){% endif %}
    {% if t['total_time_sec'] %}· {{ '%.0f'|format(t['total_time_sec']/60) }}분{% endif %}
    · {{ t['started_at'] }}
  </div>
  {{ charts[t['id']]|safe }}
</div>
{% endfor %}

<h2>평가 결과 {{ '(' ~ evals|length ~ '건)' }} <span class="muted small">행을 누르면 어떤 데이터로 평가했는지 상세가 열립니다</span></h2>
<div class="tbl tall"><table>
<tr><th>일시</th><th>모델</th><th>학습세션</th><th>평가 데이터셋</th><th>유형</th><th class="n">정확도</th><th class="n">F1(macro)</th><th class="n">표본</th><th>리포트</th></tr>
{% for e in evals %}
<tr class="click" onclick="var d=document.getElementById('ev{{ e['id'] }}'); d.hidden=!d.hidden">
    <td class="muted">{{ e['evaluated_at'] }}</td><td>{{ e['model'] }}</td>
    <td>{% if e['session_id'] %}<span class="pill">#{{ e['session_id'] }}</span>{% else %}<span class="muted small">DB 도입 전</span>{% endif %}</td>
    <td>{{ e['dataset'] }}</td>
    <td class="muted">{{ e['eval_type'] }}</td>
    <td class="n"><b>{{ '%.1f'|format(e['accuracy']) }}%</b></td>
    <td class="n">{{ '%.1f'|format(e['f1_macro']) if e['f1_macro'] is not none else '—' }}</td>
    <td class="n">{{ e['correct'] }}/{{ e['total_samples'] }}</td>
    <td class="small muted" title="{{ e['report_path'] or '' }}">{{ (e['report_path'] or '').split('/')[-1] }}</td></tr>
<tr id="ev{{ e['id'] }}" hidden><td colspan="9" style="white-space:normal">
    {% for t in rpt[e['id']]['tables'] %}
    <div class="small" style="margin:8px 0 4px"><b>{{ t['name'] }}</b> — 클래스별 결과 <span class="muted">(리포트 파일 기준)</span></div>
    <div class="tbl" style="margin-bottom:8px"><table>
    <tr><th>클래스</th><th class="n">n</th><th class="n">판정</th><th class="n">정답</th><th class="n">정확도</th>
        <th class="n">D med</th><th class="n">CAD</th><th class="n">차이</th><th>오판 내역</th></tr>
    {% for c in t['classes'] %}
    <tr><td>{{ c['cls'] }}</td><td class="n">{{ c['n'] }}</td><td class="n">{{ c['judged'] }}</td>
        <td class="n">{{ c['correct'] }}</td>
        <td class="n">{{ '%.1f%%'|format(c['acc']) if c['acc'] is not none else '—' }}</td>
        <td class="n">{{ '%.1f'|format(c['d_med']) if c['d_med'] is not none else '—' }}</td>
        <td class="n">{{ c['cad'] if c['cad'] is not none else '—' }}</td>
        <td class="n">{{ '%+.0f'|format(c['diff']) if c['diff'] is not none else '—' }}</td>
        <td class="warn small">{{ c['wrong'] }}</td></tr>
    {% endfor %}
    </table></div>
    {% endfor %}
    {% if rpt[e['id']]['pose'] %}
    <div class="small" style="margin:8px 0 4px"><b>세션별 자세 추정</b>
      <span class="muted">(리포트 파일 기준 — RMS·std 단위 mm, 각도 °)</span></div>
    <div class="tbl" style="margin-bottom:8px"><table>
    <tr><th>세션</th><th>클래스</th><th class="n">프레임</th><th class="n">사용</th>
        <th class="n">RMS med</th><th class="n">θ</th><th class="n">tilt</th><th class="n">std z</th><th>집계</th></tr>
    {% for p in rpt[e['id']]['pose'] %}
    <tr><td class="small">{{ p['session'] }}</td><td>{{ p['cls'] }}</td>
        <td class="n">{{ p['n'] }}</td><td class="n">{{ p['used'] }}</td>
        <td class="n">{{ '%.2f'|format(p['rms']) if p['rms'] is not none else '—' }}</td>
        <td class="n">{{ '%.2f'|format(p['theta']) if p['theta'] is not none else '—' }}</td>
        <td class="n">{{ '%.2f'|format(p['tilt']) if p['tilt'] is not none else '—' }}</td>
        <td class="n">{{ '%.2f'|format(p['stdz']) if p['stdz'] is not none else '—' }}</td>
        <td>{% if p['ok'] %}<span class="ok">ok</span>{% else %}<span class="warn">불충분</span>{% endif %}</td></tr>
    {% endfor %}
    </table></div>
    {% endif %}
    {% if rpt[e['id']]['held'] %}
    <div class="small" style="margin:8px 0 6px"><b>보류 이미지 {{ rpt[e['id']]['held']|length }}건</b>
      <span class="muted">— no_corner=모서리(힌지·래치) 홀 미검출, no_frame=볼트 4점 프레임 피팅 실패,
      latch_offset=래치 홀이 예상 범위 밖, near_border=홀이 화면 가장자리.
      같은 report_path로 재평가하면 파일이 덮어써져 최신 실행 내용이 보입니다.</span></div>
    <div class="grid" style="margin-bottom:10px">
      {% for h in rpt[e['id']]['held'] %}
      <div class="thumb">
        {% if h['img_id'] %}<a href="{{ url_for('raw_image', img_id=h['img_id']) }}" target="_blank">
          <img src="{{ url_for('thumb', img_id=h['img_id']) }}" loading="lazy" alt=""></a>
        {% else %}<div style="height:110px;display:flex;align-items:center;justify-content:center;background:var(--paper)" class="muted small">로컬 파일 없음</div>{% endif %}
        <div class="cap"><span class="warn">{{ h['gate'] }}</span> · {{ h['cls'] }}<br>{{ h['label'] }}</div>
      </div>{% endfor %}
    </div>
    {% endif %}
    <pre class="notes" style="margin:4px 0">{{ details[e['id']] }}</pre></td></tr>
{% endfor %}
</table></div>
"""


def _load_cad_d():
    """hole_classifier.py의 CAD_D 상수를 import 없이 읽는다 (torch 로딩 회피)."""
    try:
        tree = ast.parse((BASE_DIR / "hole_classifier.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "CAD_D" for t in node.targets):
                return ast.literal_eval(node.value)
    except (OSError, ValueError, SyntaxError):
        pass
    return {}


CAD_D = _load_cad_d()


def _class_stats(rows):
    """이미지별 평가 행 → 17번 터미널 출력과 같은 클래스별 집계."""
    by = {}
    for r in rows:
        cls = r.get("cls")
        if cls not in CAD_D:            # Unknown 등 라벨 없는 행은 점수 집계와 동일하게 제외
            continue
        st = by.setdefault(cls, {"n": 0, "judged": 0, "correct": 0, "Ds": [], "wrong": {}})
        st["n"] += 1
        if r.get("pred"):
            st["judged"] += 1
            if r["pred"] == cls:
                st["correct"] += 1
            else:
                st["wrong"][r["pred"]] = st["wrong"].get(r["pred"], 0) + 1
        if r.get("gate") == "ok" and r.get("D_mm"):
            st["Ds"].append(r["D_mm"])
    out = []
    for cls in sorted(by):
        st = by[cls]
        med = statistics.median(st["Ds"]) if st["Ds"] else None
        cad = CAD_D.get(cls)
        out.append({"cls": cls, "n": st["n"], "judged": st["judged"], "correct": st["correct"],
                    "acc": 100 * st["correct"] / st["judged"] if st["judged"] else None,
                    "d_med": med, "cad": cad,
                    "diff": med - cad if med is not None and cad is not None else None,
                    "wrong": ", ".join(f"{k}×{v}" for k, v in sorted(st["wrong"].items()))})
    return out


def _pose_rows(data):
    """pos_pipeline/03 리포트(sessions 구조) → 세션별 자세 추정 표."""
    rows = data.get("sessions")
    if not (isinstance(rows, list) and rows and isinstance(rows[0], dict) and "agg" in rows[0]):
        return []
    out = []
    for s in rows:
        agg = s.get("agg") or {}
        std = agg.get("std") or {}
        out.append({"session": s.get("session", "?"), "cls": s.get("cls", "?"),
                    "n": s.get("n"), "used": s.get("n_used"), "ok": agg.get("ok"),
                    "rms": agg.get("rms_med"), "theta": agg.get("theta_deg"),
                    "tilt": agg.get("tilt_deg"), "stdz": std.get("z")})
    return out


def eval_report_view(con, report_path):
    """평가 리포트 JSON → 세트별 클래스 집계표 + 포즈 세션표 + 보류 이미지(DB 썸네일 매칭)."""
    empty = {"tables": [], "held": [], "pose": []}
    if not report_path or not (BASE_DIR / report_path).exists():
        return empty
    try:
        data = json.loads((BASE_DIR / report_path).read_text())
    except ValueError:
        return empty
    if not isinstance(data, dict):
        return empty
    tables, held = [], []
    for set_name, node in data.items():
        rows = node.get("rows") if isinstance(node, dict) else None
        if not isinstance(rows, list):
            continue
        stats = _class_stats(r for r in rows if isinstance(r, dict))
        if stats:
            tables.append({"name": set_name, "classes": stats})
        held.extend(r for r in rows if isinstance(r, dict)
                    and r.get("gate") not in (None, "ok"))
    out_held = []
    for r in held[:80]:
        img = r.get("image", "")
        first, _, rel = img.partition("/")
        row = con.execute(
            """SELECT i.id, i.synced_local FROM images i JOIN classes c ON c.id=i.class_id
               JOIN datasets d ON d.id=c.dataset_id
               WHERE i.rgb_path=? AND (d.base_path=? OR d.base_path LIKE 'nas:%')""",
            (rel, first + "/")).fetchone()
        parts = img.split("/")
        if len(parts) >= 4 and len(parts[1]) == 8 and parts[1].isdigit():
            label = f"{parts[1]}/{parts[-2]}/{parts[-1]}"   # 날짜/세션/파일 (미러 구조)
        else:
            label = "/".join(parts[-2:])
        out_held.append({"image": img, "gate": r.get("gate"), "cls": r.get("cls"),
                         "label": label,
                         "img_id": row["id"] if row and row["synced_local"] else None})
    return {"tables": tables, "held": out_held, "pose": _pose_rows(data)}


def curve_svg(metrics):
    """에폭별 loss·accuracy 를 간단한 인라인 SVG 두 개로."""
    if not metrics:
        return '<span class="muted small">에폭 기록 없음</span>'
    W, H, PAD = 360, 130, 26

    def poly(pts, lo, hi, color, dash=""):
        if hi <= lo:
            hi = lo + 1e-9
        xs = [PAD + (W - PAD - 6) * i / max(1, len(pts) - 1) for i in range(len(pts))]
        ys = [H - PAD / 2 - (H - PAD) * (v - lo) / (hi - lo) for v in pts]
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        return (f'<polyline points="{p}" fill="none" stroke="{color}" '
                f'stroke-width="1.6" stroke-dasharray="{dash}"/>')

    tl = [m["train_loss"] for m in metrics]
    vl = [m["val_loss"] for m in metrics if m["val_loss"] is not None]
    va = [m["val_accuracy"] for m in metrics if m["val_accuracy"] is not None]
    lo, hi = 0, max(tl + (vl or [0]))
    svg1 = (f'<svg width="{W}" height="{H}" role="img" aria-label="loss curve">'
            f'<text x="{PAD}" y="12">loss (실선 train / 점선 val) · {len(tl)} epochs</text>'
            + poly(tl, lo, hi, "var(--accent)") + (poly(vl, lo, hi, "var(--flag)", "4 3") if vl else "")
            + f'<text x="2" y="{H-8}">{lo:.1f}</text><text x="2" y="24">{hi:.2f}</text></svg>')
    svg2 = ""
    if va:
        lo2, hi2 = min(va), 100
        svg2 = (f'<svg width="{W}" height="{H}" role="img" aria-label="accuracy curve">'
                f'<text x="{PAD}" y="12">val accuracy (%) · best {max(va):.2f}</text>'
                + poly(va, lo2, hi2, "var(--good)")
                + f'<text x="2" y="{H-8}">{lo2:.0f}</text><text x="2" y="24">{hi2}</text></svg>')
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap">{svg1}{svg2}</div>'


@app.route("/training")
def training():
    con = db()
    trains = con.execute(
        """SELECT t.*, m.name model, d.name dataset FROM training_sessions t
           JOIN models m ON m.id=t.model_id JOIN datasets d ON d.id=t.dataset_id
           ORDER BY t.started_at DESC""").fetchall()
    charts = {t["id"]: curve_svg(con.execute(
        "SELECT * FROM training_metrics WHERE session_id=? ORDER BY epoch",
        (t["id"],)).fetchall()) for t in trains}
    evals = con.execute(
        """SELECT e.*, m.name model, d.name dataset FROM evaluation_results e
           JOIN models m ON m.id=e.model_id JOIN datasets d ON d.id=e.dataset_id
           ORDER BY e.evaluated_at DESC""").fetchall()
    details, held = {}, {}
    for e in evals:
        try:
            details[e["id"]] = json.dumps(json.loads(e["per_class_results"]),
                                          ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            details[e["id"]] = e["per_class_results"] or "클래스별 상세 기록 없음"
        held[e["id"]] = eval_report_view(con, e["report_path"])
    return page("학습·평가", "training", TRAINING, trains=trains, charts=charts,
                evals=evals, details=details, rpt=held)


# ── 이미지 서빙 ──────────────────────────────────────────

def _img_row(img_id):
    r = db().execute(
        """SELECT i.*, d.base_path FROM images i JOIN classes c ON c.id=i.class_id
           JOIN datasets d ON d.id=c.dataset_id WHERE i.id=?""", (img_id,)).fetchone()
    if not r:
        abort(404)
    return r


@app.route("/img/<int:img_id>")
def raw_image(img_id):
    r = _img_row(img_id)
    p = image_abs_path(r["base_path"], r["rgb_path"])
    if not p or not p.exists():
        abort(404, "로컬에 파일이 없습니다 (NAS에만 존재).")
    return send_file(p)


@app.route("/thumb/<int:img_id>")
def thumb(img_id):
    THUMB_DIR.mkdir(exist_ok=True)
    cached = THUMB_DIR / f"{img_id}.jpg"
    if not cached.exists():
        r = _img_row(img_id)
        p = image_abs_path(r["base_path"], r["rgb_path"])
        if not p or not p.exists():
            abort(404)
        im = Image.open(p).convert("RGB")
        im.thumbnail((360, 360))
        im.save(cached, "JPEG", quality=82)
    return send_file(cached, max_age=86400)


# ── main ────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--readonly", action="store_true", help="열람 전용 (수정 버튼 비활성)")
    a = ap.parse_args()
    app.config["DB_PATH"] = a.db
    app.config["READONLY"] = a.readonly
    print(f"door_pipeline DB 웹 도구: http://localhost:{a.port}  (DB: {a.db}"
          f"{', 열람 전용' if a.readonly else ''})")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
