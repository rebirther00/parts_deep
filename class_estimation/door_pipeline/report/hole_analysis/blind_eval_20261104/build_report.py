"""블라인드 보강 평가(2026-11-04, 10/23~10/29 미검사 33세션) 리포트 생성 → report.html(아티팩트) + README.md + web/ 이미지.
입력: datasets_blind_new_20261104/manifest.json, 17 결과 json, 04 결과 json + per_session.json, DB(session_radar_check·capture_sessions·vlm_audits·evaluation_results),
      session_metrics_all.json(20_session_drift 전 세션), door_partno/report/blind_new_20261104.json, 샘플 몽타주·비교 이미지.
실행: python report/hole_analysis/blind_eval_20261104/build_report.py
"""
import json, sqlite3, collections, statistics, os, html, shutil, datetime as dt
import cv2
D = '/home/koceti/parts_deep/class_estimation/door_pipeline'
O = f'{D}/report/hole_analysis/blind_eval_20261104'
W = f'{O}/web'
RUN_VLM = '20261104_152517'
con = sqlite3.connect(f'{D}/db/door_pipeline.db')
man = json.load(open(f'{D}/datasets_blind_new_20261104/manifest.json')); sessions = man['sessions']
hole = list(json.load(open(f'{D}/attribute_models/hole_landmarks/eval_classifier_datasets_blind_new_20261104_collect_pixel.json')).values())[0]
cnn_ps = json.load(open(f'{O}/rgbe_noaux_448_seed916/per_session.json'))
cnn_all = json.load(open(f'{O}/rgbe_noaux_448_seed916/factory_eval_results.json'))
metrics = json.load(open(f'{O}/session_metrics_all.json'))
partno = json.load(open(f'{D}/../door_partno/report/blind_new_20261104.json'))
CAD = {'E23_door_LH_FRT': 458, 'E25_door_LH_FRT': 724, 'E30_door_LH_FRT': 765, 'E38_door_LH_FRT': 812, 'E25_door_LH_RR': 1037, 'E30_door_LH_RR': 1158, 'E38_door_LH_RR': 1352, 'E25_door_RH': 886, 'E30_E38_door_RH': 1087}
short = lambda c: c.replace('_door', '') if c else '-'

# ---------- per-session aggregation ----------
hp = collections.defaultdict(collections.Counter); hd = collections.defaultdict(list); hm = collections.defaultdict(list)
for r in hole['rows']:
    p = r['image'].split('/'); sd = f'{p[2]}/{p[4]}'      # (날짜/세션) 키 — 뷰의 클래스 폴더는 DB 정정 라벨이라 입력 폴더명과 다를 수 있다
    hp[sd][r['pred'] or 'abstain'] += 1; hd[sd].append(r['D_mm']); hm[sd].append(r['margin_mm'] if r['margin_mm'] is not None else 0)
vlm = {}
for sd, cls, conf, radar, anom, desc in con.execute("SELECT session_dir, vlm_class, vlm_class_conf, vlm_radar, vlm_anomalies, vlm_description FROM vlm_audits WHERE run_id=? AND source='collect' AND error IS NULL", (RUN_VLM,)):
    vlm[sd] = dict(cls=cls, conf=conf, radar=radar, anom=anom, desc=desc)
rows = []
for sd in sessions:
    day, cls_in, ss = sd.split('/')
    sid = con.execute('SELECT id FROM capture_sessions WHERE session_dir=?', (sd,)).fetchone()[0]
    cls = con.execute('SELECT c.name FROM images i JOIN classes c ON c.id=i.class_id WHERE i.session_id=? GROUP BY c.name ORDER BY COUNT(*) DESC LIMIT 1', (sid,)).fetchone()[0]
    rc = con.execute('SELECT auto_flag, n_judged, n_radar, n_none, score_med, crop_path FROM session_radar_check WHERE session_id=? ORDER BY id DESC LIMIT 1', (sid,)).fetchone()
    opt = con.execute('SELECT option_radar, option_source FROM capture_sessions WHERE id=?', (sid,)).fetchone()
    k = f'{day}/{ss}'; c = hp[k]; maj = c.most_common(1)[0]
    cp = cnn_ps[sd]
    rows.append(dict(sd=sd, cls_in=cls_in, day=f'{day[4:6]}/{day[6:8]}', time=f'{ss[2:4]}:{ss[4:6]}', cls=cls, n=sum(c.values()),
                     hole_maj=maj[0], hole_n=maj[1], hole_votes=dict(c), D=statistics.median(hd[k]), margin=min(hm[k]),
                     cnn_maj=cp['major'], cnn_n=cp['n_major'], cnn_votes=cp['votes'], cnn_p=cp['prob_med'],
                     radar=(dict(flag=rc[0], judged=rc[1], n1=rc[2], n0=rc[3], score=rc[4], crop=rc[5]) if rc else None), opt=opt,
                     vlm=vlm.get(sd)))
n_frames = sum(r['n'] for r in rows)
cnn_pairs = collections.Counter()
for sd, cp in cnn_ps.items():
    for f in cp['frames']:
        if f['pred'] != cp['label']: cnn_pairs[(cp['label'], f['pred'])] += 1
cnn_wrong_n = sum(cnn_pairs.values())
cnn_pairs_txt = ', '.join(f"{short(a)}→{short(b)} {n}장" for (a, b), n in cnn_pairs.most_common())
hole_wrong_sess = [r for r in rows if r['hole_maj'] != r['cls']]
cnn_wrong_sess = [r for r in rows if r['cnn_maj'] != r['cls']]
radar_rows = [r for r in rows if r['radar']]
n_r1 = sum(1 for r in radar_rows if r['radar']['flag'] == 1); n_r0 = sum(1 for r in radar_rows if r['radar']['flag'] == 0)
rule = partno['results']['rule']

# ---------- cumulative (confirmed sessions radar agreement) ----------
agree = con.execute("""SELECT COUNT(*), SUM(CASE WHEN s.option_radar=rc.auto_flag THEN 1 ELSE 0 END) FROM session_radar_check rc JOIN capture_sessions s ON s.id=rc.session_id
                       WHERE s.option_source='user' AND rc.auto_flag IS NOT NULL""").fetchone()
evals10 = con.execute("SELECT total_samples, correct, accuracy, report_path, evaluated_at FROM evaluation_results WHERE model_id=10 AND eval_type='cross_domain' ORDER BY id").fetchall()
print('seed916 evals:', evals10)

# ---------- images ----------
def jpg(src, dst, width=None, q=82):
    im = cv2.imread(src)
    if width and im.shape[1] > width: im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, q]); return os.path.getsize(dst)
S = f'{D}/report/hole_analysis/samples_datasets_blind_new_20261104_collect_pixel'
sizes = {}
sizes['compare.jpg'] = jpg(f'{O}/compare/compare_3panel.jpg', f'{W}/compare.jpg', 2000)
sizes['hole_success.jpg'] = jpg(f'{S}/montage_success.png', f'{W}/hole_success.jpg', 1600)
if os.path.exists(f'{S}/montage_failure.png'):
    sizes['hole_failure.jpg'] = jpg(f'{S}/montage_failure.png', f'{W}/hole_failure.jpg', 1600)
    failure_fig = '<figure><img src="hole_failure.jpg" alt="오판 샘플 몽타주" loading="lazy"><figcaption>오판 샘플.</figcaption></figure>'
else:
    if os.path.exists(f'{W}/hole_failure.jpg'): os.remove(f'{W}/hole_failure.jpg')
    failure_fig = '<figure><img src="compare.jpg" alt="정정 세션과 같은 날 E38·E30 도어 비교" loading="lazy"><figcaption>정정 라벨 기준 오판 0. 정정 전 오판 20장은 전부 위 비교 사진 왼쪽 세션이었다.</figcaption></figure>'
sizes['cnn_wrong.jpg'] = jpg(f'{O}/rgbe_noaux_448_seed916/factory_wrong.png', f'{W}/cnn_wrong.jpg', 1600)
shutil.copy(f'{D}/attribute_models/hole_landmarks/eval_classifier_datasets_blind_new_20261104_collect_pixel_confusion_matrix.png', f'{W}/hole_cm.png')
shutil.copy(f'{O}/rgbe_noaux_448_seed916/factory_confusion_matrix.png', f'{W}/cnn_cm.png')
for r in radar_rows:
    name = r['sd'].split('/')[0] + '_' + r['sd'].split('/')[2] + '.jpg'
    sizes['radar/' + name] = jpg(f"{D}/{r['radar']['crop']}", f'{W}/radar/{name}', 800, 80); r['radar']['web'] = 'radar/' + name
print('web images:', len(sizes), 'files', sum(sizes.values()) // 1024, 'KB')

# ---------- drift chart (inline SVG) ----------
ms = sorted([m for m in metrics if m['dev_pix_mm'] is not None], key=lambda m: m['session_dir'])
Wc, Hc, L, R, T, B = 960, 340, 56, 16, 18, 44
ymin, ymax = -70, 70
def X(i): return L + (Wc - L - R) * i / max(1, len(ms) - 1)
def Y(v): v = max(ymin, min(ymax, v)); return T + (Hc - T - B) * (ymax - v) / (ymax - ymin)
svg = [f'<svg class="chart" viewBox="0 0 {Wc} {Hc}" role="img" aria-label="세션별 D 편차(픽셀 경로 vs depth 경로), 2026-08-10~10-29 372세션">']
for v in (-60, -30, 0, 30, 60):
    svg.append(f'<line class="grid" x1="{L}" x2="{Wc-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="tick" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:+d}</text>')
for v in (-15, 15):
    svg.append(f'<line class="band" x1="{L}" x2="{Wc-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
svg.append(f'<text class="tick" x="{L-8}" y="{Y(15)+4:.1f}" text-anchor="end">+15</text><text class="tick" x="{L-8}" y="{Y(-15)+4:.1f}" text-anchor="end">−15</text>')
last = None
for i, m in enumerate(ms):
    mo = m['session_dir'][:6]
    if mo != last:
        last = mo; svg.append(f'<line class="grid" x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{T}" y2="{Hc-B}"/><text class="tick" x="{X(i)+4:.1f}" y="{Hc-B+16}">{mo[:4]}-{mo[4:]}</text>')
svg.append(f'<text class="tick" x="{Wc-R}" y="{Hc-B+16}" text-anchor="end">세션(시간순) → 10/29</text>')
svg.append(f'<text class="axis" transform="translate(14 {(T+Hc-B)/2:.0f}) rotate(-90)" text-anchor="middle">D 중앙값 − CAD (mm)</text>')
for i, m in enumerate(ms):
    if m['dev_mm'] is not None:
        svg.append(f'<circle class="s2" cx="{X(i):.1f}" cy="{Y(m["dev_mm"]):.1f}" r="2.6"><title>{m["session_dir"]} · depth 경로 {m["dev_mm"]:+.1f} mm (K {m["k"]:.3f}, z {m["z"]:.0f})</title></circle>')
for i, m in enumerate(ms):
    svg.append(f'<circle class="s1" cx="{X(i):.1f}" cy="{Y(m["dev_pix_mm"]):.1f}" r="3"><title>{m["session_dir"]} · 픽셀 경로 {m["dev_pix_mm"]:+.1f} mm (마진 {m["margin_pix_min"] or 0:.0f})</title></circle>')
    if m['session_dir'].endswith('s_075858'):
        svg.append(f'<circle class="ring" cx="{X(i):.1f}" cy="{Y(m["dev_pix_mm"]):.1f}" r="9"/><text class="label" x="{X(i)-12:.1f}" y="{Y(m["dev_pix_mm"])-14:.1f}" text-anchor="end">10/29 s_075858 정정 후 {m["dev_pix_mm"]:+.1f} (정정 전 +47)</text>')
svg.append(f'<g class="legend" transform="translate({L+8} {T+10})"><circle class="s1" cx="0" cy="0" r="4"/><text x="10" y="4">픽셀 폭 경로(현행 판정)</text><circle class="s2" cx="170" cy="0" r="3"/><text x="180" y="4">depth 경로(참고, 9/24 전 방식)</text></g>')
svg.append('</svg>')
drift_svg = ''.join(svg)
n_pix_out = sum(1 for m in ms if abs(m['dev_pix_mm']) > 15); n_dep_out = sum(1 for m in ms if m['dev_mm'] is not None and abs(m['dev_mm']) > 30)

# ---------- HTML ----------
E = html.escape
def pill(kind, text): return f'<span class="pill {kind}">{E(text)}</span>'
def row_html(i, r):
    hole_ok = r['hole_maj'] == r['cls']; cnn_ok = r['cnn_maj'] == r['cls']
    hole_cell = f"{pill('ok' if hole_ok else 'crit', short(r['hole_maj']))} <span class='mono'>{r['hole_n']}/{r['n']}</span>"
    cnn_cell = f"{pill('ok' if cnn_ok else 'crit', short(r['cnn_maj']))} <span class='mono'>{r['cnn_n']}/{r['n']} · p {r['cnn_p']:.2f}</span>"
    if r['radar']:
        f = r['radar']['flag']; rk = 'warn' if f is None else ('radar' if f == 1 else 'none')
        radar_cell = f"{pill(rk, '레이더 O' if f == 1 else ('레이더 X' if f == 0 else '미정'))} <span class='mono'>{r['radar']['n1']}:{r['radar']['n0']} · s {r['radar']['score']:.2f}</span>"
    else: radar_cell = '<span class="muted">FRT (옵션 없음)</span>'
    v = r['vlm']; vlm_cell = (f"{short(v['cls'])} <span class='muted'>({v['conf']})</span>" if v else '<span class="muted">—</span>')
    note = ''
    if r['sd'].endswith('s_075858'): note = '작업자 E30 입력, 11/4 육안 확인 후 E38 정정 · CNN 11/20 E30'
    elif r['sd'].endswith('s_095115'): note = 'CNN 11/12 E30 오판, 홀은 E38 12/12'
    elif not cnn_ok: note = 'CNN 다수결 오판'
    elif r['cnn_n'] < r['n']: note = f"CNN 소수표 {r['n']-r['cnn_n']}장"
    cls_tr = ' class="flag"' if (not hole_ok or not cnn_ok) else ''
    return (f"<tr{cls_tr}><td class='mono'>{i}</td><td class='mono'>{r['day']} {r['time']}</td><td>{E(short(r['cls']))}{(' <span class="muted">(입력 ' + E(short(r['cls_in'])) + ')</span>') if r['cls_in'] != r['cls'] else ''}</td><td class='mono'>{r['n']}</td>"
            f"<td>{hole_cell}</td><td class='mono'>{r['D']:.1f} <span class='muted'>/{CAD[r['cls']]}</span></td><td class='mono'>{r['margin']:.0f}</td>"
            f"<td>{cnn_cell}</td><td>{radar_cell}</td><td>{vlm_cell}</td><td class='note'>{E(note)}</td></tr>")
table = ''.join(row_html(i + 1, r) for i, r in enumerate(rows))
radar_cards = ''.join(
    f"<figure class='rcard'><img src='{r['radar']['web']}' alt='{E(r['sd'])} 브래킷 홀 투영 크롭' loading='lazy'>"
    f"<figcaption><span class='mono'>{r['day']} {r['time']}</span> {E(short(r['cls']))} {pill('radar' if r['radar']['flag']==1 else 'none', '레이더 O' if r['radar']['flag']==1 else '레이더 X')}"
    f"<span class='mono muted'> {r['radar']['n1']}:{r['radar']['n0']} · 점수 {r['radar']['score']:.2f}</span></figcaption></figure>" for r in radar_rows)
cnn_cls = ''.join(f"<tr><td>{E(short(k))}</td><td class='mono'>{('%.1f%%' % v) if v is not None else '—'}</td></tr>" for k, v in cnn_all['class_accuracies'].items() if v is not None)
hole_cls = collections.defaultdict(lambda: [0, 0])
for r in hole['rows']:
    hole_cls[r['cls']][0] += 1; hole_cls[r['cls']][1] += (r['pred'] == r['cls'])
hole_cls_html = ''.join(f"<tr><td>{E(short(k))}</td><td class='mono'>{v[1]}/{v[0]}</td><td class='mono'>{100*v[1]/v[0]:.1f}%</td></tr>" for k, v in sorted(hole_cls.items()))
rule_n, rule_ok = rule['judged'], rule['correct']

page = f"""<title>도어 판별 보강 평가 11/4</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: single column report, summary tiles first, then finding → session table → detail sections; cool steel neutrals, blue accent, orange = depth path */
:root{{--bg:#f3f5f8;--surface:#ffffff;--fg:#16212c;--muted:#5d6a78;--line:#d6dde6;--accent:#2a78d6;--s1:#2a78d6;--s2:#eb6834;--good:#1a7f37;--warn:#b45309;--crit:#c62828;--good-bg:#e6f4ea;--warn-bg:#fff4e0;--crit-bg:#fdecea;--radar-bg:#e8f0fb;--chip:#eaeef3;
 --font:"IBM Plex Sans KR",system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#12171c;--surface:#1a2129;--fg:#e6ebf0;--muted:#a2adb9;--line:#2c3640;--accent:#3987e5;--s1:#3987e5;--s2:#d95926;--good:#3fb950;--warn:#e3a008;--crit:#ef6b6b;--good-bg:#17301f;--warn-bg:#352a12;--crit-bg:#3a1f1f;--radar-bg:#1d2b3f;--chip:#242d37;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#12171c;--surface:#1a2129;--fg:#e6ebf0;--muted:#a2adb9;--line:#2c3640;--accent:#3987e5;--s1:#3987e5;--s2:#d95926;--good:#3fb950;--warn:#e3a008;--crit:#ef6b6b;--good-bg:#17301f;--warn-bg:#352a12;--crit-bg:#3a1f1f;--radar-bg:#1d2b3f;--chip:#242d37;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:var(--font);font-size:15px;line-height:1.55;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 48px}}
main{{max-width:1180px;margin:0 auto;display:grid;gap:28px}}
header.top{{padding-block:28px 6px;border-bottom:1px solid var(--line)}}
header.top h1{{font-size:26px;font-weight:700;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}}
header.top p{{margin:0;color:var(--muted);max-width:72ch}}
.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h2{{font-size:19px;margin:0 0 10px;font-weight:700;text-wrap:balance}} h3{{font-size:15.5px;margin:18px 0 6px;font-weight:600}}
section{{min-width:0}} p{{max-width:78ch}} .lede{{font-size:16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:grid;gap:4px;min-width:0}}
.tile .k{{font-size:12.5px;color:var(--muted);font-weight:500}} .tile .v{{font-family:var(--mono);font-size:26px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.1}}
.tile .s{{font-size:12.5px;color:var(--muted)}} .tile.good .v{{color:var(--good)}} .tile.warn .v{{color:var(--warn)}} .tile.crit .v{{color:var(--crit)}}
.pill{{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:999px;background:var(--chip);white-space:nowrap;vertical-align:middle}}
.pill.ok{{background:var(--good-bg);color:var(--good)}} .pill.crit{{background:var(--crit-bg);color:var(--crit)}} .pill.warn{{background:var(--warn-bg);color:var(--warn)}}
.pill.radar{{background:var(--radar-bg);color:var(--accent)}} .pill.none{{background:var(--chip);color:var(--muted)}}
.mono{{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:13.5px}} .muted{{color:var(--muted)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}}
th{{font-size:12px;color:var(--muted);font-weight:600;letter-spacing:.03em;background:var(--surface);position:sticky;top:0}}
tr.flag td{{background:var(--warn-bg)}} td.note{{white-space:normal;min-width:16ch;color:var(--muted)}}
table.small{{width:auto;min-width:280px}} tbody tr:last-child td{{border-bottom:0}}
figure{{margin:0;min-width:0}} figure img{{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#000}}
figcaption{{font-size:13px;color:var(--muted);margin-top:6px;line-height:1.45}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;align-items:start}}
.finding{{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--warn);border-radius:8px;padding:16px 18px;display:grid;gap:10px}}
.finding ul{{margin:0;padding-left:20px}} .finding li{{margin:2px 0}}
.rgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}} .rcard img{{aspect-ratio:960/286}}
.chart{{width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--line);border-radius:8px}}
.chart .grid{{stroke:var(--line);stroke-width:1}} .chart .band{{stroke:var(--warn);stroke-width:1;stroke-dasharray:4 4;opacity:.8}}
.chart .tick,.chart .axis,.chart .legend text,.chart .label{{fill:var(--muted);font-family:var(--mono);font-size:11px}} .chart .label{{fill:var(--fg);font-weight:600;font-family:var(--font)}}
.chart .s1{{fill:var(--s1);opacity:.9}} .chart .s2{{fill:var(--s2);opacity:.55}} .chart .ring{{fill:none;stroke:var(--crit);stroke-width:2}}
ol.steps{{padding-left:22px;margin:0}} ol.steps li{{margin:6px 0}} code{{font-family:var(--mono);font-size:12.5px;background:var(--chip);padding:1px 5px;border-radius:4px}}
details{{border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:8px 14px}} summary{{cursor:pointer;font-weight:600}}
footer{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}} footer ul{{margin:6px 0 0;padding-left:18px}}
a{{color:var(--accent)}} :focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (prefers-reduced-motion:reduce){{*{{scroll-behavior:auto}}}}
</style>
<main>
<header class="top">
 <div class="eyebrow">명성공업 도어 용접 라인 · 배포 모델 그대로 · 평가 2026-11-04</div>
 <h1>10/23~10/29 미검사 33세션 보강 평가와 누적 결과</h1>
 <p>작업자 라벨링이 끝난 10/29 까지의 수집분 가운데 아직 어느 평가에도 쓰이지 않은 33세션(638장)을 NAS 에서 받아, 배포 중인 홀 랜드마크 판별기(bracket 망, 픽셀 폭 D)와 CNN(seed916)을 그대로 돌리고, 레이더 옵션 규칙과 VLM 2차 의견까지 붙였다. 모델·상수 변경 없음.</p>
</header>

<section>
 <div class="tiles">
  <div class="tile"><div class="k">보강 평가 집합</div><div class="v">33 <span style="font-size:15px">세션</span></div><div class="s">{n_frames}장 · 10/23 10:53 ~ 10/29 13:43 · 8클래스</div></div>
  <div class="tile good"><div class="k">홀 판별기 판정률</div><div class="v">100%</div><div class="s">638/638 판정, 보류 0, 게이트 전부 ok</div></div>
  <div class="tile good"><div class="k">홀 판별기 판정 정확도</div><div class="v">{hole['acc_judged']:.1f}%</div><div class="s">{hole['correct']}/{hole['judged']} · 정정 전 라벨 기준 96.9% (20장 = 오라벨 세션 1개)</div></div>
  <div class="tile warn"><div class="k">CNN seed916 (폴백)</div><div class="v">{cnn_all['accuracy']:.1f}%</div><div class="s">{cnn_all['correct_samples']}/{cnn_all['total_samples']} · 세션 다수결 {33-len(cnn_wrong_sess)}/33 · 오판은 앞문 E30↔E38·E25</div></div>
  <div class="tile"><div class="k">레이더 옵션 자동 판정</div><div class="v">{len(radar_rows)} <span style="font-size:15px">세션</span></div><div class="s">레이더 O {n_r1} · X {n_r0} · 미정 0, 전부 만장일치 → 확정 대기</div></div>
  <div class="tile"><div class="k">품번 규칙 (확정 9세션)</div><div class="v">{100*rule_ok/rule_n:.1f}%</div><div class="s">{rule_ok}/{rule_n}장 · 나머지 24세션은 레이더 확정 후 채점</div></div>
 </div>
</section>

<section>
 <h2>핵심 발견: 10/29 07:58 세션은 작업자가 E30 앞문으로 입력했지만 E38 앞문이었다</h2>
 <div class="finding">
  <p class="lede" style="margin:0">홀 판별기가 라벨과 다르게 판정한 20장은 전부 <span class="mono">20261029/E30_door_LH_FRT/s_075858</span> 한 세션이었다. 코너 홀 간 거리는 픽셀 경로 중앙값 <b>812.3 mm</b> 로 E38 앞문 CAD(812)와 0.3 mm 차이, E30 CAD(765)와는 47 mm 차이였고, 같은 날 E38 라벨 세션(09:51)은 811.5, E30 라벨 세션(10:53)은 764.2 였다. <b>11/4 사용자가 육안으로 확인해 작업자 기종 오선택으로 판정했고, DB 라벨을 E38_door_LH_FRT 로 정정했다</b>(입력 원본 폴더명은 유지, 이력은 notes). 정정 후 홀 판별기는 638/638, 품번 규칙은 확정 9세션 161/161 이다.</p>
  <ul>
   <li><b>두 측정 경로가 같은 쪽.</b> depth 경로 D 824.7(K −7.2%), 픽셀 경로 812.3. 9/24 이후 372세션 중 픽셀 편차가 15 mm 를 넘은 세션은 이 하나였고, 정정 뒤에는 0 이다.</li>
   <li><b>CNN 은 11:9 로 갈렸다</b>(E30 11, E38 9, 중앙 확신 0.55). 정정 라벨 기준으로는 세션 다수결 오판이고, 외관으로는 두 앞문을 못 가른다는 뜻이다. VLM 도 FRT 클래스 변별은 신뢰 범위 밖이다.</li>
   <li>작업자 라벨 오류는 누적 11/379 세션(E25→E23 6, E30_RR→RH 3, E30_RR→E25_RR 1, 이번 E30→E38 FRT 1)이 됐고, 전부 홀 판별기가 잡아냈다.</li>
  </ul>
  <figure><img src="compare.jpg" alt="왼쪽: 정정 세션(입력 E30, 측정 812mm). 가운데: 같은 날 E38 라벨 도어(812mm). 오른쪽: 같은 날 E30 라벨 도어(764mm). 코너 홀 2점과 볼트 홀 4점에 검출 표시."><figcaption>왼쪽이 정정된 세션(입력 E30 → E38), 가운데가 같은 날 E38, 오른쪽이 같은 날 E30. 빨간 선이 힌지 코너 홀과 래치 코너 홀을 잇는 D 이며, 왼쪽과 가운데는 오른쪽 클램프까지의 여유가 같고 오른쪽 패널만 도어가 좁다.</figcaption></figure>
 </div>
</section>

<section>
 <h2>세션별 결과 (33세션)</h2>
 <p>라벨은 DB 정정 라벨이고 작업자 입력과 다르면 괄호에 적었다. 홀·CNN 은 세션 다수결과 표 수, D 는 픽셀 경로 중앙값과 CAD, 마진은 최근접 다른 클래스까지의 최소 여유다. 레이더는 규칙 자동 판정(레이더 표:없음 표, 어두운 홀 점수)이고 아직 사용자 확정 전이다. VLM 은 FRT 세션만 감사했다(RR/RH 는 레이더 확정 후 대상이 된다).</p>
 <div class="tablewrap"><table>
  <thead><tr><th>#</th><th>세션</th><th>라벨</th><th>장</th><th>홀 판별기</th><th>D / CAD</th><th>마진</th><th>CNN seed916</th><th>레이더 자동</th><th>VLM</th><th>비고</th></tr></thead>
  <tbody>{table}</tbody>
 </table></div>
</section>

<section>
 <h2>홀 랜드마크 판별기</h2>
 <div class="two">
  <div>
   <p>638장 전부 게이트를 통과해 판정했고(보류 0, pose_warn 0), 정정 라벨 기준 오판 0 이다. 마진 최소값은 FRT 37.6 mm(E25_LH_FRT, 인접 E30 까지 41 mm), RR/RH 는 45 mm 이상이다.</p>
   <table class="small"><thead><tr><th>클래스</th><th>정답/판정</th><th>정확도</th></tr></thead><tbody>{hole_cls_html}</tbody></table>
  </div>
  <figure><img src="hole_cm.png" alt="홀 판별기 혼동행렬: 대각선 외에는 E30_door_LH_FRT 행의 E38_door_LH_FRT 열 20 만 있다"><figcaption>혼동행렬(정답 행 × 예측 열, 정정 라벨). 비대각 성분 없음.</figcaption></figure>
 </div>
 <h3>세션별 D 편차, 8/10~10/29 전 372세션</h3>
 <p>현행 판정이 쓰는 픽셀 폭 경로(파랑)는 372세션 전부 ±15 mm 안에 있다(정정 전에는 s_075858 이 +47 로 유일한 이탈이었고, 그것이 오라벨의 단서였다). depth 경로(주황, 9/24 전 방식)는 10월 후반 K 가 −3~−7% 로 흔들려 {n_dep_out}세션이 ±30 mm 를 벗어났고, 그대로였다면 10/23 E25_RH 3세션·10/26 RH 2세션·10/27~28 E30_FRT 3세션이 오판됐다. 픽셀 전환이 막은 것이다.</p>
 {drift_svg}
 <div class="two" style="margin-top:14px">
  <figure><img src="hole_success.jpg" alt="홀 판별기 정상 판정 샘플 몽타주, 클래스별 검출점 표시" loading="lazy"><figcaption>정상 판정 샘플(클래스별). 파랑 볼트 홀 4, 빨강 힌지 코너, 주황 래치 코너.</figcaption></figure>
  {failure_fig}
 </div>
</section>

<section>
 <h2>CNN seed916 (홀 게이트 실패 시 폴백)</h2>
 <div class="two">
  <div>
   <p>장 단위 {cnn_all['accuracy']:.1f}%({cnn_all['correct_samples']}/{cnn_all['total_samples']}), 세션 다수결 {33-len(cnn_wrong_sess)}/33. 틀린 {cnn_wrong_n}장은 전부 앞문 사이다({cnn_pairs_txt}). 세션으로는 정정된 s_075858(E38, 11/20 을 E30)과 10/29 09:51 E38 세션(11/12 을 E30)이 다수결 오판이고, RR/RH 7클래스는 전부 맞았다. 폭 차 41~47 mm 인 앞문 세 종류를 외관으로 가르는 데 한계가 있다는 10/23 결론과 같다.</p>
   <table class="small"><thead><tr><th>클래스</th><th>정확도</th></tr></thead><tbody>{cnn_cls}</tbody></table>
  </div>
  <figure><img src="cnn_cm.png" alt="CNN 혼동행렬: 앞문 세 클래스 사이에서만 비대각 성분"><figcaption>CNN 혼동행렬. 비대각은 E25/E30/E38 앞문 사이에만 있다.</figcaption></figure>
 </div>
 <figure style="margin-top:14px"><img src="cnn_wrong.jpg" alt="CNN 오판 샘플 몽타주" loading="lazy"><figcaption>CNN 오판 샘플(04_evaluate_factory). 신뢰도 0.55~0.63 의 앞문 혼동.</figcaption></figure>
</section>

<section>
 <h2>레이더 옵션 (RR·RH 24세션, 확정 대기)</h2>
 <p>브래킷 홀 2개를 6점 도어 좌표계로 투영해 어두운 홀 유무를 보는 규칙이다. 24세션 모두 판정 프레임 18~20장이 만장일치였고, 레이더 O 가 3세션(10/26 E25_LH_RR 08:28, 10/26 E25_RH 09:22, 10/28 E30_LH_RR 07:29), X 가 21세션이다. 지금까지 사용자가 확정한 {agree[0]}세션에서 자동 판정은 {agree[1]}건 일치({100*agree[1]/agree[0]:.1f}%)였다. 아래 크롭을 보고 webapp /options 에서 확정하거나, 전부 맞으면 <code>python partno/radar_check.py --accept-auto</code> 로 일괄 승인한 뒤 <code>python ../door_partno/19_blind_eval.py --since 20261023 --tag new_20261104</code> 를 다시 돌리면 품번 14종 채점이 완성된다.</p>
 <div class="rgrid">{radar_cards}</div>
</section>

<section>
 <h2>지금까지의 블라인드 결과 (배포 모델, 학습·보정에 쓰지 않은 세션)</h2>
 <div class="tablewrap"><table>
  <thead><tr><th>평가일 · 집합</th><th>세션</th><th>장</th><th>홀 판별기 판정률 / 판정 정확도</th><th>품번 규칙(레이더 포함) 판정률 / 정확도</th><th>CNN seed916 장 단위</th><th>비고</th></tr></thead>
  <tbody>
   <tr><td>9/23 · split 미지정 35세션</td><td class="mono">35</td><td class="mono">663</td><td class="mono">99.8% / 100%</td><td class="mono">99.8% / 100%</td><td class="mono">99.4%</td><td class="note">seed916 정식 폴백 채택 근거</td></tr>
   <tr><td>10/2 · 9/29 이후 신규</td><td class="mono">46</td><td class="mono">904</td><td class="mono">100% / 100%</td><td class="mono">100% / 100%</td><td class="mono">94.6% (738장)</td><td class="note">검출기 블라인드</td></tr>
   <tr><td>10/12~10/21 · 10/3 이후 신규</td><td class="mono">82</td><td class="mono">1,632</td><td class="mono">99.5% / 100%</td><td class="mono">99.5% / 100%</td><td class="mono">99.3%</td><td class="note">검출기 블라인드</td></tr>
   <tr><td>10/21 · 10/12 이후 신규</td><td class="mono">51</td><td class="mono">1,013</td><td class="mono">99.2% / 100%</td><td class="mono">99.2% / 100%</td><td class="mono">—</td><td class="note">검출기 블라인드</td></tr>
   <tr><td>10/23 · split 미지정 전체</td><td class="mono">189</td><td class="mono">3,792</td><td class="mono">99.8% / 100%</td><td class="mono">99.8% / 100% (3,453장)</td><td class="mono">—</td><td class="note">10/23 재학습 검출기, 위 집합 포함</td></tr>
   <tr class="flag"><td>11/4 · 10/23 이후 미검사 (이번)</td><td class="mono">33</td><td class="mono">638</td><td class="mono">100% / {hole['acc_judged']:.0f}% (정정 전 96.9%)</td><td class="mono">100% / {100*rule_ok/rule_n:.0f}% (확정 9세션 {rule_n}장)</td><td class="mono">{cnn_all['accuracy']:.1f}%</td><td class="note">s_075858 오라벨 정정(E30→E38) · 레이더 24세션 확정 대기</td></tr>
  </tbody>
 </table></div>
 <div class="two" style="margin-top:14px">
  <div><h3>같이 보는 수치</h3>
   <ul style="margin:0;padding-left:20px">
    <li>CNN 세션 그룹 5-fold CV(sqrt 레시피, 186세션): 99.68%, 클래스 붕괴 없음(9/24).</li>
    <li>SX-10 실측 3장(E25_RH, E38_FRT, E30_RR): 코너 D 실물−CAD −0.4~−1.6 mm, 픽셀 D 와 +0.6~1.2 mm(10/8).</li>
    <li>VLM 독립 감사 339세션(11/3): 형상군 100%, 레이더 O/X 100%, 9클래스 94.1%(앞문 변별 불가는 전제).</li>
    <li>수집 전체: 8/10~10/29 유효 372세션 36,770장, 9형상군·품번 14종, 라벨 정정 11(전부 홀 판별기가 적발), 무효 7.</li>
   </ul>
  </div>
  <div><h3>읽는 법</h3>
   <p style="margin:0">홀 판별기의 "판정률"은 6점 검출과 기하 게이트를 통과한 비율이고 "판정 정확도"는 그 안에서 맞춘 비율이다. CNN 은 보류가 없어 장 단위 정확도 하나다. 세션 단위로 운용하므로 세션 다수결이 현장 성능에 가깝고, 이번 33세션은 홀 33/33, CNN {33-len(cnn_wrong_sess)}/33 이다.</p>
  </div>
 </div>
</section>

<section>
 <h2>다음 단계</h2>
 <ol class="steps">
  <li><b>레이더 24세션 확정</b>(webapp /options 또는 --accept-auto) 뒤 19_blind_eval 재실행으로 품번 채점 완성, 그 뒤 RR/RH 24세션 VLM 감사(≈$0.9).</li>
  <li><b>데이터 동결.</b> decisions.csv(10/29 기준, s_075858 정정 반영)와 이번 결과·<code>tools/build_blind_set.py</code> 를 커밋. 논문 설계(메모리)대로 전체 재분할·CV 는 그 다음이다.</li>
  <li>현장 PC 의 타임랩스 배포 확인(NAS 에 timelapse/ 폴더 아직 없음).</li>
 </ol>
</section>

<footer>
 근거 파일 (door_pipeline 기준)
 <ul>
  <li>집합: <code>datasets_blind_new_20261104/</code>(<code>tools/build_blind_set.py --since 20261023 --exclude datasets_blind_unsplit_20261023</code>), NAS pull 11/4 15:00</li>
  <li>홀: <code>attribute_models/hole_landmarks/eval_classifier_datasets_blind_new_20261104_collect_pixel.json</code> (DB evaluation_results #67), 드리프트 <code>report/hole_analysis/blind_eval_20261104/session_metrics_all.json</code></li>
  <li>CNN: <code>report/hole_analysis/blind_eval_20261104/rgbe_noaux_448_seed916/</code> (DB #68, per_session.json)</li>
  <li>레이더: <code>partno/radar_check.py</code> → DB session_radar_check, 크롭 <code>partno/artifacts/radar_crops/</code>; 품번 <code>../door_partno/report/blind_new_20261104.md</code></li>
  <li>VLM: <code>report/vlm_audit/audit_collect_20261104_152517.md</code> (9세션 $0.26)</li>
 </ul>
</footer>
</main>
"""
open(f'{O}/report.html', 'w', encoding='utf-8').write(page)
print('report.html', len(page) // 1024, 'KB')

# ---------- README.md ----------
md = [f"# 블라인드 보강 평가 2026-11-04 — 10/23~10/29 미검사 33세션(638장), 배포 모델 그대로", "",
      "집합: `tools/build_blind_set.py datasets_blind_new_20261104 --since 20261023 --exclude datasets_blind_unsplit_20261023` (NAS pull 11/4, 세션당 ≤20장). 모델: 홀 bracket 망(9/23)+픽셀 폭 D, CNN seed916. 리포트 `report.html`(아티팩트), 생성 `build_report.py`.", "",
      "| 항목 | 결과 |", "|---|---|",
      f"| 홀 판별기 | 판정 {hole['judged']}/{hole['n']}, 정확도 {hole['acc_judged']:.1f}% ({hole['correct']}/{hole['judged']}) — 정정 전 라벨 기준 96.9%: 오판 20장 = `20261029/E30_door_LH_FRT/s_075858`(픽셀 D 812.3 = E38 CAD) → 11/4 사용자 육안 확인, E38_door_LH_FRT 로 정정 |",
      f"| CNN seed916 | {cnn_all['accuracy']:.1f}% ({cnn_all['correct_samples']}/{cnn_all['total_samples']}), 세션 다수결 {33-len(cnn_wrong_sess)}/33 — 오판 전부 앞문({cnn_pairs_txt}); s_075858(E38 정정) 11/20·s_095115(E38) 11/12 을 E30 으로 |",
      f"| 레이더 자동 | RR/RH {len(radar_rows)}세션: O {n_r1} / X {n_r0} / 미정 0, 전부 만장일치, 사용자 확정 대기 (확정 누적 {agree[0]}세션 일치 {agree[1]}) |",
      f"| 품번 규칙 | 확정 9세션 {rule_n}장 {rule_ok}/{rule_n} ({100*rule_ok/rule_n:.1f}%) |",
      f"| 드리프트 | 372세션 픽셀 편차 >15mm {n_pix_out}세션(정정 전 s_075858 +47 이 유일) vs depth 경로 >30mm {n_dep_out}세션(10월 후반 K −3~−7%) |",
      f"| VLM | FRT 9세션 형상군 9/9, 클래스 3/9(앞문 변별 불가) |", "",
      "## 세션별", "", "| 세션 | 라벨 | 장 | 홀 | D(픽셀) | 마진 | CNN | 레이더 자동 | 비고 |", "|---|---|---:|---|---:|---:|---|---|---|"]
for r in rows:
    rd = (f"{'O' if r['radar']['flag']==1 else 'X' if r['radar']['flag']==0 else '?'} {r['radar']['n1']}:{r['radar']['n0']} s{r['radar']['score']:.2f}" if r['radar'] else 'FRT')
    note = '작업자 E30 입력→E38 정정(11/4)' if r['sd'].endswith('s_075858') else ('CNN 오판' if r['cnn_maj'] != r['cls'] else '')
    md.append(f"| {r['sd']} | {short(r['cls'])}{' (입력 '+short(r['cls_in'])+')' if r['cls_in']!=r['cls'] else ''} | {r['n']} | {short(r['hole_maj'])} {r['hole_n']}/{r['n']} | {r['D']:.1f}/{CAD[r['cls']]} | {r['margin']:.0f} | {short(r['cnn_maj'])} {r['cnn_n']}/{r['n']} p{r['cnn_p']:.2f} | {rd} | {note} |")
md += ["", "## 다음 단계", "1. 레이더 24세션 확정(webapp /options 또는 `python partno/radar_check.py --accept-auto`) → `python ../door_partno/19_blind_eval.py --since 20261023 --tag new_20261104` 재채점 → RR/RH VLM 감사",
       "2. 데이터 동결 커밋(decisions.csv, 이번 결과, tools/build_blind_set.py)"]
open(f'{O}/README.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('README.md written;', 'hole wrong sessions', [r['sd'] for r in hole_wrong_sess], 'cnn wrong', [r['sd'] for r in cnn_wrong_sess])
