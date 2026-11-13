"""Blind second-reading kit (config/extension_v16_protocol.json, B2).

Writes reports/blind_reading_v16/kit/ (index.html + images; this folder goes to the reader) and
reports/blind_reading_v16/key.json (answer key; stays with the corresponding author). The reader sees one image per item
under a neutral number and labelled reference images of the same camera state; no operator input, corrected label,
measured D, model output, session name or date is in the kit.

  build   make the kit (refuses to overwrite)
  score   compare a returned answers CSV with the key:  score reports/blind_reading_v16/answers_<name>.csv
"""
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from door8.core import ROOT, SOURCE, now, read_json, sha256, snapshot

OUT = ROOT / 'reports/blind_reading_v16'; KIT = OUT / 'kit'
PROTOCOL = ROOT / 'config/extension_v16_protocol.json'
CROP = (200, 1780, 60, 1190); SCALE = 0.75
BUMPED = ('20260921/E38_door_LH_RR/s_155719', '20260922/E38_door_LH_RR/s_080559')
ERRORS = {'20260903/E30_door_LH_RR/s_151736': 'E25_door_LH_RR', '20260903/E30_door_LH_RR/s_183159': 'E30_E38_door_RH',
          '20260905/E30_door_LH_RR/s_123451': 'E30_E38_door_RH', '20260914/E30_door_LH_RR/s_170035': 'E30_E38_door_RH',
          '20261029/E30_door_LH_FRT/s_075858': 'E38_door_LH_FRT'}                      # session -> corrected label
DISPUTED = ['20261022/E25_door_LH_FRT/s_084658', '20261022/E25_door_LH_FRT/s_123253']
NAME = {'E25_door_LH_FRT': 'E25 좌측 앞 (LH FRT)', 'E30_door_LH_FRT': 'E30 좌측 앞 (LH FRT)', 'E38_door_LH_FRT': 'E38 좌측 앞 (LH FRT)',
        'E25_door_LH_RR': 'E25 좌측 뒤 (LH RR)', 'E30_door_LH_RR': 'E30 좌측 뒤 (LH RR)', 'E38_door_LH_RR': 'E38 좌측 뒤 (LH RR)',
        'E25_door_RH': 'E25 우측 (RH)', 'E30_E38_door_RH': 'E30/E38 우측 (RH)'}
ORDER = list(NAME)


def state(session_dir):
    return 'before' if session_dir[:8] < '20260921' else 'after'


def save(src, dst):
    x0, x1, y0, y1 = CROP; im = cv2.imread(str(src))[y0:y1, x0:x1]
    cv2.imwrite(str(dst), cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])


def middle(paths):
    paths = sorted(paths); return paths[len(paths) // 2]


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    KIT.mkdir(parents=True); _, _, frames, _ = snapshot(); by = {}
    for r in frames:
        by.setdefault(r['group_id'], []).append(r)
    cls = {g: rr[0]['class_name'] for g, rr in by.items()}
    refs = {}
    for st in ('before', 'after'):
        for c in ORDER:
            g = sorted(x for x in by if cls[x] == c and state(x) == st and x not in BUMPED and x not in DISPUTED)[0]
            refs[(st, c)] = g; save(middle(r['rgb'] for r in by[g]), KIT / f'ref_{st}_{ORDER.index(c) + 1}.jpg')
    rng = random.Random(20261112); taken = set(refs.values()) | set(DISPUTED) | set(BUMPED); decoys = []
    for c in ORDER:
        pool = sorted(x for x in by if cls[x] == c and x not in taken); decoys += rng.sample(pool, 2)
    items = [dict(session=g, label=lab, kind='input_error') for g, lab in ERRORS.items()] \
        + [dict(session=g, label=cls[g], kind='cnn_disputed') for g in DISPUTED] + [dict(session=g, label=cls[g], kind='decoy') for g in decoys]
    rng.shuffle(items); base = (SOURCE / 'datasets_factory_collect').resolve()
    for n, it in enumerate(items, 1):
        g = it['session']
        src = middle(r['rgb'] for r in by[g]) if g in by else middle(str(p) for p in (base / g).glob('rgb_*.png'))
        it.update(item=n, state=state(g), image=Path(src).name, image_sha256=sha256(src)); save(src, KIT / f'item_{n:02d}.jpg')
    (KIT / 'index.html').write_text(page([dict(item=it['item'], state=it['state']) for it in items]), encoding='utf-8')
    key = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), crop=CROP, scale=SCALE,
               references={f'{st}:{c}': g for (st, c), g in refs.items()}, items=items)
    (OUT / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(items), 'items;', dict((k, sum(i['kind'] == k for i in items)) for k in ('input_error', 'cnn_disputed', 'decoy')))


def page(items):
    data = json.dumps(items); names = json.dumps([NAME[c] for c in ORDER], ensure_ascii=False)
    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>도어 형상군 판독</title>
<style>
:root{--bg:#f6f6f4;--ink:#1c1c1a;--mute:#666;--line:#cfcfc9;--acc:#1f5fbf;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Noto Sans KR",sans-serif}
header{padding:12px 16px;background:var(--card);border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header b{font-size:17px}main{padding:12px 16px;max-width:1500px;margin:auto}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pane{background:var(--card);border:1px solid var(--line);padding:8px}
.pane h3{margin:0 0 6px;font-size:14px;color:var(--mute);font-weight:600}.stack{position:relative}.stack img{width:100%;display:block}
#over{position:absolute;inset:0;opacity:0;pointer-events:none}
.opts{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px}
button{font:inherit;padding:8px 10px;border:1px solid var(--line);background:var(--card);cursor:pointer;border-radius:4px;text-align:left}
button.sel{border-color:var(--acc);background:#e8f0fd;font-weight:600}button.view{outline:2px dashed var(--acc)}
.nav{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}.nav button{text-align:center;min-width:84px}
#list{display:flex;gap:4px;flex-wrap:wrap;margin:8px 0}#list span{width:30px;height:26px;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;cursor:pointer;background:var(--card)}
#list span.done{background:#dcefdc}#list span.cur{border:2px solid var(--acc)}
ol{margin:6px 0 0 18px;padding:0;color:var(--mute);font-size:14px}input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:4px}
@media(max-width:900px){.row{grid-template-columns:1fr}.opts{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><b>도어 형상군 판독</b><label>판독자 <input type="text" id="reader" placeholder="이름"></label>
<span id="prog"></span><button id="save" style="margin-left:auto">답안 파일 저장</button></header>
<main>
<ol><li>왼쪽 사진의 도어가 어느 형상군인지 고릅니다. 아래 단추에 마우스를 올리거나 누르면 오른쪽에 그 형상군의 기준 사진이 나옵니다.</li>
<li>기준 사진은 같은 카메라 상태에서 찍은 것입니다. "겹쳐 보기"를 움직이면 기준 사진이 왼쪽 사진 위에 겹쳐집니다.</li>
<li>판단이 서지 않으면 "판독 불가"를 고릅니다. 모든 문항에 답한 뒤 "답안 파일 저장"을 눌러 받은 파일을 보내 주십시오.</li></ol>
<div id="list"></div>
<div class="nav"><button id="prev">이전</button><b id="title"></b><button id="next">다음</button>
<label style="margin-left:16px">겹쳐 보기 <input type="range" id="mix" min="0" max="100" value="0"></label></div>
<div class="row"><div class="pane"><h3>판독할 사진</h3><div class="stack"><img id="item" alt=""><img id="over" alt=""></div></div>
<div class="pane"><h3 id="reftitle">기준 사진</h3><img id="ref" alt="" style="width:100%;display:block"></div></div>
<div class="opts" id="opts"></div>
</main>
<script>
const ITEMS=__DATA__, NAMES=__NAMES__; let cur=0, view=0; let ans={};
try{ans=JSON.parse(localStorage.getItem('door_reading_v16')||'{}')}catch(e){}
const $=id=>document.getElementById(id);
function store(){try{localStorage.setItem('door_reading_v16',JSON.stringify(ans))}catch(e){}}
function refsrc(k){return 'ref_'+ITEMS[cur].state+'_'+(k+1)+'.jpg'}
function show(){const it=ITEMS[cur]; $('item').src='item_'+String(it.item).padStart(2,'0')+'.jpg'; $('title').textContent='문항 '+(cur+1)+' / '+ITEMS.length;
 $('ref').src=refsrc(view); $('over').src=refsrc(view); $('reftitle').textContent='기준 사진: '+NAMES[view];
 const o=$('opts'); o.innerHTML='';
 NAMES.concat(['판독 불가']).forEach((n,k)=>{const b=document.createElement('button'); b.textContent=n;
  if(ans[it.item]===k) b.classList.add('sel'); if(k===view&&k<NAMES.length) b.classList.add('view');
  if(k<NAMES.length){b.onmouseenter=()=>{view=k; $('ref').src=refsrc(k); $('over').src=refsrc(k); $('reftitle').textContent='기준 사진: '+NAMES[k]}}
  b.onclick=()=>{ans[it.item]=k; if(k<NAMES.length) view=k; store(); show()}; o.appendChild(b)});
 const l=$('list'); l.innerHTML=''; ITEMS.forEach((x,i)=>{const s=document.createElement('span'); s.textContent=i+1;
  if(ans[x.item]!==undefined) s.classList.add('done'); if(i===cur) s.classList.add('cur'); s.onclick=()=>{cur=i; show()}; l.appendChild(s)});
 $('prog').textContent='답한 문항 '+Object.keys(ans).length+' / '+ITEMS.length}
$('prev').onclick=()=>{cur=(cur+ITEMS.length-1)%ITEMS.length; show()}; $('next').onclick=()=>{cur=(cur+1)%ITEMS.length; show()};
$('mix').oninput=e=>{$('over').style.opacity=e.target.value/100};
$('save').onclick=()=>{const miss=ITEMS.filter(x=>ans[x.item]===undefined).length; if(miss&&!confirm('답하지 않은 문항이 '+miss+'개 있습니다. 그대로 저장할까요?')) return;
 const name=$('reader').value.trim()||'reader'; let csv='item,answer_index,answer\\n';
 ITEMS.forEach(x=>{const k=ans[x.item]; csv+=x.item+','+(k===undefined?'':k)+','+(k===undefined?'':(k<NAMES.length?NAMES[k]:'판독 불가'))+'\\n'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv'})); a.download='answers_'+name+'.csv'; a.click()};
show();
</script></body></html>
""".replace('__DATA__', data).replace('__NAMES__', names)


def score(path):
    key = read_json(OUT / 'key.json'); truth = {it['item']: it for it in key['items']}; rows = []
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        it = truth[int(r['item'])]; k = r['answer_index']
        ans = None if k == '' else ('cannot_tell' if int(k) >= len(ORDER) else ORDER[int(k)])
        rows.append(dict(item=it['item'], kind=it['kind'], session=it['session'], label=it['label'], answer=ans, agree=ans == it['label']))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), answers_file=Path(path).name, answers_sha256=sha256(path),
               summary={k: dict(n=sum(r['kind'] == k for r in rows), agree=sum(r['kind'] == k and r['agree'] for r in rows),
                                cannot_tell=sum(r['kind'] == k and r['answer'] == 'cannot_tell' for r in rows)) for k in ('input_error', 'cnn_disputed', 'decoy')},
               rows=rows)
    dst = OUT / (Path(path).stem.replace('answers', 'score') + '.json')
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'); print(json.dumps(out['summary'], indent=1)); print(dst)


if __name__ == '__main__':
    build() if sys.argv[1] == 'build' else score(sys.argv[2])
