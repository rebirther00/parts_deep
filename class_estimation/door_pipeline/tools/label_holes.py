"""도어 홀 랜드마크 라벨링 도구 (브라우저).

이미지당 6점: 래치 볼트홀 4 (각공 기준 좌상/우상/좌하/우하) + 상단 모서리 프레임 홀 2 (힌지측/래치측).
좌표는 원본 픽셀. 결과: labels/holes/<클래스>__<idx>.json

실행:
  python tools/label_holes.py                       # datasets/ 8종 × 15장
  python tools/label_holes.py --per-class 40
  python tools/label_holes.py --extra datasets_field # 추가 디렉터리(<class>/rgb_*.png)
  → http://<이 PC IP>:8090

단축키: 1~8 점 선택 (7·8 = 하단 모서리 홀, 선택) / 클릭 = 현재 점 찍기(자동으로 다음 점) / X 안 보임 / Z 되돌리기
        ←→ 이전·다음 이미지 / 휠 확대·축소 / 드래그(우클릭 또는 스페이스+드래그) 이동 / R 전체 초기화
"""
import argparse
import glob
import json
import os
import time

from flask import Flask, jsonify, request, send_file, Response

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(BASE, 'labels', 'holes')
POINTS = [
    ('bolt_tl', '볼트홀 좌상', '#3b82f6'), ('bolt_tr', '볼트홀 우상', '#3b82f6'),
    ('bolt_bl', '볼트홀 좌하', '#60a5fa'), ('bolt_br', '볼트홀 우하', '#60a5fa'),
    ('corner_hinge', '모서리 홀 (힌지측)', '#ef4444'), ('corner_latch', '모서리 홀 (래치측)', '#f97316'),
    ('corner_hinge_bottom', '하단 모서리 홀 (힌지측, 선택)', '#a855f7'), ('corner_latch_bottom', '하단 모서리 홀 (래치측, 선택)', '#d946ef'),
]

ap = argparse.ArgumentParser()
ap.add_argument('--per-class', type=int, default=15)
ap.add_argument('--datasets', default=os.path.join(BASE, 'datasets'))
ap.add_argument('--extra', action='append', default=[], help='추가 디렉터리 (<class>/rgb_*.png), 전부 포함')
ap.add_argument('--port', type=int, default=8090)
args = ap.parse_args()

items = []
for cls_dir in sorted(glob.glob(os.path.join(args.datasets, '*/'))):
    cls = os.path.basename(cls_dir.rstrip('/'))
    files = sorted(glob.glob(os.path.join(cls_dir, 'rgb_*.png')))
    step = max(1, len(files) // args.per_class)
    for f in files[::step][:args.per_class]:
        items.append(dict(cls=cls, idx=os.path.basename(f)[4:8], path=f, src='datasets'))
for ex in args.extra:
    for f in sorted(glob.glob(os.path.join(ex, '**', 'rgb_*.png'), recursive=True)):
        rel = os.path.relpath(os.path.dirname(f), ex).replace(os.sep, '_')
        items.append(dict(cls=rel, idx=os.path.basename(f)[4:8], path=f, src=os.path.basename(ex.rstrip('/'))))
os.makedirs(LABEL_DIR, exist_ok=True)
for i, it in enumerate(items):
    it['id'] = i
    it['key'] = f"{it['src']}__{it['cls']}__{it['idx']}"

app = Flask(__name__)


def label_path(key):
    return os.path.join(LABEL_DIR, key + '.json')


@app.route('/api/list')
def api_list():
    out = []
    for it in items:
        p = label_path(it['key'])
        done = None
        if os.path.exists(p):
            d = json.load(open(p))
            done = sum(1 for k, _, _ in POINTS[:6] if d['points'].get(k) is not None or d['visible'].get(k) is False)
        out.append(dict(id=it['id'], key=it['key'], cls=it['cls'], idx=it['idx'], src=it['src'], done=done))
    return jsonify(dict(items=out, points=POINTS))


@app.route('/api/label/<int:i>', methods=['GET', 'POST'])
def api_label(i):
    it = items[i]
    p = label_path(it['key'])
    if request.method == 'POST':
        d = request.get_json()
        d.update(image=os.path.relpath(it['path'], BASE), cls=it['cls'], idx=it['idx'], src=it['src'],
                 labeled_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(d, open(p, 'w'), ensure_ascii=False, indent=1)
        return jsonify(ok=True)
    if os.path.exists(p):
        return jsonify(json.load(open(p)))
    return jsonify(points={k: None for k, _, _ in POINTS}, visible={k: True for k, _, _ in POINTS})


@app.route('/img/<int:i>')
def img(i):
    return send_file(items[i]['path'])


PAGE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>홀 라벨링</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee;display:flex;height:100vh;overflow:hidden}
#side{width:300px;background:#1c1c1c;padding:12px;box-sizing:border-box;display:flex;flex-direction:column;gap:8px;overflow:auto}
#main{flex:1;position:relative;overflow:hidden;cursor:crosshair}
canvas{position:absolute;left:0;top:0}
#loupe{position:absolute;right:12px;top:12px;width:220px;height:220px;border:2px solid #888;background:#000;pointer-events:none}
.pt{padding:6px 8px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;border:2px solid transparent}
.pt.cur{border-color:#fff}
.pt .st{font-size:12px;opacity:.8}
#list{flex:1;overflow:auto;font-size:12px}
#list div{padding:2px 4px;cursor:pointer;display:flex;justify-content:space-between}
#list div.cur{background:#333}
.ok{color:#4ade80}.part{color:#facc15}.none{color:#777}
button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:6px;cursor:pointer}
kbd{background:#333;border-radius:3px;padding:0 4px;font-size:11px}
</style></head><body>
<div id="side">
 <div id="title" style="font-weight:600"></div>
 <div id="pts"></div>
 <div style="display:flex;gap:6px"><button onclick="nav(-1)">◀ 이전</button><button onclick="nav(1)">다음 ▶</button><button onclick="resetAll()">초기화 R</button></div>
 <div style="font-size:12px;line-height:1.7;opacity:.85">
 <kbd>1</kbd>~<kbd>8</kbd> 점 선택 · 클릭 = 찍기(다음 점으로) · <kbd>X</kbd> 안 보임 · <kbd>Z</kbd> 되돌리기<br>
 <kbd>←</kbd><kbd>→</kbd> 이미지 · 휠 확대 · 우클릭 드래그 이동 · <kbd>F</kbd> 맞춤<br>
 순서: 볼트홀 4개(각공 주변, 좌상→우상→좌하→우하) → 상단 모서리 홀 힌지측 → 래치측 → (선택) 하단 모서리 홀 2개.<br>
 저장은 자동입니다.</div>
 <div id="prog" style="font-size:12px"></div>
 <div id="list"></div>
</div>
<div id="main"><canvas id="c"></canvas><canvas id="loupe"></canvas></div>
<script>
let ITEMS=[],PTS=[],cur=0,curPt=0,img=new Image(),lab=null,hist=[];
let scale=1,ox=0,oy=0,drag=null,mouse=null;
const C=document.getElementById('c'),X=C.getContext('2d'),LP=document.getElementById('loupe'),LX=LP.getContext('2d'),M=document.getElementById('main');
function fit(){C.width=M.clientWidth;C.height=M.clientHeight;if(!img.width)return;scale=Math.min(C.width/img.width,C.height/img.height);ox=(C.width-img.width*scale)/2;oy=(C.height-img.height*scale)/2;draw()}
window.onresize=fit;
async function load(){const r=await fetch('/api/list');const d=await r.json();ITEMS=d.items;PTS=d.points;renderList();await open(cur)}
async function open(i){cur=Math.max(0,Math.min(ITEMS.length-1,i));const it=ITEMS[cur];document.getElementById('title').textContent=`[${cur+1}/${ITEMS.length}] ${it.cls} #${it.idx} (${it.src})`;
 lab=await (await fetch('/api/label/'+it.id)).json();hist=[];curPt=firstOpen();img=new Image();img.onload=fit;img.src='/img/'+it.id+'?t='+Date.now();renderPts();renderList()}
function firstOpen(){for(let k=0;k<PTS.length;k++){const n=PTS[k][0];if(lab.points[n]==null&&lab.visible[n]!==false)return k}return 0}
function renderPts(){const el=document.getElementById('pts');el.innerHTML='';PTS.forEach((p,k)=>{const n=p[0];const d=document.createElement('div');d.className='pt'+(k==curPt?' cur':'');d.style.background=p[2]+'33';
 const st=lab.visible[n]===false?'안 보임':(lab.points[n]?`(${lab.points[n][0].toFixed(1)}, ${lab.points[n][1].toFixed(1)})`:'—');
 d.innerHTML=`<span><b style="color:${p[2]}">${k+1}</b> ${p[1]}</span><span class="st">${st}</span>`;d.onclick=()=>{curPt=k;renderPts();draw()};el.appendChild(d)})}
function renderList(){const el=document.getElementById('list');el.innerHTML='';let done=0;ITEMS.forEach((it,i)=>{const d=document.createElement('div');d.className=i==cur?'cur':'';
 const REQ=6;const c=it.done==null?'none':(it.done>=REQ?'ok':'part');if(it.done>=REQ)done++;
 d.innerHTML=`<span>${it.cls} #${it.idx}</span><span class="${c}">${it.done==null?'':it.done+'/'+REQ}</span>`;d.onclick=()=>open(i);el.appendChild(d)});
 document.getElementById('prog').textContent=`완료 ${done}/${ITEMS.length}`;const c=el.children[cur];if(c)c.scrollIntoView({block:'nearest'})}
function draw(){X.clearRect(0,0,C.width,C.height);if(!img.width)return;X.imageSmoothingEnabled=scale<2;X.drawImage(img,ox,oy,img.width*scale,img.height*scale);
 PTS.forEach((p,k)=>{const q=lab.points[p[0]];if(!q)return;const x=ox+q[0]*scale,y=oy+q[1]*scale;X.strokeStyle=p[2];X.lineWidth=k==curPt?3:1.5;X.beginPath();X.arc(x,y,9,0,7);X.stroke();X.beginPath();X.moveTo(x-14,y);X.lineTo(x+14,y);X.moveTo(x,y-14);X.lineTo(x,y+14);X.stroke();X.fillStyle=p[2];X.font='12px sans-serif';X.fillText(k+1,x+11,y-11)});
 drawLoupe()}
function drawLoupe(){if(!mouse||!img.width){LX.clearRect(0,0,220,220);return}const ix=(mouse.x-ox)/scale,iy=(mouse.y-oy)/scale,z=6,w=220/z;LX.imageSmoothingEnabled=false;LX.clearRect(0,0,220,220);
 LX.drawImage(img,ix-w/2,iy-w/2,w,w,0,0,220,220);LX.strokeStyle='#0f0';LX.lineWidth=1;LX.beginPath();LX.moveTo(110,0);LX.lineTo(110,220);LX.moveTo(0,110);LX.lineTo(220,110);LX.stroke();
 PTS.forEach((p,k)=>{const q=lab.points[p[0]];if(!q)return;const x=(q[0]-ix)*z+110,y=(q[1]-iy)*z+110;if(x<0||y<0||x>220||y>220)return;LX.strokeStyle=p[2];LX.beginPath();LX.arc(x,y,8,0,7);LX.stroke()})}
async function save(){const it=ITEMS[cur];await fetch('/api/label/'+it.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lab)});
 it.done=PTS.slice(0,6).reduce((a,p)=>a+((lab.points[p[0]]!=null||lab.visible[p[0]]===false)?1:0),0);renderList()}
function setPt(x,y){const n=PTS[curPt][0];hist.push(JSON.stringify(lab));lab.points[n]=[+x.toFixed(1),+y.toFixed(1)];lab.visible[n]=true;save();
 const nx=PTS.findIndex((p,k)=>k>curPt&&lab.points[p[0]]==null&&lab.visible[p[0]]!==false);if(nx>=0)curPt=nx;renderPts();draw()}
M.addEventListener('mousedown',e=>{if(e.button==2||e.shiftKey){drag={x:e.clientX,y:e.clientY,ox,oy};return}
 const r=C.getBoundingClientRect();const ix=(e.clientX-r.left-ox)/scale,iy=(e.clientY-r.top-oy)/scale;if(ix<0||iy<0||ix>img.width||iy>img.height)return;setPt(ix,iy)});
M.addEventListener('mousemove',e=>{const r=C.getBoundingClientRect();mouse={x:e.clientX-r.left,y:e.clientY-r.top};if(drag){ox=drag.ox+(e.clientX-drag.x);oy=drag.oy+(e.clientY-drag.y)}draw()});
M.addEventListener('mouseup',()=>drag=null);M.addEventListener('contextmenu',e=>e.preventDefault());
M.addEventListener('wheel',e=>{e.preventDefault();const r=C.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;const f=e.deltaY<0?1.25:0.8;const ns=Math.max(0.1,Math.min(20,scale*f));ox=mx-(mx-ox)*ns/scale;oy=my-(my-oy)*ns/scale;scale=ns;draw()},{passive:false});
function nav(d){open(cur+d)}
function resetAll(){hist.push(JSON.stringify(lab));PTS.forEach(p=>{lab.points[p[0]]=null;lab.visible[p[0]]=true});curPt=0;save();renderPts();draw()}
document.addEventListener('keydown',e=>{if(e.key>='1'&&e.key<='8'){curPt=+e.key-1;renderPts();draw()}
 else if(e.key=='ArrowRight'){nav(1)}else if(e.key=='ArrowLeft'){nav(-1)}
 else if(e.key=='x'||e.key=='X'){const n=PTS[curPt][0];hist.push(JSON.stringify(lab));lab.points[n]=null;lab.visible[n]=false;save();const nx=PTS.findIndex((p,k)=>k>curPt&&lab.points[p[0]]==null&&lab.visible[p[0]]!==false);if(nx>=0)curPt=nx;renderPts();draw()}
 else if(e.key=='z'||e.key=='Z'){if(hist.length){lab=JSON.parse(hist.pop());save();renderPts();draw()}}
 else if(e.key=='r'||e.key=='R'){resetAll()}else if(e.key=='f'||e.key=='F'){fit()}});
load();
</script></body></html>'''


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


if __name__ == '__main__':
    print(f'라벨 대상 {len(items)}장 → 저장 {LABEL_DIR}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
