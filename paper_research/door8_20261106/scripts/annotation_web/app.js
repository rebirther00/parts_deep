'use strict';
const $ = id => document.getElementById(id);
const keys = ['bolt_tl','bolt_tr','bolt_bl','bolt_br','corner_hinge','corner_latch'];
const names = ['래치 볼트 좌상','래치 볼트 우상','래치 볼트 좌하','래치 볼트 우하','힌지 측 기준 홀','래치 측 기준 홀'];
const sessionMeta = document.querySelector('meta[name="annotation-token"]');
const token = (sessionMeta && sessionMeta.content) || new URLSearchParams(location.hash.slice(1)).get('token') || '';
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let items = [], row = null, index = 0, selected = 0, image = null;
let scale = 1, ox = 0, oy = 0, dirty = false, busy = false, pan = null;

// Remembering an author is optional; privacy restrictions must not block work.
function rememberedAuthor() {
  try { return localStorage.getItem('door8Annotator') || ''; } catch (_) { return ''; }
}
function rememberAuthor(author) {
  try { localStorage.setItem('door8Annotator',author); } catch (_) { /* Optional preference. */ }
}
function startupError(message) {
  if (window.door8StartupError) window.door8StartupError(message);
  else $('progress').textContent = '연결 상태 확인 필요';
  notice(message,true);
}
function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('button,select').forEach(el => { el.disabled = value; });
}
async function api(path, body) {
  const headers = {'X-Annotation-Token': token};
  const controller = new AbortController();
  const options = {headers, cache:'no-store', signal:controller.signal};
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    options.method = 'POST'; options.body = JSON.stringify(body);
  }
  const timeout = setTimeout(()=>controller.abort(),8000);
  try {
    const response = await fetch(path, options);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `요청 실패 (${response.status})`);
    }
    // Consume the body before clearing the timeout, including image downloads.
    const blob = await response.blob();
    return {blob:async()=>blob,json:async()=>JSON.parse(await blob.text())};
  } catch (error) {
    if (error.name==='AbortError') throw new Error('서버 응답이 8초 안에 도착하지 않았습니다. 서버 터미널과 포트 전달을 확인하세요.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
function changed() { dirty = true; render(); }
function draw() {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
    canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
  if (!image || !row) return;
  ctx.drawImage(image,ox,oy,image.width*scale,image.height*scale);
  keys.forEach((key,i) => {
    const point = row.points[key]; if (!point) return;
    const x = point[0]*scale+ox, y = point[1]*scale+oy;
    ctx.strokeStyle = i===selected ? '#ffd06a' : '#53efc7'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-10,y); ctx.lineTo(x+10,y); ctx.moveTo(x,y-10); ctx.lineTo(x,y+10); ctx.stroke();
    ctx.font = 'bold 15px system-ui'; ctx.lineWidth = 4; ctx.strokeStyle = '#10151d';
    ctx.strokeText(String(i+1),x+12,y-10); ctx.fillStyle = i===selected ? '#ffd06a' : '#53efc7';
    ctx.fillText(String(i+1),x+12,y-10);
  });
}
function render() {
  if (!row) return;
  $('dirty').textContent = dirty ? '저장하지 않은 변경 있음' : '';
  keys.forEach((key,i) => {
    const button = $('point-'+i), p = row.points[key];
    button.classList.toggle('active',i===selected);
    button.classList.toggle('done',p !== null || !row.visible[key]);
    button.querySelector('.state').textContent = !row.visible[key] ? '가려짐' : p ? `${Math.round(p[0])}, ${Math.round(p[1])}` : '미표시';
  });
  draw();
}
function fit() {
  if (!image) return;
  scale = Math.min(canvas.clientWidth/image.width,canvas.clientHeight/image.height)*.96;
  ox = (canvas.clientWidth-image.width*scale)/2; oy = (canvas.clientHeight-image.height*scale)/2;
  draw();
}
async function refreshItems() {
  items = (await (await api('/api/items')).json()).items;
  $('items').replaceChildren(...items.map((item,i) => {
    const option = document.createElement('option'); option.value = i;
    option.textContent = `${item.saved ? '✓' : '○'} ${i+1}/${items.length} · ${item.class_name}`;
    return option;
  }));
  $('items').value = String(index);
  $('progress').textContent = `저장 완료 ${items.filter(item=>item.saved).length} / ${items.length}장`;
}
async function load(nextIndex, ask = true) {
  if (busy) return;
  if (ask && dirty && !window.confirm('저장하지 않은 변경이 있습니다. 변경을 버리고 이동할까요?')) {
    $('items').value = String(index); return;
  }
  setBusy(true);
  try {
    nextIndex = (nextIndex+items.length)%items.length;
    const id = encodeURIComponent(items[nextIndex].annotation_id);
    const nextRow = await (await api('/api/item?id='+id)).json();
    const blob = await (await api('/api/image?id='+id)).blob();
    const url = URL.createObjectURL(blob), nextImage = new Image();
    try { nextImage.src = url; await nextImage.decode(); } finally { URL.revokeObjectURL(url); }
    index = nextIndex; row = nextRow; image = nextImage; selected = 0; dirty = false;
    if (!$('author').value && row.annotator) $('author').value = row.annotator;
    $('items').value = String(index);
    $('detail').textContent = `${row.session_dir} · CV 그룹 ${row.cv_group}`;
    $('dimensions').textContent = `${image.width} × ${image.height} · ${row.class_name}`;
    notice(row.revision ? '저장한 좌표를 불러왔습니다. 수정 후 다시 저장할 수 있습니다.' : '원본을 확인하며 기준점을 표시하세요.');
    $('startup-error').hidden = true;
    fit(); render();
  } catch (error) { $('items').value = String(index); startupError(error.message); }
  finally { setBusy(false); }
}
function invisible() {
  if (busy || !row) return;
  const key = keys[selected]; row.points[key] = null; row.visible[key] = false;
  selected = (selected+1)%keys.length; changed();
}
async function save() {
  if (busy || !row) return;
  setBusy(true);
  try {
    const result = await (await api('/api/save',{
      annotation_id:row.annotation_id, annotator:$('author').value,
      points:row.points, visible:row.visible, revision:row.revision
    })).json();
    row.revision = result.revision; dirty = false;
    rememberAuthor($('author').value);
    await refreshItems(); render();
    let nextIndex = -1;
    for (let step=1;step<=items.length;step++) {
      const candidate = (index+step)%items.length;
      if (!items[candidate].saved) { nextIndex=candidate; break; }
    }
    setBusy(false);
    if (nextIndex>=0) await load(nextIndex,false);
    else notice(`${items.length}장 모두 저장했습니다. 영상 선택에서 기존 좌표를 다시 확인할 수 있습니다.`);
  } catch (error) { notice(error.message,true); }
  finally { setBusy(false); }
}
keys.forEach((key,i) => {
  const button = document.createElement('button'); button.id = 'point-'+i; button.className = 'point';
  const label = document.createElement('span'); label.textContent = `${i+1}. ${names[i]}`;
  const state = document.createElement('span'); state.className = 'state';
  button.append(label,state); button.onclick = () => {selected=i; render();}; $('points').append(button);
});
$('author').value = rememberedAuthor();
$('author').oninput = () => { if (row) changed(); };
$('items').onchange = () => load(Number($('items').value));
$('previous').onclick = () => load(index-1); $('next').onclick = () => load(index+1);
$('fit').onclick = fit; $('invisible').onclick = invisible; $('save').onclick = save;
$('clear').onclick = () => {
  if (busy || !row) return;
  row.points[keys[selected]]=null; row.visible[keys[selected]]=true; changed();
};
canvas.addEventListener('contextmenu',event=>event.preventDefault());
canvas.addEventListener('pointerdown',event=>{
  if (busy || !image) return;
  canvas.focus(); const bounds=canvas.getBoundingClientRect();
  const x=event.clientX-bounds.left, y=event.clientY-bounds.top;
  if (event.button===2) {
    pan={x:event.clientX,y:event.clientY,ox,oy}; canvas.setPointerCapture(event.pointerId); return;
  }
  if (event.button!==0) return;
  const px=(x-ox)/scale, py=(y-oy)/scale;
  if (px<0 || py<0 || px>=image.width || py>=image.height) return;
  row.points[keys[selected]]=[px,py].map(v=>Math.round(v*1000)/1000);
  row.visible[keys[selected]]=true; selected=(selected+1)%keys.length; changed();
});
canvas.addEventListener('pointermove',event=>{
  if (!pan) return;
  ox=pan.ox+event.clientX-pan.x; oy=pan.oy+event.clientY-pan.y; draw();
});
canvas.addEventListener('pointerup',()=>{pan=null;});
canvas.addEventListener('pointercancel',()=>{pan=null;});
canvas.addEventListener('wheel',event=>{
  event.preventDefault(); if (!image || busy) return;
  const bounds=canvas.getBoundingClientRect(), x=event.clientX-bounds.left, y=event.clientY-bounds.top;
  const next=Math.max(.05,Math.min(12,scale*(event.deltaY<0?1.25:.8))), factor=next/scale;
  ox=x-(x-ox)*factor; oy=y-(y-oy)*factor; scale=next; draw();
},{passive:false});
window.addEventListener('keydown',event=>{
  if (busy || !row || event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  const key=event.key.toLowerCase();
  if (/^[1-6]$/.test(key)) {selected=Number(key)-1; render();}
  else if (key==='v') invisible(); else if (key==='f') fit(); else if (key==='s') {event.preventDefault(); save();}
});
window.addEventListener('beforeunload',event=>{if (dirty) {event.preventDefault(); event.returnValue='';}});
if (typeof ResizeObserver==='function') {
  new ResizeObserver(()=>{if (image) fit(); else draw();}).observe($('canvas-wrap'));
} else window.addEventListener('resize',fit);
(async()=>{
  setBusy(true);
  try {
    if (!token) throw new Error('실행 중인 서버의 접속 정보가 없습니다. 서버를 Ctrl+C로 종료하고 같은 명령으로 다시 실행한 뒤 새로고침해 주세요.');
    await refreshItems(); setBusy(false);
    const first=items.findIndex(item=>!item.saved); await load(first<0?0:first,false);
  } catch (error) {startupError(error.message); setBusy(false);}
  finally {window.door8BootSettled=true;}
})();
