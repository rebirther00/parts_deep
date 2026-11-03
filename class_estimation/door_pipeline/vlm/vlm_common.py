"""VLM(Claude vision) 공통 — 클라이언트·이미지 인코딩·대표 이미지 프리픽스(캐시)·비용 집계.

API 크레딧(2026-11-17 만료) 활용 작업(2026-11-03). 키는 ANTHROPIC_API_KEY 환경변수 또는
~/.config/anthropic/api_key 파일(600, dx300_rl 과 공용). 모델은 Claude Opus 5.5 고정.
"""
import base64
import io
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)                                   # door_pipeline/
REF_DIR = os.path.join(DOOR, 'reference_images')
ART = os.path.join(HERE, 'artifacts')
KEY_FILES = [os.path.expanduser('~/.config/anthropic/api_key'),            # dx300_rl 과 같은 관례(정본)
             os.path.expanduser('~/.config/parts_deep/anthropic_api_key')]
KEY_FILE = KEY_FILES[0]

MODEL = 'claude-opus-5-5'
# $/MTok (Claude Opus 5.5, 2026-10 기준): 입력 4, 출력 20, 캐시 쓰기 5(1.25x), 캐시 읽기 0.20
PRICE = {'input': 4.0, 'output': 20.0, 'cache_write': 5.0, 'cache_read': 0.20}
IMG_W = 960                                                   # 960x600 ≈ 768 토큰/장

# 9 형상군 — hole_classifier.CAD_D 와 동일(토치 import 회피용 사본, 값은 hole_classifier 가 정본)
CAD_D = {'E23_door_LH_FRT': 458, 'E25_door_LH_FRT': 724, 'E30_door_LH_FRT': 765, 'E38_door_LH_FRT': 812,
         'E25_door_LH_RR': 1037, 'E30_door_LH_RR': 1158, 'E38_door_LH_RR': 1352, 'E25_door_RH': 886, 'E30_E38_door_RH': 1087}
CLASSES = list(CAD_D)
GROUP = {c: ('FRT' if 'FRT' in c else 'RH' if c.endswith('RH') else 'LH_RR') for c in CAD_D}
WIDTH = {c: d + 106 for c, d in CAD_D.items()}                # 도어 폭 ≈ D + 106 (사양표 값이 있으면 아래서 덮어씀)
try:
    for _p in json.load(open(os.path.join(DOOR, 'partno', 'part_numbers.json'), encoding='utf-8'))['parts']:
        if _p.get('width_mm'):
            WIDTH[_p['class_name']] = _p['width_mm']
except Exception:
    pass


def load_part_numbers():
    p = os.path.join(DOOR, 'partno', 'part_numbers.json')
    return {(x['class_name'], x['radar']): x for x in json.load(open(p, encoding='utf-8'))['parts']}


def part_for(cls, radar, pn=None):
    pn = pn or load_part_numbers()
    if cls not in CAD_D:
        return None
    return pn.get((cls, None)) or (pn.get((cls, int(radar))) if radar not in (None, '') else None)


def api_key():
    k = os.environ.get('ANTHROPIC_API_KEY')
    if k:
        return k.strip()
    for f in KEY_FILES:
        if os.path.exists(f):
            k = open(f).read().strip()
            if k:
                return k
    return None


def get_client(max_retries=5, timeout=180.0):
    import anthropic
    k = api_key()
    if not k:
        sys.exit(f'API 키 없음 — ANTHROPIC_API_KEY 또는 {KEY_FILE} 에 넣을 것')
    return anthropic.Anthropic(api_key=k, max_retries=max_retries, timeout=timeout)


def img_b64(path, width=IMG_W):
    """PNG/JPG → 가로 width 로 축소한 JPEG base64 (API 가 1.15MP 초과분을 다시 줄이므로 미리 맞춘다)."""
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode('ascii')


def image_block(path, cache=False):
    b = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': img_b64(path)}}
    if cache:
        b['cache_control'] = {'type': 'ephemeral'}
    return b


def ref_label(fname):
    """reference_images 파일명 → (class, radar) ; empty_jig → ('EMPTY', None)"""
    stem = os.path.splitext(fname)[0]
    if stem.startswith('empty_jig'):
        return 'EMPTY', None
    m = re.match(r'^(.+?)(?:__radar([01]))?$', stem)
    return m.group(1), (int(m.group(2)) if m.group(2) else None)


def reference_blocks():
    """대표 이미지 14장 + 빈 지그 1장 → [text, image, text, image, ...] (마지막 블록에 캐시 브레이크포인트).
    _pts(점 오버레이) 파일은 제외 — 판정기 결과가 새어 나가지 않게."""
    files = sorted(f for f in os.listdir(REF_DIR) if f.endswith('.jpg') and '_pts' not in f)
    blocks = [{'type': 'text', 'text': f'아래는 참조 이미지 {len(files)}장이다. 각 이미지 앞 줄에 정답 라벨을 적었다. 모두 같은 카메라·같은 거리에서 찍었으므로 도어의 화면상 크기는 실제 폭에 비례한다.'}]
    for f in files:
        cls, radar = ref_label(f)
        if cls == 'EMPTY':
            label = '[참조] 빈 지그 (도어 없음)'
        else:
            label = f'[참조] class={cls} / group={GROUP[cls]} / 폭≈{WIDTH[cls]}mm'
            if radar is not None:
                label += ' / radar=' + ('O (브래킷 있음)' if radar else 'X (브래킷 없음)')
            else:
                label += ' / radar=해당 없음(FRT)'
        blocks.append({'type': 'text', 'text': label})
        blocks.append(image_block(os.path.join(REF_DIR, f)))
    blocks[-1]['cache_control'] = {'type': 'ephemeral'}
    return blocks


DOMAIN_TEXT = f"""너는 굴착기 사이드 도어 용접 지그를 위에서 내려다보는 고정 카메라 영상을 검사하는 독립 검사관이다.
도어는 지그에 눕혀져 안쪽 판(홀이 있는 면)이 카메라를 향한다. 화면 가운데의 금속 판이 도어이고, 주변의 어두운 큰 판과 빨간 손잡이 클램프는 지그다.
지그의 세로 바 2개는 도어 기종에 따라 좌우로 움직이므로 바 위치는 단서가 아니다. 카메라와 거리가 고정이라 **도어의 화면상 폭은 실제 폭에 비례**한다.

형상군(group) 3종과 클래스 9종 (폭 = 사양표 기준):
- FRT (앞문, 레이더 옵션 없음): E23_door_LH_FRT {WIDTH['E23_door_LH_FRT']}mm(가장 좁음) · E25_door_LH_FRT {WIDTH['E25_door_LH_FRT']} · E30_door_LH_FRT {WIDTH['E30_door_LH_FRT']} · E38_door_LH_FRT {WIDTH['E38_door_LH_FRT']}.
  판에 통풍 그릴(루버)이 없고 작은 사각 구멍만 있다. E30↔E38 은 폭 47mm 차이뿐이라 육안 구분이 어렵다 — 확신이 없으면 class_confidence 를 low 로 두라.
- LH_RR (왼쪽 뒷문): E25_door_LH_RR {WIDTH['E25_door_LH_RR']} · E30_door_LH_RR {WIDTH['E30_door_LH_RR']} · E38_door_LH_RR {WIDTH['E38_door_LH_RR']}(가장 넓음).
  위·아래 두 구획 모두에 큰 사선 루버 그릴이 있다.
- RH (오른쪽 문): E25_door_RH {WIDTH['E25_door_RH']} · E30_E38_door_RH {WIDTH['E30_E38_door_RH']}.
  위 구획에만 가로 슬릿 그릴이 있고 아래 구획은 평판이다.

레이더 옵션(radar): LH_RR·RH 도어에만 있다. 도어 중앙 가로 보강대(스트립) 위에 **둥근 구멍 2개가 뚫린 넓은 브래킷 판**이 붙어 있으면 O, 중앙 스트립이 좁은 평판이면 X. FRT 는 "N/A".

이상(anomalies): 호스·케이블·공구가 도어 위를 가로지름, 작업자 손·몸, 빈 지그, 도어가 아닌 부품, 밑판만 있고 보강대/브래킷이 아직 안 올라간 상태, 도어 기울어짐/치우침, 조명 불량 등.
판단은 오직 이미지와 참조 이미지에 근거하고, 모르면 unsure 로 답하라. description 과 reasoning 은 한국어로 짧게."""

AUDIT_SCHEMA = {
    'type': 'object',
    'properties': {
        'door_present': {'type': 'boolean'},
        'group': {'type': 'string', 'enum': ['FRT', 'LH_RR', 'RH', 'none', 'unsure']},
        'class_name': {'type': 'string', 'enum': CLASSES + ['none', 'unsure']},
        'class_confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'radar': {'type': 'string', 'enum': ['O', 'X', 'N/A', 'unsure']},
        'anomalies': {'type': 'array', 'items': {'type': 'string', 'enum': [
            'hose_or_cable_over_door', 'worker_or_hand', 'tool_or_object_on_door', 'empty_jig', 'non_door_part',
            'bottom_plate_only', 'bracket_missing', 'door_tilted_or_shifted', 'poor_lighting', 'other']}},
        'description': {'type': 'string'},
        'reasoning': {'type': 'string'},
    },
    'required': ['door_present', 'group', 'class_name', 'class_confidence', 'radar', 'anomalies', 'description', 'reasoning'],
    'additionalProperties': False,
}


def usage_cost(u):
    """anthropic usage → (dict, usd)"""
    d = {'input_tokens': getattr(u, 'input_tokens', 0) or 0,
         'cache_read_tokens': getattr(u, 'cache_read_input_tokens', 0) or 0,
         'cache_write_tokens': getattr(u, 'cache_creation_input_tokens', 0) or 0,
         'output_tokens': getattr(u, 'output_tokens', 0) or 0}
    usd = (d['input_tokens'] * PRICE['input'] + d['output_tokens'] * PRICE['output']
           + d['cache_write_tokens'] * PRICE['cache_write'] + d['cache_read_tokens'] * PRICE['cache_read']) / 1e6
    return d, usd


def text_of(resp):
    return next((b.text for b in resp.content if b.type == 'text'), '')


class Tally:
    def __init__(self):
        self.lock = threading.Lock(); self.n = 0; self.usd = 0.0; self.tok = {}
    def add(self, d, usd):
        with self.lock:
            self.n += 1; self.usd += usd
            for k, v in d.items():
                self.tok[k] = self.tok.get(k, 0) + v
    def __str__(self):
        return f'{self.n}건 ${self.usd:.3f} ' + ' '.join(f'{k}={v}' for k, v in self.tok.items())


def run_id():
    return time.strftime('%Y%m%d_%H%M%S')
