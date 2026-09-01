# 외부 열람 매뉴얼 — webapp·sqlite_web 인터넷 공개 (tailscale funnel)

DB 열람 웹 도구(`db/webapp.py`)와 SQL 조회 도구(`sqlite_web`)를
**tailscale 앱 없는 외부 사용자**가 브라우저로 볼 수 있게 하는 구성. 2026-09-01 구축.

```
외부 브라우저 ──HTTPS──▶ tailscale funnel (공개 터널) ──▶ 학습 PC(koceti-5090) 로컬 서버
                         443  → 127.0.0.1:5050  webapp   (basic auth + 열람 전용)
                         8443 → 127.0.0.1:8080  sqlite_web (비밀번호 + 읽기 전용)
```

## 1. 접속 정보 (외부 사용자에게 전달)

| 용도 | 주소 | 인증 |
|---|---|---|
| 열람 도구(대시보드·세션·이미지·라벨·평가) | https://koceti-5090.tailf62dcb.ts.net/ | 브라우저 팝업에 id + 비밀번호 |
| SQL 직접 조회(테이블·쿼리) | https://koceti-5090.tailf62dcb.ts.net:8443/ | 로그인 화면에 비밀번호만 |

- 비밀번호는 이 문서에 적지 않는다(서버 시작 명령에서 지정). 전달은 별도 채널로.
- 두 서비스 모두 **읽기 전용** — 외부에서 데이터 변경 불가.
- 주소는 **`https://`를 포함한 전체 링크**로 전달할 것 — 스킴 없이 치면 http로 붙어 연결 리셋됨(§6).

## 2. 평상시 시작 (터미널에서 실행할 것은 이 2개뿐)

funnel은 한 번 등록하면 tailscale에 저장되어 재부팅해도 유지된다.
따라서 평소(재부팅 후 포함)에는 로컬 서버 2개만 띄우면 된다.

```bash
cd ~/parts_deep/class_estimation/door_pipeline
PY=/home/koceti/miniconda3/envs/lecture/bin/python   # flask·waitress·PIL 준비된 환경

# 1) webapp — 열람 전용 + 기본 인증 (id:비밀번호는 원하는 값으로)
nohup $PY db/webapp.py --readonly --auth guest:비밀번호 >> ~/webapp.log 2>&1 &

# 2) sqlite_web — 읽기 전용 + 비밀번호 (127.0.0.1 바인드: funnel 통해서만 접근됨)
SQLITE_WEB_PASSWORD='비밀번호' nohup $PY db/serve_sqlite_web.py \
    -r -P -H 127.0.0.1 -p 8080 db/door_pipeline.db >> ~/sqlite_web.log 2>&1 &
```

두 서버 모두 **waitress(운영용 WSGI 서버)** 로 뜬다 — Flask 개발 서버를 funnel 뒤에 두면
간헐적으로 연결이 리셋되어(제출 직후 `ERR_CONNECTION_RESET`, 재시도하면 성공) 운영용 서버를 쓴다.
`db/serve_sqlite_web.py`는 sqlite_web과 옵션이 동일한 waitress 래퍼다.

`nohup ... &` 이므로 터미널을 닫아도 유지된다.
이미 떠 있는 프로세스를 교체하려면 먼저 종료: `pkill -f webapp.py; pkill -f sqlite_web`

## 3. 상태 확인

```bash
tailscale funnel status                      # 두 터널(443, 8443) 등록 확인
ss -tlnp | grep -E "5050|8080"               # 로컬 서버 2개 떠 있는지
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5050/   # 401 이면 정상(인증 요구)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/   # 302 이면 정상(로그인 리다이렉트)
```

공개 URL 자체 점검(외부 관점): 브라우저 시크릿 창에서 두 주소 접속 → 인증 화면이 뜨면 정상.

## 4. 최초 1회 설정 (완료됨 — 재구축 시에만 필요)

```bash
# funnel 등록 (tailscale 관리 콘솔에서 HTTPS Certificates·funnel 권한 활성화 필요할 수 있음)
sudo tailscale funnel --bg 5050                 # 443  → webapp
sudo tailscale funnel --bg --https=8443 8080    # 8443 → sqlite_web

# sqlite-web 설치 (lecture conda 환경에 설치되어 있음)
/home/koceti/miniconda3/envs/lecture/bin/pip install sqlite-web
```

- funnel 공개 가능 포트는 443·8443·10000 세 개뿐이다(로컬 포트는 자유).
- `webapp.py --auth` 옵션은 2026-09-01 추가(basic auth, env `WEBAPP_AUTH`로도 지정 가능).

## 5. 전부 내리기 (공개 종료)

```bash
sudo tailscale funnel --https=443 off
sudo tailscale funnel --https=8443 off
pkill -f webapp.py; pkill -f sqlite_web
```

funnel만 내리면 공개는 끊기고 로컬(:5050, :8080)은 계속 쓸 수 있다.

## 6. 트러블슈팅

| 증상 | 확인·조치 |
|---|---|
| `ERR_CONNECTION_RESET` (특히 :8443) | ① 주소창에 **`https://` 없이** 입력하면 브라우저가 http로 시도 — funnel은 HTTPS만 받아 리셋됨. `https://` 포함 전체 주소로 접속·전달 ② 서버를 Flask 개발 서버로 띄운 경우 제출 직후 간헐 리셋 — §2 명령(waitress)으로 실행했는지 확인 |
| 비밀번호가 맞는데 로그인 안 됨 | ① 주소 오타 확인(정확히 `koceti-5090.tailf62dcb.ts.net`) ② 한/영 전환·앞뒤 공백 ③ 브라우저 쿠키 차단 시 sqlite_web 로그인이 유지 안 됨 |
| 공개 URL이 아예 안 열림 | `tailscale funnel status`로 터널 확인 → 없으면 §4 재등록. `tailscale status`로 tailscale 자체 가동 확인 |
| 터널은 있는데 502/연결 실패 | 로컬 서버 죽음 — §3으로 확인 후 §2로 재시작. `~/webapp.log`, `~/sqlite_web.log` 확인 |
| 포트 이미 사용 중 | `ss -tlnp \| grep 5050` 등으로 기존 프로세스 확인 후 `pkill` |
| webapp에서 수정 버튼이 보임 | `--readonly` 빠뜨림 — 즉시 재시작(공개 중 필수) |

## 7. 보안 수칙

- 공개 시 webapp은 **반드시 `--readonly` + `--auth`**, sqlite_web은 **반드시 `-r` + `-P`**.
- 미팅·점검 기간에만 열고 평소에는 §5로 내려두는 운용을 권장.
- 비밀번호는 문서·저장소에 남기지 않는다. 유출 의심 시 서버 재시작으로 즉시 교체 가능.
- sqlite_web은 DB 전체(경로·메타)가 그대로 보이므로 공개 대상을 판단하고 열 것.
