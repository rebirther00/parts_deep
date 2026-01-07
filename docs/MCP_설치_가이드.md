# 🔌 MCP (Model Context Protocol) 설치 가이드

## 📋 개요

MCP(Model Context Protocol)는 AI 모델이 외부 도구와 데이터 소스에 접근할 수 있도록 하는 표준 프로토콜입니다. Cursor IDE에서 MCP 서버를 설정하여 AI의 기능을 확장할 수 있습니다.

---

## ⚠️ 중요: Deprecated 서버 안내 (2026년 1월 기준)

다음 서버들은 **더 이상 지원되지 않습니다**:
- ~~`@modelcontextprotocol/server-brave-search`~~ → Cursor 내장 웹 검색 사용
- ~~`@modelcontextprotocol/server-github`~~ → 대안 확인 중
- ~~`@modelcontextprotocol/server-puppeteer`~~ → `@playwright/mcp` 사용

---

## 🛠️ 필수 요구사항

- **Node.js**: 20 이상 (필수!)
- **Cursor IDE**: 최신 버전

---

## 📦 1단계: Node.js 설치 (NVM 권장)

### 1.1 NVM 설치

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
```

### 1.2 Node.js 20 설치

```bash
nvm install 20
nvm use 20
nvm alias default 20
```

### 1.3 설치 확인

```bash
node --version  # v20.x.x 이상
npm --version   # 10.x.x 이상
```

---

## 🎯 2단계: 지원되는 MCP 서버 목록

### 2.1 필수 서버 (추천) ✅

| 서버 | 패키지 | 기능 |
|------|--------|------|
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | 파일 읽기/쓰기, 디렉토리 탐색 |
| **Memory** | `@modelcontextprotocol/server-memory` | 대화 기록 저장 및 검색 |
| **Sequential Thinking** | `@modelcontextprotocol/server-sequential-thinking` | 단계별 사고 프로세스 |
| **Playwright** | `@playwright/mcp` | 웹 브라우저 자동화 (강력!) |

### 2.2 설치 명령어

```bash
# NVM 로드 (새 터미널에서 필요)
source ~/.nvm/nvm.sh

# 필수 서버 설치
npm install -g \
  @modelcontextprotocol/server-filesystem \
  @modelcontextprotocol/server-memory \
  @modelcontextprotocol/server-sequential-thinking \
  @playwright/mcp

# Playwright 브라우저 설치
npx playwright install chromium
```

---

## ⚙️ 3단계: Cursor IDE MCP 설정

### 3.1 설정 파일 위치

```bash
~/.cursor/mcp.json
```

### 3.2 설정 파일 내용

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": [
        "/home/koceti/.nvm/versions/node/v20.19.6/lib/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
        "/home/koceti/parts_deep"
      ]
    },
    "memory": {
      "command": "node",
      "args": [
        "/home/koceti/.nvm/versions/node/v20.19.6/lib/node_modules/@modelcontextprotocol/server-memory/dist/index.js"
      ]
    },
    "sequential-thinking": {
      "command": "node",
      "args": [
        "/home/koceti/.nvm/versions/node/v20.19.6/lib/node_modules/@modelcontextprotocol/server-sequential-thinking/dist/index.js"
      ]
    },
    "playwright": {
      "command": "node",
      "args": [
        "/home/koceti/.nvm/versions/node/v20.19.6/lib/node_modules/@playwright/mcp/cli.js"
      ]
    }
  }
}
```

> ⚠️ Node.js 버전 경로(`v20.19.6`)는 설치된 버전에 맞게 수정하세요.
> 확인 명령: `ls ~/.nvm/versions/node/`

---

## ✅ 4단계: 설치 확인

### 4.1 설치된 서버 확인

```bash
source ~/.nvm/nvm.sh
npm list -g | grep -E "@modelcontextprotocol|@playwright"
```

**예상 출력:**
```
├── @modelcontextprotocol/server-filesystem@2025.12.18
├── @modelcontextprotocol/server-memory@2025.11.25
├── @modelcontextprotocol/server-sequential-thinking@2025.12.18
├── @playwright/mcp@0.0.54
```

### 4.2 Cursor IDE 적용

1. **Cursor IDE 재시작** (필수!)
2. 설정 → MCP Servers 확인
3. 4개 서버가 연결되었는지 확인

---

## 🎭 5단계: 각 서버 기능 상세

### 📁 Filesystem Server
- 파일 읽기/쓰기
- 디렉토리 목록 조회
- 파일 검색
- 프로젝트 구조 탐색

### 🧠 Memory Server
- 대화 기록 저장
- 이전 컨텍스트 검색
- 장기 기억 유지

### 🔄 Sequential Thinking Server
- 복잡한 문제 단계별 분석
- 체계적 사고 프로세스
- 중간 결과 추적

### 🎭 Playwright Server
- 웹 페이지 탐색 (URL 접속, 클릭)
- 스크린샷 촬영
- 폼 작성/제출
- JavaScript 실행
- PDF 생성
- 웹 스크래핑

---

## 🔧 6단계: 문제 해결

### 6.1 Node.js 버전 문제

```bash
# 현재 버전 확인
node --version

# Node.js 20 이상 필요
source ~/.nvm/nvm.sh
nvm install 20
nvm use 20
```

### 6.2 권한 문제 (EACCES)

NVM을 사용하면 권한 문제가 발생하지 않습니다.
시스템 Node.js 사용 시:

```bash
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 6.3 MCP 서버 연결 실패

```bash
# 설정 파일 확인
cat ~/.cursor/mcp.json

# 서버 경로 확인
ls ~/.nvm/versions/node/v20.19.6/lib/node_modules/

# Cursor 로그 확인
tail -f ~/.config/Cursor/logs/*/exthost/anysphere.cursor-mcp/*.log
```

### 6.4 Playwright 브라우저 없음

```bash
source ~/.nvm/nvm.sh
npx playwright install chromium
```

---

## 📚 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Playwright MCP](https://github.com/playwright/mcp)
- [Cursor IDE MCP 가이드](https://docs.cursor.com/mcp)

---

## 📅 문서 정보

- **작성일**: 2026년 1월 7일
- **수정일**: 2026년 1월 7일
- **대상 환경**: Ubuntu 24.04 LTS, Cursor IDE
- **Node.js 버전**: 20.19.6 (NVM 관리)

---

## 📊 현재 설치 상태 요약

| 서버 | 버전 | 상태 |
|------|------|------|
| filesystem | 2025.12.18 | ✅ 정상 |
| memory | 2025.11.25 | ✅ 정상 |
| sequential-thinking | 2025.12.18 | ✅ 정상 |
| playwright | 0.0.54 | ✅ 정상 |

> 💡 **팁**: Cursor IDE 재시작 후 MCP 서버가 활성화됩니다!
