/* Door Dataset Builder — 프론트엔드 */

const state = {
    selectedClass: null,
    recording: false,
    selectedFrames: new Set(),
};

let pollTimer = null;

document.addEventListener("DOMContentLoaded", () => {
    loadCameraInfo();
    refreshDashboard();
    setInterval(refreshDashboard, 5000);
});

/* ── 카메라 정보 ──────────────────────────── */

async function loadCameraInfo() {
    try {
        const res = await fetch("/api/camera_info");
        const data = await res.json();
        const badge = document.getElementById("cameraBadge");
        badge.textContent = data.camera_type;
        if (data.connected) badge.classList.add("connected");
    } catch { /* 무시 */ }
}

/* ── 클래스 선택 ──────────────────────────── */

function selectClass(className) {
    state.selectedClass = className;
    document.querySelectorAll(".class-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.class === className);
    });
    updateStatus(className + " 선택됨");
}

/* ── 녹화 ─────────────────────────────────── */

function toggleRecording() {
    if (state.recording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    if (!state.selectedClass) {
        alert("클래스를 먼저 선택하세요.");
        return;
    }

    const interval = parseInt(document.getElementById("frameInterval").value) || 5;
    const threshold = parseInt(document.getElementById("blurThreshold").value) || 100;

    try {
        const res = await fetch("/api/start_recording", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                class_name: state.selectedClass,
                frame_interval: interval,
                blur_threshold: threshold,
            }),
        });
        if (res.ok) {
            state.recording = true;
            setRecordingUI(true);
            startPolling();
        }
    } catch (e) {
        alert("녹화 시작 실패: " + e.message);
    }
}

async function stopRecording() {
    try {
        const res = await fetch("/api/stop_recording", { method: "POST" });
        const data = await res.json();
        state.recording = false;
        setRecordingUI(false);
        stopPolling();
        updateStatus(
            "녹화 정지 | " + data.extracted_count + "개 프레임 추출됨"
        );
        await loadFrames();
    } catch (e) {
        alert("녹화 정지 실패: " + e.message);
    }
}

function setRecordingUI(on) {
    const btn = document.getElementById("btnRecord");
    const indicator = document.getElementById("recordIndicator");
    btn.textContent = on ? "녹화 정지" : "녹화 시작";
    btn.classList.toggle("recording", on);
    indicator.classList.toggle("active", on);
}

/* ── 녹화 중 폴링 ─────────────────────────── */

function startPolling() {
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch("/api/recording_status");
            const d = await res.json();
            updateStatus(
                "녹화 중 | 프레임: " + d.frame_count +
                " | 추출: " + d.extracted_count
            );
        } catch { /* 무시 */ }
    }, 500);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

/* ── 스냅샷 ───────────────────────────────── */

async function takeSnapshot() {
    if (!state.selectedClass) {
        alert("클래스를 먼저 선택하세요.");
        return;
    }
    try {
        const res = await fetch("/api/snapshot", { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            updateStatus("스냅샷 완료 (blur: " + data.blur_score + ")");
            await loadFrames();
        } else {
            alert(data.error || "스냅샷 실패");
        }
    } catch (e) {
        alert("스냅샷 실패: " + e.message);
    }
}

/* ── 프레임 목록 ──────────────────────────── */

async function loadFrames() {
    try {
        const res = await fetch("/api/extracted_frames");
        const frames = await res.json();
        renderFrames(frames);
    } catch { /* 무시 */ }
}

function renderFrames(frames) {
    const grid = document.getElementById("framesGrid");

    if (frames.length === 0) {
        grid.innerHTML =
            '<div class="empty-state">추출된 프레임이 없습니다.<br>' +
            "녹화 또는 스냅샷으로 프레임을 획득하세요.</div>";
        updateSelectionInfo(0, 0);
        return;
    }

    state.selectedFrames = new Set(frames.map((f) => f.filename));

    grid.innerHTML = frames
        .map(
            (f) =>
                '<div class="frame-card selected" data-filename="' +
                f.filename +
                '" onclick="toggleFrame(\'' +
                f.filename +
                "')\">" +
                '<img src="/temp_frames/' +
                f.filename +
                '" width="200" height="140" loading="lazy" style="width:100%;height:140px;min-height:140px;object-fit:cover;display:block;background:#1e293b;">' +
                '<div class="check-mark">&#10003;</div>' +
                '<div class="frame-info">' +
                f.filename +
                "</div></div>"
        )
        .join("");

    updateSelectionInfo(state.selectedFrames.size, frames.length);
}

function toggleFrame(filename) {
    const card = document.querySelector(
        '.frame-card[data-filename="' + filename + '"]'
    );
    if (!card) return;

    if (state.selectedFrames.has(filename)) {
        state.selectedFrames.delete(filename);
        card.classList.remove("selected");
    } else {
        state.selectedFrames.add(filename);
        card.classList.add("selected");
    }
    updateSelectionInfo(
        state.selectedFrames.size,
        document.querySelectorAll(".frame-card").length
    );
}

function selectAllFrames() {
    document.querySelectorAll(".frame-card").forEach((card) => {
        state.selectedFrames.add(card.dataset.filename);
        card.classList.add("selected");
    });
    updateSelectionInfo(
        state.selectedFrames.size,
        document.querySelectorAll(".frame-card").length
    );
}

function deselectAllFrames() {
    state.selectedFrames.clear();
    document.querySelectorAll(".frame-card").forEach((card) => {
        card.classList.remove("selected");
    });
    updateSelectionInfo(0, document.querySelectorAll(".frame-card").length);
}

function updateSelectionInfo(selected, total) {
    const el = document.getElementById("selectionInfo");
    if (total > 0) {
        el.textContent = "(" + selected + "/" + total + " 선택)";
    } else {
        el.textContent = "";
    }
}

/* ── 저장 / 삭제 ──────────────────────────── */

async function saveSelected() {
    if (!state.selectedClass) {
        alert("클래스를 먼저 선택하세요.");
        return;
    }
    if (state.selectedFrames.size === 0) {
        alert("저장할 프레임을 선택하세요.");
        return;
    }

    try {
        const res = await fetch("/api/save_selected", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                class_name: state.selectedClass,
                filenames: Array.from(state.selectedFrames),
            }),
        });
        const data = await res.json();
        if (res.ok) {
            updateStatus(
                data.count + "개 이미지 저장 완료 → " + data.class_name
            );
            await clearTemp();
            await refreshDashboard();
        } else {
            alert(data.error || "저장 실패");
        }
    } catch (e) {
        alert("저장 실패: " + e.message);
    }
}

async function clearTemp() {
    try {
        await fetch("/api/clear_temp", { method: "POST" });
        state.selectedFrames.clear();
        document.getElementById("framesGrid").innerHTML =
            '<div class="empty-state">추출된 프레임이 없습니다.<br>' +
            "녹화 또는 스냅샷으로 프레임을 획득하세요.</div>";
        updateSelectionInfo(0, 0);
    } catch { /* 무시 */ }
}

/* ── 대시보드 ─────────────────────────────── */

async function refreshDashboard() {
    try {
        const res = await fetch("/api/dataset_status");
        const status = await res.json();
        renderDashboard(status);
    } catch { /* 무시 */ }
}

function renderDashboard(status) {
    const container = document.getElementById("dashboard");
    const TARGET = 500;
    let total = 0;

    let html = "";
    for (const [cls, count] of Object.entries(status)) {
        total += count;
        const pct = Math.min(100, (count / TARGET) * 100);
        html +=
            '<div class="dashboard-item">' +
            '<div class="label"><span>' +
            cls +
            "</span><span>" +
            count +
            "</span></div>" +
            '<div class="progress-bar"><div class="fill" style="width:' +
            pct +
            '%"></div></div></div>';
    }
    html +=
        '<div class="dashboard-total">합계: ' + total + " / " + (TARGET * 9) + "</div>";
    container.innerHTML = html;

    document.querySelectorAll(".class-btn .count").forEach((el) => {
        const cls = el.closest(".class-btn").dataset.class;
        if (status[cls] !== undefined) {
            el.textContent = status[cls];
        }
    });
}

/* ── 상태 표시 ─────────────────────────────── */

function updateStatus(msg) {
    document.getElementById("statusBar").textContent = msg;
}
