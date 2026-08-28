"""학습·평가 스크립트 → SQLite(db/door_pipeline.db) 자동 기록 (DBMS 설계 3단계).

원칙
- 실패해도 학습/평가를 멈추지 않는다: 모든 공개 함수는 예외를 삼키고 경고만 출력.
- 환경변수 DOOR_DB_LOG=0 이면 완전히 비활성 (모든 함수가 None 반환).
- DB 파일이 없으면 schema.sql로 생성한다 (멱등).
- 정확도류는 JSON 산출물과 동일하게 백분율(0~100)로 저장한다.

사용 예 (02_train.py)
    from db.db_log import DBLog
    db = DBLog()
    model_id = db.register_model(name=run_name, architecture="ResNet18",
                                 in_channels=4, num_classes=8,
                                 weights_path=f"artifacts/{run_name}/model.pth")
    sess = db.start_training(dataset_name="door_real", model_id=model_id, ...)
    db.log_epoch(sess, epoch, train_loss, val_loss, val_acc, lr, elapsed)
    db.finish_training(sess, status="completed", ...)
"""

import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE_DIR, "db", "door_pipeline.db")
SCHEMA = os.path.join(BASE_DIR, "db", "schema.sql")

# 스크립트의 데이터셋 디렉터리명 → datasets.name (ingest_local.py와 동일)
DATASET_BY_DIR = {
    "datasets": "door_real",
    "datasets_factory": "door_factory",
    "datasets_aug": "door_aug",
    "datasets_aug2": "door_aug2",
    "original": "door_real",           # 03_evaluate.py의 variant 이름
    "datasets_field": "door_field",    # 현장 세션 로컬 복사본 (17_evaluate)
    "datasets_factory_collect": "door_factory_collect",  # 06 수집 로컬 복사본 = NAS 정본과 같은 데이터셋
    "datasets_factory_v2": "door_factory_collect",       # 미러의 학습·평가용 링크 뷰 (db/build_dataset.py)
}


def dataset_name_for(dir_or_variant):
    """경로의 마지막 폴더부터 상위로 올라가며 DATASET_BY_DIR에 있는 첫 이름을 데이터셋으로 본다.
    예: datasets_factory_v2/test → door_factory_collect. 어디에도 없으면 마지막 폴더명(자동 생성용)."""
    parts = [p for p in str(dir_or_variant).rstrip("/").split(os.sep) if p and p != "."]
    for key in reversed(parts):
        if key in DATASET_BY_DIR:
            return DATASET_BY_DIR[key]
    return parts[-1] if parts else str(dir_or_variant)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe(fn):
    """DB 오류를 경고로 강등. 비활성 상태면 즉시 None."""
    def wrapper(self, *a, **kw):
        if not self.enabled:
            return None
        try:
            return fn(self, *a, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[db_log] {fn.__name__} 실패 (무시): {e}", file=sys.stderr)
            if os.environ.get("DOOR_DB_LOG_DEBUG"):
                traceback.print_exc()
            return None
    return wrapper


class DBLog:
    def __init__(self, db_path=None):
        self.enabled = os.environ.get("DOOR_DB_LOG", "1") not in ("0", "false", "no")
        self.conn = None
        if not self.enabled:
            return
        try:
            path = db_path or os.environ.get("DOOR_DB_PATH") or DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            with open(SCHEMA, encoding="utf-8") as f:
                self.conn.executescript(f.read())
            self.conn.commit()
            print(f"[db_log] 기록: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[db_log] DB 연결 실패 → 기록 비활성: {e}", file=sys.stderr)
            self.enabled = False

    # ── 조회/등록 ────────────────────────────────────────────

    @_safe
    def dataset_id(self, name_or_dir, create=True, base_path=None,
                   dtype="real", description=None):
        """datasets.name → id. 없으면(옵션) 자리표시 행 생성."""
        name = dataset_name_for(name_or_dir)
        row = self.conn.execute(
            "SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        if not create:
            return None
        cur = self.conn.execute(
            "INSERT INTO datasets(name, type, description, base_path) "
            "VALUES (?, ?, ?, ?)",
            (name, dtype, description or "db_log 자동 생성 (ingest 전)",
             base_path or f"{name_or_dir}/"))
        self.conn.commit()
        return cur.lastrowid

    @_safe
    def register_model(self, name, architecture, num_classes, in_channels=4,
                       weights_path=None, input_size=None, pretrained_base=None,
                       description=None):
        """models 행 반환(이름+가중치 경로 기준 재사용)."""
        if weights_path:
            weights_path = os.path.relpath(
                os.path.abspath(os.path.join(BASE_DIR, weights_path)), BASE_DIR)
        row = self.conn.execute(
            "SELECT id FROM models WHERE name = ? AND "
            "COALESCE(weights_path,'') = COALESCE(?, '')",
            (name, weights_path)).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO models(name, architecture, in_channels, num_classes, "
            "pretrained_base, weights_path, input_size, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, architecture, in_channels, num_classes, pretrained_base,
             weights_path, input_size or "224x224", description))
        self.conn.commit()
        return cur.lastrowid

    @_safe
    def find_model(self, weights_path=None, name=None):
        """평가 스크립트용: 가중치 경로(우선) 또는 이름으로 model id."""
        if weights_path:
            rel = os.path.relpath(
                os.path.abspath(os.path.join(BASE_DIR, weights_path)), BASE_DIR)
            row = self.conn.execute(
                "SELECT id FROM models WHERE weights_path = ? "
                "ORDER BY id DESC LIMIT 1", (rel,)).fetchone()
            if row:
                return row["id"]
        if name:
            row = self.conn.execute(
                "SELECT id FROM models WHERE name = ? ORDER BY id DESC LIMIT 1",
                (name,)).fetchone()
            if row:
                return row["id"]
        return None

    @_safe
    def latest_training_session(self, model_id):
        row = self.conn.execute(
            "SELECT id FROM training_sessions WHERE model_id = ? "
            "ORDER BY id DESC LIMIT 1", (model_id,)).fetchone()
        return row["id"] if row else None

    # ── 학습 ────────────────────────────────────────────────

    @_safe
    def start_training(self, dataset_name, model_id, optimizer="Adam",
                       learning_rate=0.001, batch_size=64, max_epochs=60,
                       early_stop_patience=10, train_ratio=0.7,
                       train_count=None, test_count=None, gpu_device=None,
                       loss_function="CrossEntropyLoss", class_weights=True,
                       split_indices_path=None):
        ds_id = self.dataset_id(dataset_name)
        if ds_id is None or model_id is None:
            return None
        cur = self.conn.execute(
            "INSERT INTO training_sessions(dataset_id, model_id, optimizer, "
            "learning_rate, batch_size, max_epochs, early_stop_patience, "
            "train_ratio, train_count, test_count, gpu_device, loss_function, "
            "class_weights, split_indices_path, started_at, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'running')",
            (ds_id, model_id, optimizer, learning_rate, batch_size, max_epochs,
             early_stop_patience, train_ratio, train_count, test_count,
             gpu_device, loss_function, bool(class_weights), split_indices_path,
             _now()))
        self.conn.commit()
        return cur.lastrowid

    @_safe
    def log_epoch(self, session_id, epoch, train_loss, val_loss=None,
                  val_accuracy=None, learning_rate=None, elapsed_sec=None):
        if session_id is None:
            return None
        self.conn.execute(
            "INSERT OR REPLACE INTO training_metrics(session_id, epoch, "
            "train_loss, val_loss, val_accuracy, learning_rate, elapsed_sec) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, epoch, train_loss, val_loss, val_accuracy,
             learning_rate, elapsed_sec))
        self.conn.commit()
        return True

    @_safe
    def finish_training(self, session_id, status="completed", actual_epochs=None,
                        best_val_accuracy=None, best_val_loss=None,
                        best_epoch=None, total_time_sec=None):
        if session_id is None:
            return None
        self.conn.execute(
            "UPDATE training_sessions SET status=?, actual_epochs=?, "
            "best_val_accuracy=?, best_val_loss=?, best_epoch=?, "
            "total_time_sec=?, ended_at=? WHERE id=?",
            (status, actual_epochs, best_val_accuracy, best_val_loss,
             best_epoch, total_time_sec, _now(), session_id))
        self.conn.commit()
        return True

    # ── 평가 ────────────────────────────────────────────────

    @_safe
    def log_evaluation(self, model_id, dataset_name, eval_type, total_samples,
                       correct, accuracy, precision_macro=None,
                       recall_macro=None, f1_macro=None, confusion_matrix=None,
                       per_class_results=None, inference_time_ms=None,
                       inference_device=None, report_path=None,
                       session_id=None):
        """eval_type: in_domain | cross_domain | inference_pipeline"""
        if model_id is None:
            return None
        ds_id = self.dataset_id(dataset_name)
        if ds_id is None:
            return None
        if session_id is None:
            session_id = self.latest_training_session(model_id)
        if report_path:
            report_path = os.path.relpath(
                os.path.abspath(os.path.join(BASE_DIR, report_path)), BASE_DIR)
        cur = self.conn.execute(
            "INSERT INTO evaluation_results(session_id, model_id, dataset_id, "
            "eval_type, total_samples, correct, accuracy, precision_macro, "
            "recall_macro, f1_macro, confusion_matrix, per_class_results, "
            "inference_time_ms, inference_device, report_path, evaluated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, model_id, ds_id, eval_type, int(total_samples),
             int(correct), float(accuracy), precision_macro, recall_macro,
             f1_macro,
             json.dumps(confusion_matrix) if confusion_matrix is not None else None,
             json.dumps(per_class_results, ensure_ascii=False)
             if per_class_results is not None else None,
             inference_time_ms, inference_device, report_path, _now()))
        self.conn.commit()
        return cur.lastrowid

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None
