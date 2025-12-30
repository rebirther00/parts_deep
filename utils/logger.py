"""
공통 로깅 유틸리티
- 터미널과 파일에 동시 출력
- 각 스크립트별 로그 파일 생성
"""
import sys
import os
from datetime import datetime


class DualLogger:
    """터미널과 파일에 동시에 출력하는 로거"""
    
    def __init__(self, log_file_path):
        """
        Args:
            log_file_path: 로그 파일 경로
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file_path = log_file_path
    
    def write(self, message):
        """터미널과 파일에 동시 출력"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 즉시 파일에 기록
    
    def flush(self):
        """버퍼 플러시"""
        self.terminal.flush()
        self.log_file.flush()
    
    def fileno(self):
        """터미널의 파일 디스크립터 반환 (faulthandler 호환)"""
        return self.terminal.fileno()
    
    def close(self):
        """로그 파일 닫기"""
        self.log_file.close()


class DualErrorLogger:
    """stderr도 동시에 로깅"""
    
    def __init__(self, log_file, terminal):
        self.terminal = terminal
        self.log_file = log_file
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def fileno(self):
        """터미널의 파일 디스크립터 반환 (faulthandler 호환)"""
        return self.terminal.fileno()


def setup_logging(script_name, log_dir=None):
    """
    로깅 설정
    
    Args:
        script_name: 스크립트 이름 (예: "01_generate", "02_classification")
        log_dir: 로그 디렉토리 경로 (기본값: ./logs)
    
    Returns:
        log_file_path: 생성된 로그 파일 경로
    
    Usage:
        # 스크립트 시작 시
        from utils.logger import setup_logging
        log_path = setup_logging("01_generate")
        
        # 이후 모든 print() 출력이 터미널과 파일에 동시 저장됨
    """
    # 로그 디렉토리 설정
    if log_dir is None:
        # 스크립트 위치 기준으로 logs 폴더 찾기
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(script_dir, "logs")
    
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 이름 생성 (스크립트명_날짜시간.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    # DualLogger 설정
    dual_logger = DualLogger(log_file_path)
    sys.stdout = dual_logger
    
    # stderr도 동일한 파일에 로깅
    sys.stderr = DualErrorLogger(dual_logger.log_file, sys.__stderr__)
    
    # 로그 시작 헤더
    print("=" * 80)
    print(f"📝 로그 파일: {log_file_path}")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  스크립트: {script_name}")
    print("=" * 80)
    print()
    
    return log_file_path


def reinit_logging(log_file_path=None):
    """
    로깅 재초기화 (SimulationApp 등이 stdout을 변경한 후 다시 설정)
    
    Args:
        log_file_path: 기존 로그 파일 경로 (없으면 새로 생성)
    
    Usage:
        # SimulationApp 초기화 후
        from utils.logger import reinit_logging
        reinit_logging(LOG_PATH)
    """
    # 기존 로그 파일이 있으면 append 모드로 열기
    if log_file_path and os.path.exists(log_file_path):
        log_file = open(log_file_path, 'a', encoding='utf-8')
    else:
        # 새 로그 파일 생성
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"reinit_{timestamp}.log")
        log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # DualLogger 재설정
    class ReinitDualLogger:
        def __init__(self, log_file, terminal):
            self.terminal = terminal
            self.log_file = log_file
            self.log_file_path = log_file_path
        
        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()
        
        def fileno(self):
            return self.terminal.fileno()
        
        def close(self):
            self.log_file.close()
    
    # 현재 stdout을 저장하고 새 DualLogger로 교체
    current_stdout = sys.stdout
    sys.stdout = ReinitDualLogger(log_file, current_stdout)
    
    # stderr도 설정
    current_stderr = sys.stderr
    sys.stderr = ReinitDualLogger(log_file, current_stderr)
    
    print(f"\n🔄 로깅 재초기화 완료 (SimulationApp 이후)")
    
    return log_file_path


def finish_logging():
    """
    로깅 종료 (선택적)
    
    Usage:
        # 스크립트 종료 시
        from utils.logger import finish_logging
        finish_logging()
    """
    print()
    print("=" * 80)
    print(f"📅 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # stdout이 DualLogger인 경우 닫기
    if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, 'log_file_path'):
        log_path = sys.stdout.log_file_path
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"\n✅ 로그 저장 완료: {log_path}")

