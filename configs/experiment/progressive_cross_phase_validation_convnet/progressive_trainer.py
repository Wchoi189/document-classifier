import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from icecream import ic
import fire

class ProgressiveTrainer:
    """연속적인 점진적 훈련 실행기"""
    
    def run_full_pipeline(self, 
                          start_phase: int = 1,
                          log_file: str = "progressive_training.log"):
        """
        전체 점진적 훈련 파이프라인 실행
        
        Args:
            start_phase: 시작할 단계 (1, 2, 또는 3)
            log_file: 로그 파일명
        """
        ic("🚀 점진적 훈련 파이프라인 시작")
        ic(f"시작 시간: {datetime.now()}")
        
        phases = [
            (1, "phase1_rotation_mild", "Phase 1: 마일드 회전 훈련 (±20°)"),
            (2, "phase2_rotation_variety", "Phase 2: 다양한 회전 훈련 (±60°)"),
            (3, "phase3_rotation_full", "Phase 3: 풀 회전 훈련 (±90°)")
        ]
        
        for phase_num, experiment_name, description in phases:
            if phase_num < start_phase:
                ic(f"⏭️ Phase {phase_num} 건너뛰기 (시작 단계: {start_phase})")
                continue
                
            ic(f"📊 {description}")
            success = self._run_phase(experiment_name, phase_num, log_file)
            
            if not success:
                ic(f"❌ Phase {phase_num} 실패, 파이프라인 중단")
                return False
                
            ic(f"✅ Phase {phase_num} 완료")
        
        ic("🎉 점진적 훈련 파이프라인 완료!")
        ic(f"종료 시간: {datetime.now()}")
        return True
    
    def _run_phase(self, experiment_name: str, phase_num: int, log_file: str) -> bool:
        """단일 단계 실행"""
        
        # --- MODIFIED ---
        # The command is now built using --config-dir and --config-name
        # The .yaml extension is added to the experiment_name.
        cmd = [
            "python", 
            "scripts/train.py",
            "--config-dir=configs/experiment/progressive_cross_phase_validation",
            f"--config-name={experiment_name}.yaml"
        ]
        # --- END MODIFICATION ---

        ic(f"실행 명령어: {' '.join(cmd)}") # Added for better logging

        try:
            with open(log_file, "a") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Phase {phase_num}: {experiment_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n") # Log the command
                f.write(f"시작 시간: {datetime.now()}\n")
                f.write(f"{'='*50}\n\n")
                
                # Using sys.stdout and sys.stderr to show logs in real-time
                # while also writing to the file.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path.cwd()
                )
                
                # Read and print output line by line
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        sys.stdout.write(line)
                        f.write(line)
                
                process.wait() # Wait for the process to complete
                
                f.write(f"\n\n{'='*50}\n")
                f.write(f"완료 시간: {datetime.now()}\n")
                f.write(f"리턴 코드: {process.returncode}\n")
                f.write(f"{'='*50}\n")
            
            return process.returncode == 0
            
        except Exception as e:
            ic(f"Phase {phase_num} 실행 중 오류: {e}")
            return False

if __name__ == "__main__":
    fire.Fire(ProgressiveTrainer)