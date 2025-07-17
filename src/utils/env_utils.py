import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_env_variables():
    """환경 변수 로드"""
    # 프로젝트 루트 디렉터리 찾기
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    
    # .env 파일이 존재하면 로드
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"경고: .env 파일을 찾을 수 없습니다. {env_path}")
        print("기본 환경 변수를 사용합니다.")


def get_s3_config() -> dict:
    """S3 설정 가져오기"""
    load_env_variables()
    
    return {
        'enabled': os.getenv('S3_BUCKET_NAME') is not None,
        'bucket_name': os.getenv('S3_BUCKET_NAME', ''),
        'region_name': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
    }


def get_wandb_config() -> dict:
    """Weights & Biases 설정 가져오기"""
    load_env_variables()
    
    return {
        'api_key': os.getenv('WANDB_API_KEY', ''),
        'project': os.getenv('WANDB_PROJECT', 'document-classification')
    }


def get_cuda_config() -> dict:
    """CUDA 설정 가져오기"""
    load_env_variables()
    
    return {
        'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', '0')
    }


def is_s3_enabled() -> bool:
    """S3가 활성화되어 있는지 확인"""
    s3_config = get_s3_config()
    return s3_config['enabled'] and s3_config['bucket_name'] != ''


def get_required_env_vars() -> list:
    """필수 환경 변수 목록 반환"""
    return [
        'WANDB_API_KEY',
        'WANDB_PROJECT'
    ]


def check_required_env_vars() -> bool:
    """필수 환경 변수가 설정되어 있는지 확인"""
    load_env_variables()
    required_vars = get_required_env_vars()
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"경고: 다음 필수 환경 변수가 설정되지 않았습니다: {missing_vars}")
        print("일부 기능이 제한될 수 있습니다.")
        return False
    
    return True 