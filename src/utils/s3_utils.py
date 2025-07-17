import os
import boto3
from typing import Optional
import torch
from .env_utils import get_s3_config, is_s3_enabled


class S3Handler:
    """S3 저장소 핸들러"""
    
    def __init__(self, bucket_name: Optional[str] = None, region_name: Optional[str] = None):
        """
        S3 핸들러 초기화
        
        Args:
            bucket_name: S3 버킷 이름 (None이면 환경 변수에서 가져옴)
            region_name: AWS 리전 (None이면 환경 변수에서 가져옴)
        """
        # 환경 변수에서 설정 가져오기
        s3_config = get_s3_config()
        
        self.bucket_name = bucket_name or s3_config['bucket_name']
        self.region_name = region_name or s3_config['region_name']
        
        # S3가 활성화되어 있는지 확인
        if not self.bucket_name:
            raise ValueError("S3 버킷 이름이 설정되지 않았습니다. .env 파일에서 S3_BUCKET_NAME을 설정하세요.")
        
        # AWS 자격 증명 설정
        aws_access_key_id = s3_config['access_key_id']
        aws_secret_access_key = s3_config['secret_access_key']
        
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # 환경 변수나 IAM 역할을 사용
            self.s3_client = boto3.client('s3', region_name=self.region_name)
        
        print(f"S3 핸들러 초기화 완료: 버킷={self.bucket_name}, 리전={self.region_name}")
    
    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """파일을 S3에 업로드"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            print(f"파일 업로드 완료: {local_path} -> s3://{self.bucket_name}/{s3_path}")
            return True
        except Exception as e:
            print(f"S3 업로드 실패: {e}")
            return False
    
    def download_file(self, s3_path: str, local_path: str) -> bool:
        """S3에서 파일 다운로드"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_path, local_path)
            print(f"파일 다운로드 완료: s3://{self.bucket_name}/{s3_path} -> {local_path}")
            return True
        except Exception as e:
            print(f"S3 다운로드 실패: {e}")
            return False
    
    def save_model(self, model: torch.nn.Module, local_path: str, s3_path: str) -> bool:
        """모델을 로컬에 저장하고 S3에 업로드"""
        try:
            # 로컬에 저장
            torch.save(model.state_dict(), local_path)
            print(f"모델 로컬 저장 완료: {local_path}")
            
            # S3에 업로드
            return self.upload_file(local_path, s3_path)
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, model: torch.nn.Module, s3_path: str, local_path: str) -> bool:
        """S3에서 모델 다운로드하고 로드"""
        try:
            # S3에서 다운로드
            if self.download_file(s3_path, local_path):
                # 모델 로드
                model.load_state_dict(torch.load(local_path))
                print(f"모델 로드 완료: {local_path}")
                return True
            return False
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> list:
        """S3 버킷의 파일 목록 조회"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"파일 목록 조회 실패: {e}")
            return []
    
    def delete_file(self, s3_path: str) -> bool:
        """S3에서 파일 삭제"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_path)
            print(f"파일 삭제 완료: s3://{self.bucket_name}/{s3_path}")
            return True
        except Exception as e:
            print(f"S3 파일 삭제 실패: {e}")
            return False


def create_s3_handler() -> Optional[S3Handler]:
    """환경 변수를 사용하여 S3 핸들러 생성"""
    if is_s3_enabled():
        try:
            return S3Handler()
        except Exception as e:
            print(f"S3 핸들러 생성 실패: {e}")
            return None
    else:
        print("S3가 활성화되지 않았습니다. .env 파일에서 S3_BUCKET_NAME을 설정하세요.")
        return None 