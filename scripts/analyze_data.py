"""
scripts/analyze_data.py

데이터 이해 도구 통합 실행 스크립트
Integrated script for running all data understanding tools
"""

import os
import sys
from pathlib import Path
import fire

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visual_verification import VisualVerificationTool
from analysis.wrong_predictions_explorer import WrongPredictionsExplorer  
from src.utils.test_image_analyzer import TestImageAnalyzer


class DataAnalysisRunner:
    """데이터 분석 도구들을 통합 실행하는 클래스"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        print(f"🔧 설정 파일: {config_path}")
    
    def run_visual_verification(self, n_train_samples: int = 5, n_test_samples: int = 5):
        """
        시각적 검증 실행 - 증강된 훈련 이미지와 테스트 조건 비교
        
        Args:
            n_train_samples: 분석할 훈련 샘플 수
            n_test_samples: 분석할 테스트 샘플 수
        """
        print("🎨 시각적 검증 도구 실행 중...")
        
        tool = VisualVerificationTool(self.config_path)
        result_path = tool.generate_comprehensive_report(n_train_samples, n_test_samples)
        
        print(f"✅ 시각적 검증 완료: {result_path}")
        return result_path
    
    def run_wrong_predictions_analysis(self, predictions_csv: str, ground_truth_csv: str = None):
        """
        오분류 분석 실행 - 잘못된 예측들의 패턴 분석
        
        Args:
            predictions_csv: 예측 결과 CSV 파일 경로
            ground_truth_csv: 정답 CSV 파일 경로 (선택적)
        """
        print("🔍 오분류 분석 도구 실행 중...")
        
        explorer = WrongPredictionsExplorer(self.config_path)
        results = explorer.generate_comprehensive_analysis(predictions_csv, ground_truth_csv)
        
        print("✅ 오분류 분석 완료")
        return results
    
    def run_test_image_analysis(self, n_samples: int = 20):
        """
        테스트 이미지 분석 실행 - 대표적이고 도전적인 샘플 선택
        
        Args:
            n_samples: 선택할 대표 샘플 수
        """
        print("📊 테스트 이미지 분석 도구 실행 중...")
        
        analyzer = TestImageAnalyzer(self.config_path)
        results = analyzer.run_comprehensive_analysis(n_samples)
        
        print("✅ 테스트 이미지 분석 완료")
        return results
    
    def run_full_analysis(self, 
                         predictions_csv: str = None,
                         ground_truth_csv: str = None,
                         n_train_samples: int = 5,
                         n_test_samples: int = 5,
                         n_representative_samples: int = 20):
        """
        전체 데이터 분석 파이프라인 실행
        
        Args:
            predictions_csv: 예측 결과 CSV (오분류 분석용, 선택적)
            ground_truth_csv: 정답 CSV (오분류 분석용, 선택적)
            n_train_samples: 시각적 검증용 훈련 샘플 수
            n_test_samples: 시각적 검증용 테스트 샘플 수
            n_representative_samples: 선택할 대표 테스트 샘플 수
        """
        print("🚀 전체 데이터 분석 파이프라인 시작...")
        print("=" * 50)
        
        results = {}
        
        # 1. 테스트 이미지 분석 (가장 우선 - 대표 샘플 식별)
        print("\n1️⃣ 테스트 이미지 분석...")
        try:
            test_analysis = self.run_test_image_analysis(n_representative_samples)
            results['test_analysis'] = test_analysis
        except Exception as e:
            print(f"⚠️ 테스트 이미지 분석 실패: {e}")
            results['test_analysis'] = None
        
        # 2. 시각적 검증 (증강 효과 확인)
        print("\n2️⃣ 시각적 검증...")
        try:
            visual_verification = self.run_visual_verification(n_train_samples, n_test_samples)
            results['visual_verification'] = visual_verification
        except Exception as e:
            print(f"⚠️ 시각적 검증 실패: {e}")
            results['visual_verification'] = None
        
        # 3. 오분류 분석 (예측 결과가 있는 경우)
        if predictions_csv:
            print("\n3️⃣ 오분류 분석...")
            try:
                wrong_pred_analysis = self.run_wrong_predictions_analysis(predictions_csv, ground_truth_csv)
                results['wrong_predictions'] = wrong_pred_analysis
            except Exception as e:
                print(f"⚠️ 오분류 분석 실패: {e}")
                results['wrong_predictions'] = None
        else:
            print("\n3️⃣ 오분류 분석 건너뜀 (예측 결과 없음)")
            results['wrong_predictions'] = None
        
        # 4. 종합 리포트 생성
        print("\n4️⃣ 종합 보고서 생성...")
        report_path = self.generate_summary_report(results)
        results['summary_report'] = report_path
        
        print("\n" + "=" * 50)
        print("✅ 전체 데이터 분석 파이프라인 완료!")
        print(f"📄 종합 보고서: {report_path}")
        
        return results
    
    def generate_summary_report(self, analysis_results: dict) -> str:
        """종합 분석 보고서 생성"""
        output_dir = Path('outputs/comprehensive_data_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'data_analysis_summary.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 종합 데이터 분석 보고서\n\n")
            f.write("이 보고서는 모델 성능 향상을 위한 데이터 이해 분석 결과를 정리합니다.\n\n")
            
            # 테스트 이미지 분석 결과
            if analysis_results.get('test_analysis'):
                f.write("## 🎯 테스트 이미지 분석\n\n")
                test_results = analysis_results['test_analysis']
                f.write(f"- **전체 분석 파일**: `{test_results.get('full_analysis', 'N/A')}`\n")
                f.write(f"- **선택된 대표 샘플**: `{test_results.get('selected_samples', 'N/A')}`\n")
                f.write(f"- **샘플 갤러리**: `{test_results.get('gallery', 'N/A')}`\n\n")
                f.write("**권장사항**: 대표 샘플들을 시각적 검증과 모델 테스트에 활용하세요.\n\n")
            
            # 시각적 검증 결과
            if analysis_results.get('visual_verification'):
                f.write("## 🎨 시각적 검증\n\n")
                f.write(f"- **비교 이미지**: `{analysis_results['visual_verification']}`\n")
                f.write(f"- **분석 요약**: `outputs/visual_verification/analysis_summary.txt`\n\n")
                f.write("**권장사항**: 생성된 비교 이미지를 확인하여 증강 강도를 조정하세요.\n\n")
            
            # 오분류 분석 결과
            if analysis_results.get('wrong_predictions'):
                f.write("## 🔍 오분류 분석\n\n")
                wrong_results = analysis_results['wrong_predictions']
                if wrong_results:
                    f.write(f"- **시각화**: `{wrong_results.get('visualization', 'N/A')}`\n")
                    f.write(f"- **샘플 갤러리**: `{wrong_results.get('gallery', 'N/A')}`\n")
                    f.write(f"- **HTML 보고서**: `{wrong_results.get('html_report', 'N/A')}`\n")
                    f.write(f"- **JSON 결과**: `{wrong_results.get('json_results', 'N/A')}`\n\n")
                    f.write("**권장사항**: HTML 보고서를 확인하여 모델 개선 방향을 설정하세요.\n\n")
            
            # 다음 단계 권장사항
            f.write("## 📋 다음 단계 권장사항\n\n")
            f.write("1. **증강 조정**: 시각적 검증 결과를 바탕으로 `config.yaml`의 증강 강도 조정\n")
            f.write("2. **대표 샘플 활용**: 선택된 테스트 샘플들로 모델 성능 지속 모니터링\n")
            f.write("3. **오류 패턴 개선**: 오분류 분석에서 발견된 패턴을 바탕으로 데이터 증강 전략 수정\n")
            f.write("4. **GradCAM 분석**: 다음 단계로 GradCAM을 구현하여 모델 해석성 향상\n\n")
            
            f.write("## 📁 생성된 파일 목록\n\n")
            
            for category, result in analysis_results.items():
                if result and category != 'summary_report':
                    f.write(f"### {category}\n")
                    if isinstance(result, dict):
                        for key, path in result.items():
                            if path:
                                f.write(f"- {key}: `{path}`\n")
                    else:
                        f.write(f"- `{result}`\n")
                    f.write("\n")
        
        return str(report_path)


def main():
    """메인 함수 - Fire CLI 인터페이스"""
    fire.Fire(DataAnalysisRunner)


if __name__ == "__main__":
    main()


# 사용 예시:
# 
# 1. 전체 분석 실행:
#    python scripts/analyze_data.py run_full_analysis
#
# 2. 예측 결과와 함께 분석:
#    python scripts/analyze_data.py run_full_analysis \
#        --predictions_csv outputs/predictions/predictions_1234.csv \
#        --ground_truth_csv data/raw/metadata/train.csv
#
# 3. 개별 도구 실행:
#    python scripts/analyze_data.py run_visual_verification
#    python scripts/analyze_data.py run_test_image_analysis --n_samples 30
#    python scripts/analyze_data.py run_wrong_predictions_analysis \
#        --predictions_csv outputs/predictions/predictions_1234.csv