#!/usr/bin/env python3
"""
run_inference.py

RunPod GPU 환경에서 Swift 식별자 추출 실행
- JSONL 데이터셋 입력
- 중단 시 재개 기능
- 진행상황 저장
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
import argparse
from llama_cpp import Llama


class SwiftIdentifierExtractor:
    """Swift 소스코드에서 식별자 추출"""

    def __init__(self,
                 base_model_path: str,
                 lora_path: str,
                 n_ctx: int = 8192,
                 n_gpu_layers: int = -1,  # -1 = 모든 레이어 GPU에 로드
                 checkpoint_file: str = "checkpoint/processed.txt"):
        """
        Args:
            base_model_path: 베이스 모델 경로
            lora_path: LoRA 어댑터 경로
            n_ctx: 컨텍스트 크기
            n_gpu_layers: GPU 레이어 수 (-1 = 전체)
            checkpoint_file: 처리 완료된 파일 기록
        """
        self.checkpoint_file = checkpoint_file
        self.processed_files = self._load_checkpoint()

        print("=" * 60)
        print("모델 로딩 중...")
        print("=" * 60)
        print(f"Base model: {base_model_path}")
        print(f"LoRA adapter: {lora_path}")
        print(f"Context size: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers if n_gpu_layers != -1 else 'ALL'}")

        try:
            self.model = Llama(
                model_path=base_model_path,
                lora_path=lora_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=4,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            print("✅ 모델 로딩 완료!\n")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise

    def _load_checkpoint(self) -> Set[str]:
        """체크포인트 파일에서 처리 완료된 파일 목록 로드"""
        if not os.path.exists(self.checkpoint_file):
            return set()

        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            processed = set(line.strip() for line in f if line.strip())

        if processed:
            print(f"✅ 체크포인트 로드: {len(processed)}개 파일 이미 처리됨")

        return processed

    def _save_checkpoint(self, filename: str):
        """처리 완료된 파일을 체크포인트에 저장"""
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
        self.processed_files.add(filename)

    def create_prompt(self, item: Dict[str, Any]) -> str:
        """추론 프롬프트 생성"""
        code = item.get('code', '')
        repo = item.get('repo', 'unknown')
        filename = item.get('filename', 'unknown')

        # 명확한 프롬프트
        prompt = f"""Analyze the following Swift code and extract all identifiers that should be excluded from obfuscation.

**Repository:** {repo}
**File:** {filename}

**Swift Code:**
```swift
{code}
```

Return ONLY a valid JSON object in this exact format:
{{"identifiers": ["identifier1", "identifier2", "identifier3"]}}

Do not include any explanations or additional text. Only return the JSON object."""

        return prompt

    def extract_identifiers(self, item: Dict[str, Any]) -> List[str]:
        """단일 항목에서 식별자 추출"""
        filename = item.get('filename', 'unknown')

        # 이미 처리된 파일 스킵
        if filename in self.processed_files:
            return []

        prompt = self.create_prompt(item)

        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a Swift code analyzer. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=2048,
                stop=["```", "\n\n\n"]  # 조기 종료로 깔끔한 출력
            )

            output = response['choices'][0]['message']['content'].strip()

            # JSON 파싱
            identifiers = self._parse_output(output)

            # 체크포인트 저장
            self._save_checkpoint(filename)

            return identifiers

        except Exception as e:
            print(f"\n⚠️  에러 발생 ({filename}): {e}")
            return []

    def _parse_output(self, output: str) -> List[str]:
        """모델 출력에서 식별자 리스트 추출 (강력한 파싱)"""
        import re

        # 방법 1: 깨끗한 JSON 블록 파싱
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = output[start_idx:end_idx + 1]

                # 주석 제거
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)

                data = json.loads(json_str)
                if 'identifiers' in data and isinstance(data['identifiers'], list):
                    identifiers = [str(id).strip() for id in data['identifiers'] if id]
                    return [id for id in identifiers if id and id != 'identifiers']
        except:
            pass

        # 방법 2: 배열만 추출 ["id1", "id2"]
        try:
            array_match = re.search(r'\[([^\]]+)\]', output)
            if array_match:
                array_str = '[' + array_match.group(1) + ']'
                identifiers = json.loads(array_str)
                return [str(id).strip() for id in identifiers if id and str(id) != 'identifiers']
        except:
            pass

        # 방법 3: 따옴표로 감싼 문자열들 추출
        try:
            identifiers = re.findall(r'"([^"]+)"', output)
            # 키워드 필터링
            filtered = [
                id for id in identifiers
                if id not in ['identifiers', 'reasoning', 'error', 'exclusions', 'evidence']
                   and len(id) > 1  # 너무 짧은 것 제외
                   and not id.startswith('is_')  # 플래그 제외
                   and not id.startswith('This ')  # 설명문 제외
            ]
            if filtered:
                return filtered[:50]  # 최대 50개
        except:
            pass

        return []

    def process_dataset(self,
                        dataset_path: str,
                        output_file: str = "output/identifiers.txt",
                        batch_size: int = 1) -> Dict[str, Any]:
        """전체 데이터셋 처리"""
        print("=" * 60)
        print("데이터셋 처리 시작")
        print("=" * 60)

        # 데이터셋 로드
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

        total_items = len(dataset)
        already_processed = len([item for item in dataset if item.get('filename') in self.processed_files])

        print(f"전체 항목: {total_items}개")
        print(f"처리 완료: {already_processed}개")
        print(f"처리 필요: {total_items - already_processed}개\n")

        if already_processed == total_items:
            print("✅ 모든 항목이 이미 처리되었습니다!")
            return self._load_results(output_file)

        # 추론 실행
        all_identifiers = []

        with tqdm(total=total_items, desc="Processing", initial=already_processed) as pbar:
            for item in dataset:
                identifiers = self.extract_identifiers(item)
                all_identifiers.extend(identifiers)
                pbar.update(1)

                # 실시간 저장 (중복 제거 후)
                if identifiers:
                    self._save_identifiers(all_identifiers, output_file)

        # 최종 결과
        unique_identifiers = sorted(list(set(all_identifiers)))
        self._save_identifiers(unique_identifiers, output_file)

        print("\n" + "=" * 60)
        print("✅ 처리 완료!")
        print("=" * 60)
        print(f"총 식별자: {len(all_identifiers)}개")
        print(f"고유 식별자: {len(unique_identifiers)}개")
        print(f"출력 파일: {output_file}")

        return {
            "total_items": total_items,
            "processed_items": total_items,
            "total_identifiers": len(all_identifiers),
            "unique_identifiers": len(unique_identifiers),
            "output_file": output_file
        }

    def _save_identifiers(self, identifiers: List[str], output_file: str):
        """식별자 목록을 파일에 저장"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        unique_sorted = sorted(list(set(identifiers)))

        with open(output_file, 'w', encoding='utf-8') as f:
            for identifier in unique_sorted:
                f.write(f"{identifier}\n")

    def _load_results(self, output_file: str) -> Dict[str, Any]:
        """기존 결과 로드"""
        if not os.path.exists(output_file):
            return {}

        with open(output_file, 'r', encoding='utf-8') as f:
            identifiers = [line.strip() for line in f if line.strip()]

        return {
            "total_items": len(self.processed_files),
            "processed_items": len(self.processed_files),
            "total_identifiers": len(identifiers),
            "unique_identifiers": len(identifiers),
            "output_file": output_file
        }


def main():
    parser = argparse.ArgumentParser(description='Swift 식별자 추출 (GPU)')

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset.jsonl',
        help='입력 JSONL 파일 (기본값: dataset.jsonl)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='models/base_model.gguf',
        help='베이스 모델 경로'
    )
    parser.add_argument(
        '--lora',
        type=str,
        default='models/lora.gguf',
        help='LoRA 어댑터 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/identifiers.txt',
        help='출력 파일 경로'
    )
    parser.add_argument(
        '--ctx',
        type=int,
        default=8192,
        help='컨텍스트 크기 (기본값: 8192)'
    )
    parser.add_argument(
        '--gpu_layers',
        type=int,
        default=-1,
        help='GPU 레이어 수 (기본값: -1 = 전체)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='체크포인트 초기화'
    )

    args = parser.parse_args()

    # 체크포인트 초기화
    if args.reset:
        checkpoint_file = "checkpoint/processed.txt"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("✅ 체크포인트 초기화됨\n")

    # 파일 존재 확인
    if not os.path.exists(args.dataset):
        print(f"❌ 데이터셋을 찾을 수 없습니다: {args.dataset}")
        return

    if not os.path.exists(args.base_model):
        print(f"❌ 베이스 모델을 찾을 수 없습니다: {args.base_model}")
        return

    if not os.path.exists(args.lora):
        print(f"❌ LoRA 어댑터를 찾을 수 없습니다: {args.lora}")
        return

    try:
        # 추출기 초기화 및 실행
        extractor = SwiftIdentifierExtractor(
            base_model_path=args.base_model,
            lora_path=args.lora,
            n_ctx=args.ctx,
            n_gpu_layers=args.gpu_layers
        )

        results = extractor.process_dataset(
            dataset_path=args.dataset,
            output_file=args.output
        )

        # 결과 요약 저장
        summary_path = args.output.replace('.txt', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n요약 파일: {summary_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다")
        print("💾 진행상황이 저장되었습니다. 다음에 다시 실행하면 이어서 처리됩니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()