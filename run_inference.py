#!/usr/bin/env python3
"""
run_inference.py

RunPod GPU 환경에서 Swift 식별자 추출 실행
- JSONL 데이터셋 입력 (instruction, input 구조)
- 학습 시 사용한 Alpaca 형식과 동일하게 추론
- 중단 시 재개 기능
- 해시 기반 체크포인트 (내용 변경 감지)
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
from llama_cpp import Llama


class SwiftIdentifierExtractor:
    """Swift 소스코드에서 식별자 추출"""

    def __init__(self,
                 base_model_path: str,
                 lora_path: str,
                 n_ctx: int = 12288,  # 학습 시 사용한 컨텍스트 크기 12288
                 n_gpu_layers: int = -1,
                 checkpoint_file: str = "checkpoint/processed.jsonl"):
        """
        Args:
            base_model_path: 베이스 모델 경로
            lora_path: LoRA 어댑터 경로
            n_ctx: 컨텍스트 크기 (학습 시 12288)
            n_gpu_layers: GPU 레이어 수 (-1 = 전체)
            checkpoint_file: 처리 완료된 파일 기록
        """
        self.checkpoint_file = checkpoint_file
        self.processed_hashes = self._load_checkpoint()

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

    def _compute_hash(self, instruction: str, input_text: str) -> str:
        """instruction + input으로 해시 생성"""
        combined = f"{instruction}::{input_text}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _load_checkpoint(self) -> Dict[str, str]:
        """체크포인트 로드

        Returns:
            Dict[hash, result]: 해시를 키로, 결과를 값으로
        """
        if not os.path.exists(self.checkpoint_file):
            return {}

        processed = {}
        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed[data['hash']] = data['result']
                    except:
                        continue

        if processed:
            print(f"✅ 체크포인트 로드: {len(processed)}개 항목 이미 처리됨")

        return processed

    def _save_checkpoint(self, content_hash: str, result: Dict[str, List[str]]):
        """체크포인트 저장"""
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        checkpoint_entry = {
            "hash": content_hash,
            "result": result
        }

        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(checkpoint_entry, ensure_ascii=False) + '\n')

        self.processed_hashes[content_hash] = result

    def _should_process(self, instruction: str, input_text: str) -> tuple:
        """처리 여부 확인

        Returns:
            (should_process: bool, cached_result: Dict or None)
        """
        current_hash = self._compute_hash(instruction, input_text)

        if current_hash in self.processed_hashes:
            return False, self.processed_hashes[current_hash]

        return True, None

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """학습 시 사용한 Alpaca 형식으로 프롬프트 생성

        형식: ### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n
        """
        inst = instruction.strip()
        inp = input_text.strip()

        if inp:
            prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

        return prompt

    def extract_identifiers(self, item: Dict[str, Any]) -> List[str]:
        """단일 항목에서 식별자 추출

        Args:
            item: {"instruction": "...", "input": "..."}
        """
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        # 체크포인트 확인
        should_process, cached_result = self._should_process(instruction, input_text)

        if not should_process:
            print(f"  💾 캐시된 결과 사용")
            return cached_result.get('identifiers', [])

        # 프롬프트 생성 (학습 형식과 동일)
        prompt = self._format_prompt(instruction, input_text)

        try:
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=8192,  # 출력은 8192면 충분
                temperature=0.1,
                top_p=0.95,
                stop=["<|endoftext|>", "###"],  # 학습 시 사용한 stop token
                echo=False
            )

            output = response['choices'][0]['text'].strip()

            # JSON 파싱
            identifiers = self._parse_output(output)

            # 결과 딕셔너리
            result = {"identifiers": identifiers}

            # 체크포인트 저장
            content_hash = self._compute_hash(instruction, input_text)
            self._save_checkpoint(content_hash, result)

            return identifiers

        except Exception as e:
            print(f"\n⚠️  에러 발생: {e}")
            return []

    def _parse_output(self, output: str) -> List[str]:
        """모델 출력에서 식별자 리스트 추출"""
        import re

        # 방법 1: JSON 블록 파싱
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

        # 방법 2: 배열만 추출
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
            filtered = [
                id for id in identifiers
                if id not in ['identifiers', 'reasoning', 'error', 'exclusions', 'evidence']
                   and len(id) > 1
                   and not id.startswith('is_')
                   and not id.startswith('This ')
            ]
            if filtered:
                return filtered[:50]
        except:
            pass

        return []

    def process_dataset(self,
                        dataset_path: str,
                        output_file: str = "output/identifiers.txt",
                        max_input_tokens: int = 10500) -> Dict[str, Any]:
        """전체 데이터셋 처리

        Args:
            dataset_path: 입력 JSONL 파일
            output_file: 출력 파일
            max_input_tokens: 최대 입력 토큰 수 (학습 시 12000 기준)
        """
        print("=" * 60)
        print("데이터셋 처리 시작")
        print("=" * 60)

        # 데이터셋 로드
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)

                        # 토큰 수 대략 추정 (문자 수 / 4)
                        instruction = item.get('instruction', '')
                        input_text = item.get('input', '')
                        combined = instruction + input_text
                        approx_tokens = len(combined) / 4

                        # 최대 토큰 수 필터링
                        if approx_tokens <= max_input_tokens:
                            dataset.append(item)
                        else:
                            print(f"⚠️  토큰 수 초과로 스킵: ~{int(approx_tokens)} tokens")
                    except json.JSONDecodeError:
                        continue

        total_items = len(dataset)

        # 처리 필요한 항목 계산
        need_processing = []
        for item in dataset:
            should_process, _ = self._should_process(
                item.get('instruction', ''),
                item.get('input', '')
            )
            if should_process:
                need_processing.append(item)

        already_processed = total_items - len(need_processing)

        print(f"전체 항목: {total_items}개")
        print(f"처리 완료: {already_processed}개")
        print(f"처리 필요: {len(need_processing)}개")
        print(f"최대 입력 토큰: ~{max_input_tokens} tokens\n")

        if len(need_processing) == 0:
            print("✅ 모든 항목이 이미 처리되었습니다!")
            return self._load_results(output_file)

        # 추론 실행
        all_identifiers = []

        with tqdm(total=total_items, desc="Processing", initial=already_processed) as pbar:
            for item in dataset:
                identifiers = self.extract_identifiers(item)
                all_identifiers.extend(identifiers)
                pbar.update(1)

                # 실시간 저장
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
            "total_items": len(self.processed_hashes),
            "processed_items": len(self.processed_hashes),
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
        default=12288,
        help='컨텍스트 크기 (기본값: 12288, 학습 시 사용)'
    )
    parser.add_argument(
        '--gpu_layers',
        type=int,
        default=-1,
        help='GPU 레이어 수 (기본값: -1 = 전체)'
    )
    parser.add_argument(
        '--max_input_tokens',
        type=int,
        default=10500,
        help='최대 입력 토큰 수 (기본값: 10500)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='체크포인트 초기화'
    )

    args = parser.parse_args()

    # 체크포인트 초기화
    if args.reset:
        checkpoint_file = "checkpoint/processed.jsonl"
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
            output_file=args.output,
            max_input_tokens=args.max_input_tokens
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