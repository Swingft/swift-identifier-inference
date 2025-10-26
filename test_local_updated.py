#!/usr/bin/env python3
"""
test_local_updated.py

맥북에서 빠르게 테스트하는 스크립트 (업데이트된 버전)
- Alpaca 형식 (instruction + input) 사용
- Metal GPU 자동 사용
- 학습 형식과 동일한 프롬프트
"""

import json
from llama_cpp import Llama


def test_inference():
    """간단한 추론 테스트"""

    print("=" * 60)
    print("로컬 테스트 시작 (업데이트된 버전)")
    print("=" * 60)

    # 설정
    BASE_MODEL = "models/base_model.gguf"
    LORA_MODEL = "models/lora.gguf"
    DATASET = "dataset.jsonl"

    # 데이터셋 로드 (처음 3개만)
    print("\n[1/3] 데이터셋 로드...")
    items = []
    with open(DATASET, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 3개만
                break
            items.append(json.loads(line))

    print(f"✓ {len(items)}개 항목 로드")

    # 모델 로드
    print("\n[2/3] 모델 로드...")
    print("(Metal GPU 자동 사용)")

    model = Llama(
        model_path=BASE_MODEL,
        lora_path=LORA_MODEL,
        n_ctx=12288,  # 학습 시 사용한 크기
        n_gpu_layers=-1,  # Metal에서 모든 레이어 GPU 사용
        n_threads=4,
        verbose=False
    )

    print("✓ 모델 로드 완료")

    # 추론 실행
    print("\n[3/3] 추론 실행...")
    print("-" * 60)

    for i, item in enumerate(items, 1):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        print(f"\n[{i}/3] 항목 {i}")
        print(f"  Instruction 길이: {len(instruction)} chars")
        print(f"  Input 길이: {len(input_text)} chars")

        # 학습 형식과 동일한 프롬프트 생성
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        print(f"  프롬프트 길이: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        # 추론
        print(f"  추론 중...")
        response = model.create_completion(
            prompt=prompt,
            max_tokens=2048,  # 테스트용으로 줄임
            temperature=0.1,
            top_p=0.95,
            stop=["<|endoftext|>", "###"],
            echo=False
        )

        output = response['choices'][0]['text'].strip()

        # 식별자 추출
        identifiers = extract_identifiers(output)

        if identifiers:
            print(f"  ✓ {len(identifiers)}개 식별자 발견:")
            for id in identifiers[:10]:  # 처음 10개만
                print(f"    - {id}")
            if len(identifiers) > 10:
                print(f"    ... 외 {len(identifiers) - 10}개")
        else:
            print(f"  ✗ 식별자를 찾을 수 없음")
            print(f"\n  원본 출력 (처음 200자):")
            print(f"  {output[:200]}")
            print("  ...")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)


def extract_identifiers(output: str) -> list:
    """모델 출력에서 식별자 추출"""
    import re

    # 방법 1: JSON 블록 찾기
    try:
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            json_str = output[start:end + 1]

            # 주석 제거
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)

            data = json.loads(json_str)
            if 'identifiers' in data:
                return data['identifiers']
    except:
        pass

    # 방법 2: 배열만 찾기
    try:
        array_match = re.search(r'\[([^\]]+)\]', output)
        if array_match:
            array_str = '[' + array_match.group(1) + ']'
            identifiers = json.loads(array_str)
            return [str(id).strip() for id in identifiers if id]
    except:
        pass

    # 방법 3: 따옴표로 감싼 문자열들 추출
    try:
        identifiers = re.findall(r'"([^"]+)"', output)
        identifiers = [id for id in identifiers if id != 'identifiers' and id != 'reasoning']
        if identifiers:
            return identifiers[:20]
    except:
        pass

    return []


if __name__ == '__main__':
    try:
        test_inference()
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
        print("\n필요한 파일:")
        print("  - models/base_model.gguf")
        print("  - models/lora.gguf")
        print("  - dataset.jsonl (Alpaca 형식: instruction, input)")
    except KeyboardInterrupt:
        print("\n\n⚠️  중단됨")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback

        traceback.print_exc()