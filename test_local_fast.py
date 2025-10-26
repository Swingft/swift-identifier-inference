#!/usr/bin/env python3
"""
test_local_fast.py

맥북에서 빠르게 테스트하는 최적화 버전
- 입력 길이 제한
- 작은 컨텍스트 크기
- chat_completion 사용
"""

import json
from llama_cpp import Llama


def truncate_input(text: str, max_chars: int = 8000) -> str:
    """입력 텍스트를 제한"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def test_inference():
    """빠른 추론 테스트"""

    print("=" * 60)
    print("맥북 빠른 테스트 (최적화 버전)")
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
            if i >= 3:
                break
            items.append(json.loads(line))

    print(f"✓ {len(items)}개 항목 로드")

    # 모델 로드 (작은 컨텍스트)
    print("\n[2/3] 모델 로드...")
    print("(Metal GPU + 최적화 설정)")

    model = Llama(
        model_path=BASE_MODEL,
        lora_path=LORA_MODEL,
        n_ctx=4096,  # 작은 컨텍스트 (테스트용)
        n_gpu_layers=-1,
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

        # 입력 길이 제한 (테스트용)
        input_text_short = truncate_input(input_text, max_chars=6000)

        original_len = len(input_text)
        truncated_len = len(input_text_short)

        print(f"\n[{i}/3] 항목 {i}")
        print(f"  원본 input: {original_len} chars (~{original_len // 4} tokens)")
        print(f"  축약 input: {truncated_len} chars (~{truncated_len // 4} tokens)")

        # chat_completion 사용 (더 빠름)
        messages = [
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": "user",
                "content": input_text_short
            }
        ]

        print(f"  추론 중...")
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=1024,  # 테스트용으로 줄임
            temperature=0.1,
            top_p=0.95,
            stop=["<|endoftext|>", "###"]
        )

        output = response['choices'][0]['message']['content'].strip()

        # 식별자 추출
        identifiers = extract_identifiers(output)

        if identifiers:
            print(f"  ✓ {len(identifiers)}개 식별자 발견:")
            for id in identifiers[:10]:
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
    print("\n💡 실제 RunPod에서는:")
    print("   - n_ctx=12288 사용")
    print("   - 입력 제한 없음")
    print("   - create_completion 사용 (Alpaca 형식)")


def extract_identifiers(output: str) -> list:
    """모델 출력에서 식별자 추출"""
    import re

    # 방법 1: JSON 블록
    try:
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            json_str = output[start:end + 1]
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            data = json.loads(json_str)
            if 'identifiers' in data:
                return data['identifiers']
    except:
        pass

    # 방법 2: 배열만
    try:
        array_match = re.search(r'\[([^\]]+)\]', output)
        if array_match:
            array_str = '[' + array_match.group(1) + ']'
            identifiers = json.loads(array_str)
            return [str(id).strip() for id in identifiers if id]
    except:
        pass

    # 방법 3: 따옴표
    try:
        identifiers = re.findall(r'"([^"]+)"', output)
        identifiers = [id for id in identifiers
                       if id not in ['identifiers', 'reasoning']
                       and len(id) > 1]
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
        print("  - dataset.jsonl")
    except KeyboardInterrupt:
        print("\n\n⚠️  중단됨")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
        import traceback

        traceback.print_exc()