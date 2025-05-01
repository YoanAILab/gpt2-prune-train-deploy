# prune_compare.py

import torch
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 模型路径 ====
model_orig_path = "../python3_distillation/gpt2_student_v2"
model_prune_path = "./gpt2_student_v2_pruned"
tokenizer_path = model_orig_path  # tokenizer 无需剪枝

# ==== 加载 Tokenizer ====
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# ==== 加载模型 ====
print("🚀 加载原始 student_v2 模型...")
model_orig = GPT2LMHeadModel.from_pretrained(model_orig_path).to(device).eval()

print("🚀 加载剪枝后模型...")
model_prune = GPT2LMHeadModel.from_pretrained(model_prune_path).to(device).eval()

# ==== 测试 Prompt 列表 ====
prompts = [
    "Hello world",
    "The sky is",
    "I love",
    "Artificial intelligence is",
    "Python is a popular"
]

# ==== 推理函数 ====
def benchmark(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()
    logits = outputs.logits
    token_id = int(logits[0, -1].argmax())
    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True).strip()
    return token_id, token_text, (end - start) * 1000  # 返回 ms

# ==== 对比测试 ====
print("\n🧪 开始剪枝前后推理对比...\n")
for i, prompt in enumerate(prompts, 1):
    id1, text1, time1 = benchmark(model_orig, prompt)
    id2, text2, time2 = benchmark(model_prune, prompt)

    consistency = "✔️ 一致" if id1 == id2 else "❌ 不一致"
    speedup = (time1 - time2) / time1 * 100

    print(f"📝 Prompt {i}: {prompt}")
    print(f"  🟩 原始模型: [{text1}] (ID={id1}) ⏱ {time1:.2f} ms")
    print(f"  🟥 剪枝模型: [{text2}] (ID={id2}) ⏱ {time2:.2f} ms")
    print(f"  🔍 结果一致性: {consistency}")
    print(f"  ⚡️ 推理加速: {speedup:.1f}%\n")
