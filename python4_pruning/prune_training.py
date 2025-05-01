# prune_training.py

import os
import torch
from torch.nn.utils import prune
from transformers import GPT2LMHeadModel

# ==== 配置 ====
model_path = "../python3_distillation/gpt2_student_v2"
save_path = "./gpt2_student_v2_pruned"
prune_ratio = 0.3  # 剪枝比例（30%）

# ==== 创建保存目录 ====
os.makedirs(save_path, exist_ok=True)

# ==== 加载小模型 ====
print("🚀 加载小模型 student_v2 ...")
model = GPT2LMHeadModel.from_pretrained(model_path)

# ==== 对所有 Linear 层应用 L1 Unstructured 剪枝 ====
print(f"✂️ 开始对 Linear 层进行 L1 剪枝，剪枝比例: {prune_ratio}")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=prune_ratio)
        # 也可以选择不保留mask，永久剪掉
        prune.remove(module, "weight")

print("✅ 剪枝完成")

# ==== 保存剪枝后模型 ====
print(f"💾 保存剪枝后模型到: {save_path}")
model.save_pretrained(save_path)

print("🏁 剪枝流程结束！")
