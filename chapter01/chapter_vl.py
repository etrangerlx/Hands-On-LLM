from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

model_path = "/home/etranger/models/Qwen3-VL-2B-Instruct"

# ===================== 1. 加载模型 =====================
print("=" * 50)
print("正在加载 Qwen3-VL-2B-Instruct 模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",  # RTX 3060 12GB 显存，2B 模型完全够用
    torch_dtype="auto",
    trust_remote_code=True,
)
print("✅ 模型加载完成\n")

# 查看模型结构（输出很长，可注释掉只保留关键部分）
print("=" * 50)
print("模型结构概览（前 8 层）:")
print(model)
print()

# ===================== 2. 加载 Processor =====================
# Qwen3VL 使用 AutoProcessor 统一处理文本 + 图像
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 查看视觉相关的 special tokens
print("=" * 50)
print("视觉相关 Special Tokens:")
print(f"  image_token:       {processor.image_token!r}")
print(f"  vision_start_token:{processor.vision_start_token!r}")
print(f"  vision_end_token:  {processor.vision_end_token!r}")
print(f"  image_token_id:    {processor.image_token_id}")
print()

# ===================== 3. 准备多模态输入 =====================
# Qwen3VL 使用结构化内容格式（list of dicts），而非 <img> 标签
image_path = "/home/rainbow/workspace/Hands-On-LLM/chapter01/cat.jpg"

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "介绍以下这张图片"},
        ],
    },
]
print(processor.chat_template)
# apply_chat_template 会将结构化 content 展开为带 <|vision_start|><|image_pad|><|vision_end|> 的文本
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,# 会在末尾添加 <|im_start|>assistant\n
)

print("=" * 50)
print("Chat template 渲染后的文本:")
print(text)
print()

# ===================== 4. Processor 预处理 =====================
# 传入文本和图像，processor 会：
#   - tokenize 文本
#   - 加载图像并进行视觉编码（resize、patchify、embed）
#   - 生成 image_grid_thw 记录图像网格尺寸
raw_image = Image.open(image_path)
model_inputs = processor(
    text=[text],
    images=[raw_image],
    return_tensors="pt",
).to(model.device)

print("=" * 50)
print("模型输入信息:")
print(f"  input_ids shape:    {model_inputs.input_ids.shape}")        # [1, seq_len]
print(f"  pixel_values shape: {model_inputs.pixel_values.shape}")     # [num_patches, hidden_dim]
print(f"  image_grid_thw:     {model_inputs.image_grid_thw}")          # [1, T, H, W] 图像网格
print(f"  image token 数量:   {(model_inputs.input_ids[0] == processor.image_token_id).sum().item()}")
print()

# ===================== 5. 生成回复 =====================
print("=" * 50)
print("模型正在生成回复...")
print("-" * 50)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

# 截取生成的部分（去掉输入 tokens）
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码输出
response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(response)
print("=" * 50)
print("✅ 推理完成!")
