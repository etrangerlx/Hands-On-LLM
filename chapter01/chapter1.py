from transformers import AutoModelForCausalLM, AutoTokenizer,Qwen2VLForConditionalGeneration


# 加载模型和 tokenizer （第一次加载的时候会进行下载）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/mnt/g/Models/Qwen2-VL-2B-Instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
# 查看 模型结构是什么？
print(model)
'''
Qwen2VLForConditionalGeneration(
  (model): Qwen2VLModel(
    (visual): Qwen2VisionTransformerPretrainedModel(
      (patch_embed): PatchEmbed(
        (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
      )
      (rotary_pos_emb): VisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-31): 32 x Qwen2VLVisionBlock(
          (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
          (attn): VisionAttention(
            (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            (proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (mlp): VisionMlp(
            (fc1): Linear(in_features=1280, out_features=5120, bias=True)
            (act): QuickGELUActivation()
            (fc2): Linear(in_features=5120, out_features=1280, bias=True)
          )
        )
      )
      (merger): PatchMerger(
        (ln_q): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (mlp): Sequential(
          (0): Linear(in_features=5120, out_features=5120, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=5120, out_features=1536, bias=True)
        )
      )
    )
    (language_model): Qwen2VLTextModel(
      (embed_tokens): Embedding(151936, 1536)
      (layers): ModuleList(
        (0-27): 28 x Qwen2VLDecoderLayer(
          (self_attn): Qwen2VLAttention(
            (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
            (k_proj): Linear(in_features=1536, out_features=256, bias=True)
            (v_proj): Linear(in_features=1536, out_features=256, bias=True)
            (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
            (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
            (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen2VLRMSNorm((1536,), eps=1e-06)
          (post_attention_layernorm): Qwen2VLRMSNorm((1536,), eps=1e-06)
        )
      )
      (norm): Qwen2VLRMSNorm((1536,), eps=1e-06)
      (rotary_emb): Qwen2VLRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
'''

# 这里有一个 Special token, 打印看一下是什么token呢？
tokenizer = AutoTokenizer.from_pretrained("/mnt/g/Models/Qwen2-VL-2B-Instruct")
# tokenizer.all_special_tokens
tokenizer.special_tokens_map
prompt = "讲一个猫有关的笑话？"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}]
#  想一下诗变成什么格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs)
'''
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,  99526,  46944, 100472,
         101063,   9370, 109959,  11319, 151645,    198, 151644,  77091,    198]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''
# 直接将输入反向解码
response = tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True)[0]
print(response)
'''
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
讲一个猫有关的笑话？
assistant
'''