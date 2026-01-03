import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    BitsAndBytesConfig
)
import numpy as np

# --- 設定：利用 RTX 3090 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用裝置: {device} (預期應為 cuda)")

# ==========================================
# 1. 載入 ASR 模型 (Whisper-Large-v3)
# ==========================================
print("正在載入 Whisper-Large-v3 (語音識別)...")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)

# ==========================================
# 2. 載入 LLM 模型 (Qwen2.5-7B)
# ==========================================
print("正在載入 Qwen2.5-7B (翻譯與潤飾)...")

llm_model_id = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    quantization_config=bnb_config, 
    device_map="auto",              
    trust_remote_code=True
)

# ==========================================
# 3. 定義核心處理邏輯
# ==========================================
def process_audio(audio_path, source_dialect, target_style):
    if audio_path is None:
        return "請先錄音或上傳檔案！", ""

    print(f"正在處理音訊: {audio_path}")
    
    # Whisper 推論
    asr_result = asr_pipe(
        audio_path, 
        generate_kwargs={"language": "chinese"} 
    )
    raw_text = asr_result["text"]
    print(f"Whisper 原始識別結果: {raw_text}")

    # LLM 翻譯/潤飾
    system_instruction = f"""
    你是由「雅言 (YaYan-AI)」專案開發的方言轉換專家。
    
    你的核心任務是：
    1. 接收使用者的方言口語輸入（原文是「{source_dialect}」）。
    2. 理解其語意，並將其精確轉換為優雅、標準的「{target_style}」。
    3. 修正語音識別可能產生的同音錯字或贅字。
    
    請注意：你是為了文化傳承與溝通效率而生，輸出必須精準、流暢且符合正體中文規範。
    直接輸出轉換結果即可，不需要自我介紹，除非使用者特別問你的名字。
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"ASR原始文字：{raw_text}"}
    ]

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.3,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    final_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return raw_text, final_response

# ==========================================
# 4. 建立 Gradio 介面
# ==========================================
with gr.Blocks(title="3090 方言語音轉換系統") as demo:
    gr.Markdown("# 🎙️ 跨方言語音轉正體中文原型機 (RTX 3090)")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="請按此說話")
            dialect_dropdown = gr.Dropdown(
                choices=["台灣口語/台灣國語", "廣東話 (粵語)", "四川話", "上海話"], 
                value="台灣口語/台灣國語", 
                label="輸入語言 (來源)"
            )
            style_dropdown = gr.Radio(
                choices=["標準新聞書面語 (正體)", "流暢口語 (正體)", "精簡摘要"], 
                value="標準新聞書面語 (正體)", 
                label="輸出風格"
            )
            submit_btn = gr.Button("開始轉換 🚀", variant="primary")

        with gr.Column(scale=1):
            raw_text_output = gr.Textbox(label="Whisper 聽到的 (原始識別)", lines=3, interactive=False)
            final_text_output = gr.Textbox(label="LLM 修正後的 (最終結果)", lines=5, interactive=False, show_copy_button=True)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, dialect_dropdown, style_dropdown],
        outputs=[raw_text_output, final_text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)