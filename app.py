import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="製薬シンポ・AI要約ツール", page_icon="💊")

@st.cache_resource
def load_summary_model():
    model_name = "google/mt5-small"
    # pipelineを使わず、直接モデルとトークナイザーを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    return tokenizer, model

st.title("💊 製薬シンポジウムAI要約ツール")

with st.spinner("AI準備中..."):
    tokenizer, model = load_summary_model()

raw_html = st.text_area("SharePointのHTMLを貼り付けてください", height=200)

if st.button("AI要約を実行"):
    if raw_html:
        soup = BeautifulSoup(raw_html, "html.parser")
        cleaned_text = soup.get_text(separator="\n", strip=True)
        input_text = "summarize: " + cleaned_text[:800] # モデルへの明示的な指示
        
        st.subheader("📝 AI要約結果")
        with st.spinner("要約を生成中..."):
            try:
                # pipelineを使わずに直接計算する手順
                inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=800, truncation=True)
                outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.success(summary_text)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
        
        with st.expander("全文表示"):
            st.write(cleaned_text)