import os
# セキュリティ制限を完全に無効化する設定
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TRUST_REMOTE_CODE"] = "True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline
import torch

# ページ設定
st.set_page_config(page_title="製薬シンポ・AI要約ツール", page_icon="💊")

# app.py の load_summary_model 部分を以下に書き換え
@st.cache_resource
def load_summary_model():
    # 1. 辞書（Tokenizer）を明示的に読み込む
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "google/mt5-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float32
    )

    # 2. 明示的に組み立てた部品を pipeline に渡す
    return pipeline(
        "summarization", 
        model=model,
        tokenizer=tokenizer
    )

st.title("💊 製薬シンポジウムAI要約ツール")
st.write("HTMLからテキストを抽出し、AIが内容を要約します。")

# モデルのロード（要約のみに絞る）
with st.spinner("AIモデルを準備中...（初回は5分ほどかかります）"):
    summarizer = load_summary_model()

# 入力エリア
raw_html = st.text_area("SharePointのHTMLを貼り付けてください", height=200)

if st.button("AI要約を実行"):
    if raw_html:
        # 1. クリーニング（BeautifulSoup）
        soup = BeautifulSoup(raw_html, "html.parser")
        cleaned_text = soup.get_text(separator="\n", strip=True)
        
        # 2. テキストが長すぎる場合の処理（AIが処理できる長さに制限）
        input_text = cleaned_text[:800]
        
        st.subheader("📝 AI要約結果")
        with st.spinner("要約を生成中..."):
            try:
                summary = summarizer(input_text, max_length=150, min_length=40)
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"要約中にエラーが発生しました: {e}")
            
        # クリーニング後の全テキスト確認用
        with st.expander("クリーニング済みの全文を表示"):
            st.write(cleaned_text)
    else:
        st.warning("HTMLが空です。")