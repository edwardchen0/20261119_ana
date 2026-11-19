import streamlit as st
from transformers import pipeline

# 設定網頁標題與主頁
st.set_page_config(page_title="中文產品評論情緒分析器", page_icon="📊", layout="centered")

st.title("📊 中文產品評論情緒分析器")
st.write("輸入您的產品使用評語，模型將會分析情緒並自動評定 1 ~ 5 顆星！")

# 快取模型以加快後續載入速度
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None
    )

try:
    with st.spinner("載入預訓練情緒模型中，請稍候..."):
        engine_pipeline = load_model()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，錯誤訊息: {e}")

def analyze_sentiment_optimized(text):
    text_clean = text.strip()
    if not text_clean:
        return 3

    # 1. 取得模型各類別的完整機率分佈
    model_outputs = engine_pipeline(text_clean)[0]
    probs = {item['label']: item['score'] for item in model_outputs}

    pos_prob = probs.get('positive', 0.0)
    neg_prob = probs.get('negative', 0.0)
    neu_prob = probs.get('neutral', 0.0)

    # 2. 計算基礎情緒連續得分
    base_score = (pos_prob * 1.0) + (neu_prob * 0.0) + (neg_prob * -1.0)

    # 3. 中文修飾詞語意強度調整
    modifier_multiplier = 1.0
    weak_positive_words = ['還行', '還可以', '還好', '算方便', '將就', '勉強']
    strong_positive_words = ['超', '極', '最', '完美', '非常', '滿分', '無敵']

    if any(w in text_clean for w in weak_positive_words):
        modifier_multiplier = 0.3
    elif any(s in text_clean for s in strong_positive_words):
        modifier_multiplier = 1.3

    final_score = base_score * modifier_multiplier
    final_score = max(-1.0, min(1.0, final_score))

    # 4. 連續得分映射到 1 到 5 星級
    star_rating_continuous = 3.0 + (final_score * 2.0)
    star_rating = int(round(star_rating_continuous))
    return max(1, min(5, star_rating))

# 使用者輸入區
user_input = st.text_area("請輸入產品評語 (例如：『用起來還可以，出貨偏慢』):", height=100)

if st.button("進行情緒分析", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ 請輸入評語內容！")
    else:
        with st.spinner("正在分析中..."):
            score = analyze_sentiment_with_model = analyze_sentiment_optimized(user_input)
            
        # 視覺化輸出等級結果
        st.markdown("--- ")
        st.subheader("📊 分析結果")
        
        # 依據星星數顯示不同的卡片樣式
        star_display = "⭐" * score + "☆" * (5 - score)
        st.markdown(f"### 評分等級： {star_display} ({score} / 5 分)")
        
        # 各評分底下的提示語意說明
        status_messages = {
            1: ("🔴 非常差", "使用者對產品極度不滿，需高度關注！"),
            2: ("🟠 不佳", "使用者體驗較差，有明顯缺陷。"),
            3: ("🟡 普通", "情緒偏向中性或『還行』，滿意度一般。"),
            4: ("🟢 良好", "使用者體驗滿意，屬於正面回饋。"),
            5: ("🔵 非常好", "極高評價！使用者非常滿意您的服務。")
        }
        
        status, desc = status_messages[score]
        st.info(f"**情緒狀態**: {status} — {desc}")
