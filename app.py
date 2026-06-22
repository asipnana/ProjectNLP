import streamlit as st
from transformers import pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Review Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="asipnana/tokopedia-indobert-sentiment"
    )

with st.spinner("Loading model..."):
    sentiment_model = load_model()

# =========================
# HEADER
# =========================
st.title("📊 Analisis Sentimen Review Produk")
st.write(
    "Masukkan review produk berbahasa Indonesia untuk mengetahui "
    "apakah sentimennya positif atau negatif."
)

# =========================
# INPUT
# =========================
user_review = st.text_area(
    "Review Produk",
    height=150,
    placeholder="Contoh: Pengiriman cepat, kualitas produk bagus, sangat puas..."
)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Analisis Sentimen", use_container_width=True):

    if not user_review.strip():
        st.warning("Silakan masukkan review terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):

            result = sentiment_model(user_review)

            raw_label = result[0]["label"]
            score = result[0]["score"]

            # Mapping label
            if raw_label in ["LABEL_0", "Negative", "NEGATIVE"]:
                sentiment = "NEGATIVE"
                emoji = "🔴"
            elif raw_label in ["LABEL_1", "Positive", "POSITIVE"]:
                sentiment = "POSITIVE"
                emoji = "🟢"
            else:
                sentiment = raw_label
                emoji = "⚪"

            st.markdown("---")
            st.subheader("Hasil Analisis")

            st.success(
                f"{emoji} Sentimen: **{sentiment}**"
            )

            st.info(
                f"🎯 Tingkat Keyakinan: **{score:.2%}**"
            )

            # Save history
            st.session_state.history.append({
                "text": user_review,
                "sentiment": sentiment,
                "score": score
            })

# =========================
# HISTORY
# =========================
st.markdown("---")
st.subheader("📜 Riwayat Prediksi")

if len(st.session_state.history) == 0:
    st.caption("Belum ada riwayat prediksi.")
else:

    for item in reversed(st.session_state.history[-5:]):

        preview = (
            item["text"][:100] + "..."
            if len(item["text"]) > 100
            else item["text"]
        )

        if item["sentiment"] == "POSITIVE":
            st.success(
                f"🟢 {item['sentiment']} ({item['score']:.2%})\n\n{preview}"
            )
        else:
            st.error(
                f"🔴 {item['sentiment']} ({item['score']:.2%})\n\n{preview}"
            )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("### 👥 Kelompok")

st.markdown("""
**Comparative Analysis of ResNet50, EfficientNet-B7, and DenseNet121 for Breast Cancer Ultrasound Image Classification**
**Anggota Kelompok:**
- Adisca Gandawidjaja - 2802420315
- Alicia Angelina Jusup - 2802420334
- Mathilda Rafaella Christy Nugroho — 2802415744
**Powered by:** IndoBERT base p1 
""")
