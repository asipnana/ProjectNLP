import streamlit as st
from transformers import pipeline

if "history" not in st.session_state:
    st.session_state.history = []

# 1. Konfigurasi Halaman (Simpel dan Bersih)
st.set_page_config(page_title="Review Sentiment Analysis", page_icon="📊")
st.title("Analisis Sentimen Review Produk")
st.write("Masukkan teks review produk di bawah ini untuk mengetahui apakah sentimennya Positif atau Negatif.")

# 2. Fungsi Load Model dengan Cache
@st.cache_resource
def load_model():
    return pipeline(task="text-classification", model="asipnana/tokopedia-indobert-sentiment")

with st.spinner("Mempersiapkan model..."):
    sentiment_model = load_model()

# 3. Kotak Input Review
user_review = st.text_area("Kotak Input Review:", placeholder="Ketik review di sini...")

# 4. Tombol Eksekusi
if st.button("Sentiment Analysis", use_container_width=True):
    # Validasi logical: Cegah eksekusi jika input kosong
    if not user_review.strip():
        st.error("Teks review tidak boleh kosong. Silakan isi terlebih dahulu.")
    else:
        with st.spinner("Menganalisis sentimen..."):
            # Model inference
            result = sentiment_model(user_review)
            
            # Ekstraksi output mentah
            raw_label = result[0]['label']
            score = result[0]['score']
            
            # Logika pemetaan label untuk mengantisipasi output 'LABEL_0' atau 'LABEL_1'
            if raw_label == "LABEL_0" or raw_label == "Negative":
                final_sentiment = "🔴 NEGATIVE"
            elif raw_label == "LABEL_1" or raw_label == "Positive":
                final_sentiment = "🟢 POSITIVE"
            else:
                final_sentiment = raw_label # Fallback jika format berbeda
                
            # 5. Menampilkan Output di Bawah Tombol
            st.markdown("### Hasil Analisis:")
            st.success(f"**Sentimen:** {final_sentiment}")
            st.info(f"**Tingkat Keyakinan (Score):** {score:.2%}")

            
            st.session_state.history.append({
                "text": user_review,
                "sentiment": final_sentiment,
                "score": score
            })


# History Management: Menyimpan Riwayat Prediksi
st.markdown("---")
st.subheader("📜 Riwayat Prediksi")

if len(st.session_state.history) == 0:
    st.caption("Belum ada prediksi. Masukkan review dan klik tombol untuk melihat hasilnya.")
else:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(
            f"""
            - **{item['sentiment']}** ({item['score']:.2%})
              > {item['text'][:80]}...
            """
        )