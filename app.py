import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


st.title("Analisis Sentimen Pandangan Masyarakat terhadap Kebijakan Pertamina")
st.subheader("Pasca Kasus Korupsi yang Terjadi di Indonesia pada Media Sosial Twitter")
st.caption("Metode: Naive Bayes")

# Load model dan vectorizer
model = joblib.load("model_nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")
tfidf = joblib.load("tfidf.pkl")

# Upload file Excel
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx) yang berisi kolom 'text'", type=["xlsx"])

if uploaded_file:
    # Baca data
    df = pd.read_excel(uploaded_file)
    st.write("📄 Data yang diupload:")
    st.dataframe(df)

    # Pastikan ada kolom 'text'
    if 'text' in df.columns:
        # Preprocessing: TF-IDF
        text = df['text'].astype(str)
        bow = vectorizer.transform(text)
        tfidf_data = tfidf.transform(bow)

        # Prediksi
        prediction = model.predict(tfidf_data)
        df['sentimen'] = prediction

        # Tampilkan hasil prediksi
        st.write("✅ Hasil Prediksi Sentimen:")
        st.dataframe(df[['text', 'sentimen']])

        # Visualisasi: Bar Chart
        st.write("📊 Distribusi Sentimen:")
        st.bar_chart(df['sentimen'].value_counts())

        # Unduh hasil prediksi
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button(
            label="📥 Download Hasil Prediksi sebagai CSV",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv',
        )
    else:
        st.warning("⚠️ Kolom 'text' tidak ditemukan di file. Harus ada kolom bernama 'text'.")
else:
    st.info("👆 Silakan upload file Excel untuk mulai analisis.")