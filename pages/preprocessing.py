
import streamlit as st
import pandas as pd
import re
import emoji
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

normalisasi_kata = {
    'makan bergizi gratis':'makan bergizi gratis', 'mbg':'makan bergizi gratis', 'MBG':'makan bergizi gratis', 'yg': 'yang',
    'mnurutku':'menurutku', 'mnding':'mending', 'dripda':'daripada', 'gak': 'tidak', 'pj':'penjabat', 'ga': 'tidak',
    'aja': 'saja', 'ya': 'iya', 'buat': 'untuk', 'kmren': 'kemarin', 'trs': 'terus', 'gw': 'saya', 'klo': 'kalau',
    'tdk': 'tidak', 'bgt': 'banget', 'dr': 'dari', 'dgn': 'dengan', 'sm': 'sama', 'sy': 'saya', 'udh': 'sudah',
    'blm': 'belum', 'pdhl': 'padahal', 'tp': 'tetapi', 'skrg': 'sekarang', 'td': 'tadi', 'jd': 'jadi', 'lg': 'lagi',
    'mo': 'mau', 'bbrp': 'beberapa', 'krn': 'karena', 'nih': 'ini', 'sbg': 'sebagai', 'utk': 'untuk', 'kyk': 'seperti',
    'kl': 'kalau', 'dlm': 'dalam', 'dg': 'dengan', 'dpt': 'dapat', 'emg': 'memang', 'org': 'orang', 'cmn': 'cuma',
    'gmn': 'gimana', 'blg': 'bilang', 'tmn': 'teman', 'dtg': 'datang', 'mlh': 'malah', 'knp': 'kenapa',
    'sbnrnya': 'sebenarnya', 'bgitu': 'begitu', 'ntr': 'nanti', 'lgsg': 'langsung', 'gitu': 'seperti itu', 'br': 'baru',
    'bkn': 'bukan', 'klw': 'kalau', 'kpn': 'kapan', 'khh': 'kah', 'tlollllll':'tolol'
}

@st.cache_resource
def get_sastrawi_tools():
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

stemmer, stopword = get_sastrawi_tools()

def normalisasi_teks(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens_norm = [normalisasi_kata.get(token, token) for token in tokens]
    text = " ".join(tokens_norm)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

def show_text_preprocessing():
    st.title("🧹 Text Preprocessing & Muat Data")
    
    pilihan = st.radio("Pilih jenis input:", ["Upload File CSV", "Input Manual"])

    if pilihan == "Input Manual":
        input_text = st.text_area("Masukkan teks untuk diuji:")
        if st.button("Preprocess Teks Tunggal"):
            with st.spinner("Memproses..."):
                hasil = normalisasi_teks(input_text)
            st.write("🔽 Hasil Preprocessing:")
            st.success(hasil)

    elif pilihan == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type="csv")
        
        if uploaded_file:
            try:
                df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
                df_raw.columns = df_raw.columns.str.lower().str.strip()

                if 'full_text' in df_raw.columns and 'sentimen' in df_raw.columns: 
                    st.write("📄 Data Asli (Semua Data):")
                    st.dataframe(df_raw)

                    if st.button("🔄 Jalankan Preprocessing"):
                        keys_to_delete = ['raw_df', 'preprocessed_df', 'tfidf_matrix', 'vectorizer', 'model_results']
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                                
                        with st.spinner('Melakukan preprocessing dan pembobotan... Mohon tunggu...'):
                            st.session_state['raw_df'] = df_raw.copy()
                            df_processed = df_raw.copy()
                            df_processed['clean_text'] = df_processed['full_text'].apply(normalisasi_teks)
                            st.session_state['preprocessed_df'] = df_processed
                            vectorizer = TfidfVectorizer()
                            X = vectorizer.fit_transform(df_processed['clean_text'])
                            st.session_state['vectorizer'] = vectorizer
                            st.session_state['tfidf_matrix'] = X
                        st.success("Preprocessing & Pembobotan TF-IDF selesai! Data siap digunakan di seluruh aplikasi. ✅")
                        st.info("Anda bisa melihat ringkasan data di Dashboard atau melanjutkan ke halaman Klasifikasi.")
                else:
                    st.error("Kolom 'full_text' dan/atau 'sentimen' tidak ditemukan setelah dibersihkan.")
                    st.write("Nama kolom yang terdeteksi:", df_raw.columns.tolist())
            except Exception as e:
                st.error(f"Gagal memproses file CSV. Error: {e}")

    if 'preprocessed_df' in st.session_state:
        st.markdown("---")
        st.header("Hasil Proses Saat Ini")
        df_display = st.session_state['preprocessed_df']
        st.write("✅ Teks Bersih (Hasil Preprocessing):")
        st.dataframe(df_display[['full_text', 'clean_text', 'sentimen']]) 
        
        csv_data = df_display[['clean_text', 'sentimen']].to_csv(index=False).encode('utf-8') 
        st.download_button("📥 Download Hasil Preprocessing", data=csv_data, file_name="preprocessed_data.csv", mime="text/csv")

        if st.button("📊 Tampilkan Visualisasi Wordcloud"):
            with st.spinner("Membuat Wordcloud..."):
                st.subheader("☁️ Wordcloud Berdasarkan Sentimen")
                positif_text = ' '.join(df_display[df_display['sentimen'].str.lower() == 'positif']['clean_text']) 
                negatif_text = ' '.join(df_display[df_display['sentimen'].str.lower() == 'negatif']['clean_text']) 
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                if positif_text:
                    wc_pos = WordCloud(width=800, height=800, background_color='white', colormap='Blues').generate(positif_text)
                    ax1.imshow(wc_pos, interpolation='bilinear')
                    ax1.set_title('Kata Sentimen Positif', fontsize=15)
                    ax1.axis('off')
                else:
                    ax1.text(0.5, 0.5, 'Tidak ada data positif', horizontalalignment='center', verticalalignment='center')
                    ax1.axis('off')
                if negatif_text:
                    wc_neg = WordCloud(width=800, height=800, background_color='white', colormap='Reds').generate(negatif_text)
                    ax2.imshow(wc_neg, interpolation='bilinear')
                    ax2.set_title('Kata Sentimen Negatif', fontsize=15)
                    ax2.axis('off')
                else:
                    ax2.text(0.5, 0.5, 'Tidak ada data negatif', horizontalalignment='center', verticalalignment='center')
                    ax2.axis('off')
                st.pyplot(fig)
show_text_preprocessing()