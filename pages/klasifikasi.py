import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Tidak perlu lagi definisi CustomDecisionTree di sini karena kita akan mengambil hasilnya dari session_state
# yang sudah dihitung di halaman 'Perbandingan Model'.

def show_decision_tree_classification():
    st.title("🌳 Detail Hasil Klasifikasi - Decision Tree")

    # Memastikan hasil pelatihan Decision Tree sudah tersedia dari halaman 'Perbandingan Model'
    if 'model_results' in st.session_state:
        st.info("Menampilkan hasil pelatihan Decision Tree dari halaman 'Perbandingan Model'.")
        
        results = st.session_state['model_results']
        y_test = results['y_test']
        y_pred_dt = results['y_pred_dt']
        dt_classes = results['dt_classes'] # Ambil kelas dari hasil yang disimpan

        # Dapatkan label kelas untuk tampilan (dari mapping yang disimpan)
        if 'sentiment_mapping' in st.session_state:
            mapping = st.session_state['sentiment_mapping']
            display_labels = [mapping.get(i, str(i)) for i in dt_classes]
        else:
            display_labels = [str(l) for l in dt_classes]

        st.subheader("📈 Hasil Evaluasi")
        acc = accuracy_score(y_test, y_pred_dt)
        st.write(f"**Akurasi Decision Tree:** `{acc:.2%}`")

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_dt, labels=dt_classes)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)

        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred_dt, target_names=display_labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())
        
    else:
        st.warning("⚠️ Data belum dilatih. Silakan jalankan proses di halaman **Perbandingan Model** terlebih dahulu untuk melatih model dan melihat hasilnya di sini.")
        st.info("Pastikan Anda telah mengupload dan memproses data di halaman **Preprocessing** sebelum melatih model.")

# Panggil fungsi untuk menampilkan halaman
show_decision_tree_classification()