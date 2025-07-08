# ==============================================================================
# 1. IMPORT PUSTAKA YANG DIBUTUHKAN
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
# Pustaka DecisionTreeClassifier dari scikit-learn tidak diimpor karena kita membuatnya sendiri.


# ==============================================================================
# 2. Syntax Decision Tree (TETAP SAMA)
# ==============================================================================

class Node:
    """
    Kelas untuk merepresentasikan sebuah simpul (node) dalam decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # Indeks fitur untuk pemisahan
        self.threshold = threshold          # Nilai ambang batas untuk pemisahan
        self.left = left                    # Node anak kiri (jika fitur <= threshold)
        self.right = right                  # Node anak kanan (jika fitur > threshold)
        self.value = value                  # Nilai prediksi jika ini adalah leaf node

    def is_leaf_node(self):
        return self.value is not None

class CustomDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.classes_ = None

    def _entropy(self, y):
        """Menghitung entropy dari sebuah set label."""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """Menemukan label yang paling umum di sebuah node (untuk leaf node)."""
        counter = Counter(y)
        if not counter:
            return None 
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _information_gain(self, y, X_column, threshold):
        """Menghitung Information Gain untuk sebuah split."""
        parent_entropy = self._entropy(y)
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _best_split(self, X, y, feat_idxs):
        """Mencari split terbaik (fitur & threshold) dengan information gain tertinggi."""
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        """Membangun pohon keputusan secara rekursif."""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Pastikan n_features tidak lebih besar dari jumlah fitur aktual
        current_n_features = min(n_feats, self.n_features if self.n_features is not None else n_feats)
        feat_idxs = np.random.choice(n_feats, current_n_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is not None:
            left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
            right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
            if len(left_idxs) > 0 and len(right_idxs) > 0:
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_feat, best_thresh, left, right)
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)
    
    def _traverse_tree(self, x, node):
        """Melakukan navigasi pohon untuk memprediksi satu sampel data."""
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        """Fungsi utama untuk melatih model."""
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """Fungsi utama untuk melakukan prediksi pada data baru."""
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


# ==============================================================================
# 3. FUNGSI-FUNGSI APLIKASI STREAMLIT (DENGAN PERBAIKAN)
# ==============================================================================

@st.cache_data
def train_all_models(_X, _y, test_size_ratio): # test_size_ratio akan tetap ada sebagai parameter tapi nilai dikunci
    """
    Fungsi untuk membagi data, melatih model, dan mengembalikan HANYA data yang bisa disimpan.
    Objek model tidak dikembalikan untuk menghindari error serialisasi.
    """
    st.info(f"Melakukan split data dengan rasio test {test_size_ratio*100:.0f}% dan melatih model...")
    
    if hasattr(_X, "toarray"):
        st.warning("Mengonversi matriks TF-IDF ke dense array untuk Custom Decision Tree. Proses ini mungkin lambat & memakan memori.")
        X_dense = _X.toarray()
    else:
        X_dense = _X
        
    y_array = _y if isinstance(_y, np.ndarray) else _y.to_numpy()

    # --- Penggunaan rasio split tetap 80:20 dan random_state=42 ---
    X_train, X_test, y_train, y_test = train_test_split(X_dense, y_array, test_size=test_size_ratio, random_state=42)
    X_train_sparse, X_test_sparse, _, _ = train_test_split(_X, y_array, test_size=test_size_ratio, random_state=42)

    # --- Melatih Custom Decision Tree ---
    st.write("Melatih Custom Decision Tree...")
    dt_model = CustomDecisionTree(max_depth=10) 
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    st.write("✔️ Pelatihan Custom Decision Tree selesai.")
    
    # --- Melatih Naive Bayes ---
    st.write("Melatih Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_sparse, y_train)
    y_pred_nb = nb_model.predict(X_test_sparse)
    st.write("✔️ Pelatihan Naive Bayes selesai.")
    
    results = {
        'y_test': y_test,
        'y_pred_dt': y_pred_dt,
        'y_pred_nb': y_pred_nb,
        'dt_classes': dt_model.classes_,
        'nb_classes': nb_model.classes_,
    }
    return results

def show_evaluasi_page():
    """
    Fungsi untuk menampilkan halaman evaluasi, disesuaikan dengan return value baru.
    """
    st.title("📊 Evaluasi & Perbandingan Model")

    if 'tfidf_matrix' in st.session_state and 'preprocessed_df' in st.session_state:
        st.subheader("⚙️ Pengaturan Model")
        # --- Menghilangkan dropdown dan langsung mengatur rasio split ---
        selected_test_size = 0.2 # Kunci rasio test 20%
        st.info(f"Menggunakan rasio pembagian data **80% Latih : 20% Uji** secara default.")
        st.divider()
        # --- Akhir Perubahan ---

        X = st.session_state['tfidf_matrix']
        y_original = st.session_state['preprocessed_df']['sentimen'] 

        if y_original.dtype == 'object':
            y, class_names = pd.factorize(y_original)
            if 'sentiment_mapping' not in st.session_state:
                st.session_state['sentiment_mapping'] = {i: name for i, name in enumerate(class_names)}
        else:
            y = y_original.values
            # Jika y sudah numerik, asumsikan class_names bisa diambil dari unique values
            class_names = np.unique(y)
            if 'sentiment_mapping' not in st.session_state:
                st.session_state['sentiment_mapping'] = {i: str(i) for i in class_names} # Default mapping jika numerik
        
        # --- Kirim 'selected_test_size' ke fungsi training (nilai sudah terkunci) ---
        eval_results = train_all_models(X, y, test_size_ratio=selected_test_size)
        
        # Simpan hasil evaluasi ke session_state agar bisa diakses oleh halaman lain (misal: Klasifikasi DT)
        st.session_state['model_results'] = eval_results

        acc_dt = accuracy_score(eval_results['y_test'], eval_results['y_pred_dt'])
        acc_nb = accuracy_score(eval_results['y_test'], eval_results['y_pred_nb'])
        
        cm_dt = confusion_matrix(eval_results['y_test'], eval_results['y_pred_dt'], labels=eval_results['dt_classes'])
        cm_nb = confusion_matrix(eval_results['y_test'], eval_results['y_pred_nb'], labels=eval_results['nb_classes'])

        st.subheader("📈 Hasil Evaluasi")
        st.markdown(f"#### 🌳 **Custom Decision Tree Akurasi:** `{acc_dt:.2%}`")
        st.markdown(f"#### 🤖 **Naive Bayes Akurasi:** `{acc_nb:.2%}`")

        labels_dt = eval_results['dt_classes']
        labels_nb = eval_results['nb_classes']
        if 'sentiment_mapping' in st.session_state:
            mapping = st.session_state['sentiment_mapping']
            display_labels_dt = [mapping.get(i, str(i)) for i in labels_dt]
            display_labels_nb = [mapping.get(i, str(i)) for i in labels_nb]
        else:
            display_labels_dt = [str(l) for l in labels_dt]
            display_labels_nb = [str(l) for l in labels_nb]

        st.subheader("📊 Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Custom Decision Tree")
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', xticklabels=display_labels_dt, yticklabels=display_labels_dt, ax=ax1)
            ax1.set_xlabel("Prediksi")
            ax1.set_ylabel("Aktual")
            st.pyplot(fig1)

        with col2:
            st.markdown("##### Naive Bayes")
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', xticklabels=display_labels_nb, yticklabels=display_labels_nb, ax=ax2)
            ax2.set_xlabel("Prediksi")
            ax2.set_ylabel("Aktual")
            st.pyplot(fig2)

        st.subheader("📄 Classification Report")
        st.markdown("##### 🌳 Custom Decision Tree")
        report_dt = classification_report(eval_results['y_test'], eval_results['y_pred_dt'], target_names=display_labels_dt, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report_dt).transpose())

        st.markdown("##### 🤖 Naive Bayes")
        report_nb = classification_report(eval_results['y_test'], eval_results['y_pred_nb'], target_names=display_labels_nb, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report_nb).transpose())
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu di halaman **Preprocessing**.")
        st.info("Pastikan Anda telah mengupload dan memproses data di halaman **Preprocessing** sebelum melatih model.")


# ==============================================================================
# 4. JALANKAN APLIKASI
# ==============================================================================
if __name__ == '__main__':
    # Untuk pengujian file ini secara mandiri, Anda bisa membuat data dummy seperti di bawah.
    if 'tfidf_matrix' not in st.session_state:
        st.info("Ini adalah halaman evaluasi. Untuk menjalankannya, data harus diproses terlebih dahulu.")
        st.info("Silakan jalankan dari halaman utama aplikasi Streamlit Anda.")
    
    show_evaluasi_page()