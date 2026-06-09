import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Set Page Config
st.set_page_config(page_title="Wine Customer Analytics", page_icon="🍷", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    rf_model = pickle.load(open('model_rf (1).pkl', 'rb'))
    rf_smt_model = pickle.load(open('model_rf_smt.pkl', 'rb'))
    nb_model = pickle.load(open('model_nb.pkl', 'rb'))
    reg_model = pickle.load(open('model_reg.pkl', 'rb'))
    reg_scaler = pickle.load(open('reg_scaler.pkl', 'rb'))
    cm_data = pickle.load(open('cm_data.pkl', 'rb'))
    metadata = pickle.load(open('metadata.pkl', 'rb'))
    df_clean = pd.read_csv('df_clean_dashboard.csv')
    return rf_model, rf_smt_model, nb_model, reg_model, reg_scaler, cm_data, metadata, df_clean

rf_model, rf_smt_model, nb_model, reg_model, reg_scaler, cm_data, metadata, df_clean = load_artifacts()

segments = {0: 'The Uninterested', 1: 'Premium Wine Enthusiasts', 2: 'Budget Wine Loyalists', 3: 'Potential Wine Converts'}

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["Prediksi", "Visualisasi & Evaluasi"])

if menu == "Prediksi":
    st.title("🔮 Prediksi Segmen Pelanggan")
    use_smote = st.checkbox("Gunakan Model SMOTE (Lebih Akurat untuk Kelas Minoritas)", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Income (Euro)", value=50000)
        age = st.slider("Age", 18, 100, 40)
        edu = st.selectbox("Education", [0, 1, 2], format_func=lambda x: ["Undergraduate", "Graduate", "Postgraduate"][x])
    with col2:
        parent = st.radio("Status Orang Tua?", [0, 1], format_func=lambda x: "Bukan" if x==0 else "Ya")
        kids = st.number_input("Jumlah Anak", 0, 10, 1)
        loyal = st.number_input("Loyalitas (Bulan)", value=20.0)

    if st.button("Predict"):
        input_data = np.array([[income, age, edu, parent, kids, loyal]])
        model = rf_smt_model if use_smote else rf_model
        pred = model.predict(input_data)[0]
        st.success(f"Segmen Pelanggan: **{segments[pred]}**")

else:
    st.title("📊 Visualisasi & Evaluasi")
    tab1, tab2, tab3 = st.tabs(["K-Means Evaluation", "Classification (SMOTE) Evaluation", "PCA Cluster Visualization"])

    with tab1:
        st.subheader("Metrik Evaluasi Klaster K-Means")
        col_ev1, col_ev2 = st.columns(2)
        
        # Elbow Plot
        fig_el, ax_el = plt.subplots()
        ax_el.plot(metadata['means_k'], metadata['inertias'], marker='o', color='royalblue')
        ax_el.set_title('Elbow Method (Inertia)')
        ax_el.set_xlabel('Number of Clusters (K)')
        ax_el.grid(True)
        col_ev1.pyplot(fig_el)
        
        # Silhouette Plot
        fig_sil, ax_sil = plt.subplots()
        ax_sil.plot(metadata['means_k'], metadata['sil_scores'], marker='s', color='crimson')
        ax_sil.set_title('Silhouette Score per K')
        ax_sil.set_xlabel('Number of Clusters (K)')
        ax_sil.grid(True)
        col_ev2.pyplot(fig_sil)
        st.info(f"Nilai K terbaik secara matematis: k={metadata['best_k_val']}. Namun k=4 digunakan untuk kebutuhan bisnis.")

    with tab2:
        st.subheader("Performa Model: Sebelum vs Sesudah SMOTE")
        fig_cm, ax_cm = plt.subplots(1, 2, figsize=(12, 5))
        ConfusionMatrixDisplay(cm_data['before'], display_labels=list(segments.values())).plot(ax=ax_cm[0], cmap='Blues', xticks_rotation=45)
        ax_cm[0].set_title('Sebelum SMOTE')
        ConfusionMatrixDisplay(cm_data['after'], display_labels=list(segments.values())).plot(ax=ax_cm[1], cmap='Greens', xticks_rotation=45)
        ax_cm[1].set_title('Sesudah SMOTE')
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.write("**Analisis:** SMOTE membantu menyeimbangkan distribusi data sehingga model lebih sensitif terhadap kelas minoritas.")

    with tab3:
        st.subheader("Sebaran Klaster PCA 2D")
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_clean, x='pca_x', y='pca_y', hue='cluster', palette='Set1', ax=ax_pca, alpha=0.7)
        ax_pca.set_title("Pemisahan Klaster Pelanggan dalam Ruang 2D")
        st.pyplot(fig_pca)

st.sidebar.markdown("---\nKelompok 7")
