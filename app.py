import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import iqr
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer

# --- Helper Function for Outlier Handling ---
def find_outlier(data, column, multiplier=1.5):
    col_data = data[column]
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = data.loc[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, upper_bound

# --- Data Loading and Preprocessing (Self-contained for Streamlit) ---
@st.cache_data
def load_and_preprocess_data_full():
    url = 'https://raw.githubusercontent.com/mazayazzz/TUBES-DATMIN/refs/heads/main/marketing_campaign.csv'
    raw_df = pd.read_csv(url, sep=';')
    df_original_five_raw = raw_df.head(5).copy()

    df = raw_df.copy()

    # Capture initial metadata
    initial_rows = df.shape[0]
    initial_cols = df.shape[1]

    # Capture missing values before processing
    missing_before_series = df.isnull().sum()
    missing_before_count = missing_before_series.sum()
    missing_before = missing_before_series[missing_before_series > 0]

    # Duplicate Data Checking
    initial_duplicates_count = df.duplicated().sum()

    # Data Cleaning
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='mixed')
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df.dropna(subset=['Income'], inplace=True)

    # Remove duplicate rows after handling Income NaNs
    df.drop_duplicates(inplace=True)
    rows_after_duplicates_drop = df.shape[0]

    # Replicate the column dropping logic as per notebook
    irrel_cols_names_list = [
        'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth',
        'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
        'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'
    ]
    df = df.drop(columns=irrel_cols_names_list, errors='ignore')

    # Capture missing values after initial cleaning and duplicate removal
    missing_after_initial_series = df.isnull().sum()
    missing_after_initial_count = missing_after_initial_series.sum()
    missing_after_initial = missing_after_initial_series[missing_after_initial_series > 0]
    rows_after_initial_drop = df.shape[0] # This will now be the count after NaNs and duplicates

    # Feature Engineering
    df['AmtTotal'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                      df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['%Wine_Share'] = (100 * df['MntWines'] / df['AmtTotal']).round(1)
    df.loc[df['AmtTotal'] == 0, '%Wine_Share'] = 0
    df['Wine_Spend'] = df['MntWines']
    df['Age'] = 2014 - df['Year_Birth']
    df['Purchase_Vol'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
    df['Num_Children'] = df['Kidhome'] + df['Teenhome']
    df['Parent'] = np.where(df['Num_Children'] > 0, 'Yes', 'No')

    encoder = LabelEncoder()
    df['Parent2'] = encoder.fit_transform(df.Parent)

    max_date = pd.to_datetime('2014-12-31')
    df['Loyalitas_Bulan'] = ((max_date - df['Dt_Customer']).dt.days / 30.44).round(1)

    education_map = {'Graduation':'Graduate', 'PhD':'Postgraduate', 'Master':'Postgraduate', 'Basic':'Undergraduate', '2n':'Undergraduate'}
    df['Education'] = df.Education.map(education_map)

    edu_categories = ['Undergraduate', 'Graduate', 'Postgraduate']
    oncoder = OrdinalEncoder(categories=[edu_categories])
    df['Education2'] = oncoder.fit_transform(df[['Education']])

    df = df.drop(columns=['MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], errors='ignore')

    df_processed = df.copy()

    # Outlier handling
    df_clean_processed = df.copy()
    Income_outliers, Income_upper = find_outlier(df_clean_processed, 'Income')
    Age_outliers, Age_upper = find_outlier(df_clean_processed, 'Age')
    Wine_outliers, Wine_upper = find_outlier(df_clean_processed, 'Wine_Spend')
    AmtTotal_outliers, AmtTotal_upper = find_outlier(df_clean_processed, 'AmtTotal')

    df_clean_processed.loc[df_clean_processed['Income'].isin(Income_outliers['Income']), 'Income'] = Income_upper
    df_clean_processed.loc[df_clean_processed['Age'].isin(Age_outliers['Age']), 'Age'] = Age_upper
    df_clean_processed.loc[df_clean_processed['Wine_Spend'].isin(Wine_outliers['Wine_Spend']), 'Wine_Spend'] = Wine_upper
    if not AmtTotal_outliers.empty:
        df_clean_processed.loc[df_clean_processed['AmtTotal'].isin(AmtTotal_outliers['AmtTotal']), 'AmtTotal'] = AmtTotal_upper

    return raw_df, df_processed, df_clean_processed, df_original_five_raw, missing_before, initial_rows, initial_cols, missing_after_initial, rows_after_initial_drop, initial_duplicates_count, rows_after_duplicates_drop, missing_before_count, missing_after_initial_count

raw_df, df_processed, df_clean, df_original_five_raw, missing_before, initial_rows, initial_cols, missing_after_initial, rows_after_initial_drop, initial_duplicates_count, rows_after_duplicates_drop, missing_before_count, missing_after_initial_count = load_and_preprocess_data_full()

# Load models
rf_model = pickle.load(open('model_rf (2).pkl', 'rb'))
nb_model = pickle.load(open('model_nb (1).pkl', 'rb'))
reg_model = pickle.load(open('model_reg.pkl', 'rb'))
reg_scaler_loaded = pickle.load(open('reg_scaler.pkl', 'rb'))
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
kmeans_scaler = pickle.load(open('kmeans_scaler.pkl', 'rb'))
nb_scaler = pickle.load(open('nb_scaler.pkl', 'rb'))
cm_dict = pickle.load(open('cm_data.pkl', 'rb')) # Load confusion matrix data

# Set Page Config
st.set_page_config(page_title="Wine Customer Analytics & Predictor", page_icon="🍷", layout="wide")

# --- MAPPING SEGMEN & REKOMENDASI KOMPREHENSIF ---
segments = {
    0: 'The Uninterested',
    1: 'Premium Wine Enthusiasts',
    2: 'Budget Wine Loyalists',
    3: 'Potential Wine Converts'
}

rekomendasi = {
    0: "🎯 **Rekomendasi untuk The Uninterested:**\n* Alokasikan dana pemasaran pada level terendah (*low priority*) untuk menghemat biaya.\n* Terapkan otomatisasi email skala besar (*mass blast*) berisi promo produk umum non-wine saja.",
    1: "🎯 **Rekomendasi untuk Premium Wine Enthusiasts:**\n* Prioritaskan segmen ini! Buat program loyalitas eksklusif VIP & penawaran tier premium.\n* Berikan kuota pre-order eksklusif untuk koleksi minuman *limited edition* or undang ke acara VIP wine-tasting.",
    2: "🎯 **Rekomendasi untuk Budget Wine Loyalists:**\n* Terapkan strategi *value-for-money* seperti promo bundling ('Beli 3 Gratis 1').\n* Kirimkan voucher khusus potongan harga berbatas waktu ekonomis pada akhir pekan.",
    3: "🎯 **Rekomendasi untuk Potential Wine Converts:**\n* Gunakan taktik *cross-selling* berdasarkan komoditas belanja umum non-wine yang sering mereka borong.\n* Edukasi mereka lewat newsletter resep makanan yang cocok disajikan bersama wine (*food-wine pairing*) disertai sampel tester diskon."
}

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4341/4341772.png", width=110)
st.sidebar.title("Navigasi Aplikasi")
menu = st.sidebar.radio("Pilih Menu:", ["Prediksi Pelanggan Baru", "Eksplorasi Data & Visualisasi"])

# ==============================================================================
# MENU 1: PREDIKSI PELANGGAN
# ==============================================================================
if menu == "Prediksi Pelanggan Baru":
    st.title("🔮 Form Prediksi Segmen & Estimasi Belanja Pelanggan")
    st.markdown("Gunakan panel ini untuk menguji performa prediksi gabungan model Klasifikasi dan Regresi.")

    # Pilihan Model Klasifikasi
    pilihan_model = st.selectbox("Pilih Algoritma Klasifikasi:", ["Random Forest Classifier (Tuned)", "Gaussian Naive Bayes"])

    st.subheader("Masukkan Profil Demografi Pelanggan")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Pendapatan Tahunan (Euro):", min_value=0, value=int(df_clean['Income'].mean()), step=1000)
        age = st.slider("Usia Pelanggan:", 18, 100, int(df_clean['Age'].mean()))
        education = st.selectbox("Tingkat Pendidikan:", options=[0, 1, 2], format_func=lambda x: ["Undergraduate", "Graduate", "Postgraduate"][x], index=int(df_clean['Education2'].mode()[0]))
    with col2:
        parent = st.radio("Apakah Status Orang Tua?", options=[0, 1], format_func=lambda x: "Bukan / Tidak" if x == 0 else "Ya", index=int(df_clean['Parent2'].mode()[0]))
        num_children = st.number_input("Jumlah Anak Kandung:", min_value=0, max_value=10, value=int(df_clean['Num_Children'].mean()))
        loyalitas_default = float(int(df_clean['Loyalitas_Bulan'].mean()))

    if st.button("Jalankan Proses Analisis"):
        # Menggabungkan parameter input dengan nilai default loyalitas
        input_data = pd.DataFrame([[income, age, float(education), int(parent), num_children, loyalitas_default]],
                                  columns=['Income', 'Age', 'Education2', 'Parent2', 'Num_Children', 'Loyalitas_Bulan'])

        # Jalankan Model Klasifikasi Terpilih
        if pilihan_model == "Random Forest Classifier (Tuned)":
            pred_class = rf_model.predict(input_data)[0]
            probs = rf_model.predict_proba(input_data).max()
        else:
            input_nb = nb_scaler.transform(input_data)
            pred_class = nb_model.predict(input_nb)[0]
            probs = nb_model.predict_proba(input_nb).max()

        # Jalankan Model Regresi (Gunakan Scaler bawaan)
        scaled_input = reg_scaler_loaded.transform(input_data)
        pred_amt = max(0, reg_model.predict(scaled_input)[0])

        # Display Hasil Akhir Secara Kolaboratif (Layout Versi Pertama)
        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Hasil Segmentasi")
            st.info(f"Klaster Terprediksi: **{segments[pred_class]}**")
            st.write(f"Tingkat Keyakinan (*Confidence*): {probs:.2%}")

        with res_col2:
            st.subheader("Estimasi Total Pengeluaran")
            st.success(f"**{pred_amt:,.2f} Euro**")

        st.divider()
        st.subheader("Rekomendasi Strategi Bisnis untuk Profil Ini")
        st.markdown(rekomendasi[pred_class])

# ==============================================================================
# MENU 2: EKSPLORASI DATA & VISUALISASI
# ==============================================================================
elif menu == "Eksplorasi Data & Visualisasi":
    st.title("📊 Profil Eksplorasi Data & Detail Pra-pemrosesan")

    tab_data, tab_cluster, tab_dist = st.tabs(["Ringkasan & Metadata Dataset", "Analisis Optimalisasi K-Means & PCA", "Sebaran Fitur & Outliers"])
    features_to_cluster_eval = ['Wine_Spend', '%Wine_Share', 'Purchase_Vol', 'Loyalitas_Bulan']
    k_inputs_eval = kmeans_scaler.transform(df_clean[features_to_cluster_eval])
    df_clean['cluster_viz'] = kmeans_model.predict(k_inputs_eval)
    with tab_data:
        st.subheader("1. Ringkasan Metadata Dataset")
        metadata_summary = pd.DataFrame({
            "Metrik": [
                "Total Baris Awal Dataset (Original)",
                "Total Kolom Awal Dataset (Original)",
                "Jumlah Baris Duplikat yang Ditemukan",
                "Total Baris Setelah Drop Duplikat",
                "Total Baris Setelah Drop Baris Kosong Income & Duplikat",
                "Total Missing Values (Sebelum Penanganan)",
                "Total Missing Values (Setelah Penanganan Awal)"
            ],
            "Nilai": [
                initial_rows,
                initial_cols,
                initial_duplicates_count,
                rows_after_duplicates_drop,
                initial_rows - missing_before_count, # Corrected: Calculate rows after all initial drops based on notebook logic
                missing_before_count,
                missing_after_initial_count
            ]
        })
        st.dataframe(metadata_summary.set_index("Metrik"))

        st.subheader("2. Deteksi Missing Values Sebelum Penanganan")
        if not missing_before.empty:
            st.dataframe(missing_before.reset_index().rename(columns={'index': 'Nama Kolom', 0: 'Jumlah Missing'}))
        else:
            st.write("Aman. Tidak ada missing value sebelum manipulasi data.")

        st.subheader("3. Deteksi Missing Values Setelah Initial Cleaning & Duplikat")
        if missing_after_initial.sum() > 0:
            st.dataframe(missing_after_initial[missing_after_initial > 0].reset_index().rename(columns={'index': 'Nama Kolom', 0: 'Jumlah Missing'}))
        else:
            st.write("Sempurna! 0 baris bernilai kosong setelah penanganan data.")

        st.subheader("4. Informasi Struktur Tipe Data Objek (Data Info)")
        # Create a DataFrame for data info instead of printing raw info()
        data_info_df = pd.DataFrame({
            'Column': df_processed.columns,
            'Non-Null Count': df_processed.count().values,
            'Dtype': df_processed.dtypes.values
        })
        st.dataframe(data_info_df.set_index('Column'))

        st.subheader("5. Analisis Statistik Deskriptif Matematika")
        st.dataframe(df_processed.describe())

        st.subheader("6. Komparasi 5 Baris Data Awal")
        st.write(">> Lima Baris Pertama Data Mentah (Original Raw):")
        st.dataframe(df_original_five_raw)
        st.write(">> Lima Baris Pertama Data Hasil Preprocessing & Feature Engineering:")
        st.dataframe(df_processed.head(5))

    with tab_cluster:
        st.subheader("Metrik Evaluasi Klaster K-Means")

        # Inertias and Silhouette Scores are still calculated for informational purposes, but only Elbow plot will be shown
        inertias_eval = []
        sil_scores_eval = []
        ks = range(2, 11)

        for k in ks:
            kmeans_eval = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans_eval.fit(k_inputs_eval)
            inertias_eval.append(kmeans_eval.inertia_)
            sil_scores_eval.append(silhouette_score(k_inputs_eval, kmeans_eval.labels_))

        best_k_idx = np.argmax(sil_scores_eval)
        best_k_silhouette = ks[best_k_idx]
        st.markdown(f"💡 **Nilai K Terbaik Secara Matematis (Silhouette Score):** `k = {best_k_silhouette}`")
        st.info("Meskipun Silhouette Score tertinggi berada pada K tertentu, nilai K=4 dipertahankan dalam analisis segmentasi akhir demi kedalaman akomodasi interpretasi profil bisnis ritel.")

        # --- Plot Metode Elbow (hanya satu plot) ---
        fig_el, ax_el = plt.subplots(1, 1, figsize=(8, 5)) # Create a single subplot

        # Elbow Method using Yellowbrick (like Colab notebook)
        model_for_elbow = KMeans(n_init=10, random_state=42)
        visualizer_elbow = KElbowVisualizer(model_for_elbow, k=(1, 10), timings=True, locate_elbow=False, random_state=42, ax=ax_el)
        visualizer_elbow.fit(k_inputs_eval)
        # Add the vertical line and legend for k=4, similar to Colab notebook
        # k_scores_ are the inertia values, k=4 is at index 3 (for k=1 to 10)
        k4_score = visualizer_elbow.k_scores_[3]
        ax_el.axvline(x=4, color='black', linestyle='--', linewidth=2,
                      label=f'elbow at k = 4, score = {k4_score:.3f}')
        ax_el.legend(loc='upper right')
        ax_el.set_title('Distortion Score Elbow for KMeans Clustering')

        # Display the single Elbow plot
        st.pyplot(fig_el)

        st.divider()
        st.subheader("Visualisasi Sebaran Spasial Klaster (K=4)")

        # Reduksi PCA untuk Visualisasi Efektif
        pca_2d = PCA(n_components=2, random_state=42)
        pca_data_2d = pca_2d.fit_transform(k_inputs_eval)
        df_clean['pca_x'] = pca_data_2d[:, 0]
        df_clean['pca_y'] = pca_data_2d[:, 1]

        pca_3d = PCA(n_components=3, random_state=42)
        pca_data_3d = pca_3d.fit_transform(k_inputs_eval)
        df_clean['pca_z'] = pca_3d.fit_transform(k_inputs_eval)[:, 2]

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write(">> Plot Distribusi 2D PCA")
            fig_pca_2d, ax_pca_2d = plt.subplots(figsize=(10, 7))
            sns.scatterplot(x='pca_x', y='pca_y', hue='cluster_viz', data=df_clean, palette='Set1', alpha=0.8, ax=ax_pca_2d)
            ax_pca_2d.set_title('Visualisasi Klaster Menggunakan PCA (2D) - K=4')
            ax_pca_2d.grid(True)
            st.pyplot(fig_pca_2d)

        with col_p2:
            st.write(">> Plot Distribusi 3D PCA")
            fig_pca_3d = plt.figure(figsize=(10, 8))
            ax_pca_3d = fig_pca_3d.add_subplot(111, projection='3d')
            scatter = ax_pca_3d.scatter(df_clean['pca_x'], df_clean['pca_y'], df_clean['pca_z'],
                                        c=df_clean['cluster_viz'], cmap='Set1', s=40, alpha=0.7)
            ax_pca_3d.set_title('Visualisasi Klaster Menggunakan PCA (3D) - K=4')
            ax_pca_3d.set_xlabel('PC 1')
            ax_pca_3d.set_ylabel('PC 2')
            ax_pca_3d.set_zlabel('PC 3')
            plt.colorbar(scatter, ax=ax_pca_3d, label='Cluster ID')
            st.pyplot(fig_pca_3d)

    with tab_dist:
        st.subheader("Karakteristik Atribut Boxplot per Klaster Segmen")
        fitur_pilih = st.selectbox("Pilih Atribut untuk Ditinjau:", ['Wine_Spend', '%Wine_Share', 'Purchase_Vol', 'Loyalitas_Bulan', 'Income', 'Age'])

        fig_box_seg, ax_box_seg = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='cluster_viz', y=fitur_pilih, data=df_clean, palette='Set1', ax=ax_box_seg)
        ax_box_seg.set_xticklabels([f"C{i} - {segments[i]}" for i in range(4)], rotation=15)
        ax_box_seg.set_title(f"Komparasi Boxplot Fitur: {fitur_pilih} Lintas Segmen")
        plt.tight_layout()
        st.pyplot(fig_box_seg)

        st.divider()

        # Before Outlier Capping Boxplots (using df_processed for original data)
        st.subheader("Deteksi Batas Outliers (Sebelum Capping)")
        fig_outliers_before, axes_before = plt.subplots(2, 2, figsize=(14, 8))
        sns.boxplot(y=df_processed['Income'], ax=axes_before[0, 0], color='skyblue')
        axes_before[0, 0].set_title('Income Outliers Status')
        sns.boxplot(y=df_processed['Age'], ax=axes_before[0, 1], color='lightgreen')
        axes_before[0, 1].set_title('Age Outliers Status')
        sns.boxplot(y=df_processed['Wine_Spend'], ax=axes_before[1, 0], color='coral')
        axes_before[1, 0].set_title('Wine Spend Outliers Status')
        sns.boxplot(y=df_processed['AmtTotal'], ax=axes_before[1, 1], color='gold')
        axes_before[1, 1].set_title('Total Amount Outliers Status')
        plt.tight_layout()
        st.pyplot(fig_outliers_before)

        # After Outlier Capping Boxplots (using df_clean)
        st.subheader("Deteksi Batas Outliers (Sesudah Capping)")
        fig_outliers_after, axes_after = plt.subplots(2, 2, figsize=(14, 8))
        sns.boxplot(y=df_clean['Income'], ax=axes_after[0, 0], color='skyblue')
        axes_after[0, 0].set_title('Income Outliers Status (Capped)')
        sns.boxplot(y=df_clean['Age'], ax=axes_after[0, 1], color='lightgreen')
        axes_after[0, 1].set_title('Age Outliers Status (Capped)')
        sns.boxplot(y=df_clean['Wine_Spend'], ax=axes_after[1, 0], color='coral')
        axes_after[1, 0].set_title('Wine Spend Outliers Status (Capped)')
        sns.boxplot(y=df_clean['AmtTotal'], ax=axes_after[1, 1], color='gold')
        axes_after[1, 1].set_title('Total Amount Outliers Status (Capped)')
        plt.tight_layout()
        st.pyplot(fig_outliers_after)

        st.subheader("Distribusi Geometris Fitur Numerik Utama (Histogram)")
        fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 8))
        df_clean[['Income','Age','Wine_Spend','Purchase_Vol','Loyalitas_Bulan', 'Num_Children']].hist(bins=20, ax=axes_hist.flatten(), color='purple', edgecolor='black', alpha=0.7)
        fig_hist.suptitle('Distributions of Key Numerical Features', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig_hist)

        st.divider()

        # Confusion Matrix plots (using cm_dict)
        st.subheader("Evaluasi Confusion Matrix (Random Forest)")
        st.write("Perbandingan Confusion Matrix sebelum dan sesudah penerapan SMOTE untuk mengatasi imbalance data.")

        col_cm1, col_cm2 = st.columns(2)

        with col_cm1:
            st.markdown("##### Confusion Matrix (Before SMOTE)")
            fig_cm_before, ax_cm_before = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm_dict['before'], display_labels=list(segments.values())).plot(ax=ax_cm_before, cmap='Blues', xticks_rotation=45)
            ax_cm_before.set_title('Before SMOTE')
            st.pyplot(fig_cm_before)

        with col_cm2:
            st.markdown("##### Confusion Matrix (After SMOTE)")
            fig_cm_after, ax_cm_after = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm_dict['after'], display_labels=list(segments.values())).plot(ax=ax_cm_after, cmap='Greens', xticks_rotation=45)
            ax_cm_after.set_title('After SMOTE')
            st.pyplot(fig_cm_after)

# --- FOOTER IDENTITAS ---
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: gray;'>Dibuat Oleh: Kelompok 7</h4>", unsafe_allow_html=True)
st.caption("Dashboard Analytics terintegrasi - K-Means, PCA, Random Forest, Naive Bayes & Linear Regression Engine.")
