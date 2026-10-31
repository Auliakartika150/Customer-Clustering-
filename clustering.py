import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="Customer Intelligence | Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet warna dashboard
NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#18A6A6"
PURPLE = "#8064A2"
ORANGE = "#E69A36"
GREEN = "#26966F"
LIGHT_BG = "#F4F7FB"
TEXT = "#243447"
MUTED = "#66788A"

CLUSTER_COLORS = [
    "#2878B5", "#18A6A6", "#E69A36", "#8064A2",
    "#26966F", "#D95D5D", "#5B6C8F", "#9A6B45",
]

FEATURES = ["Age", "Income", "Education", "Occupation", "Settlement size"]
FEATURE_LABELS = {
    "Age": "Usia",
    "Income": "Pendapatan",
    "Education": "Pendidikan",
    "Occupation": "Pekerjaan",
    "Settlement size": "Ukuran wilayah tempat tinggal",
}


# =========================================================
# 2. CSS: TAMPILAN DASHBOARD
# =========================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {LIGHT_BG};
    }}

    [data-testid="stHeader"] {{
        background: rgba(244, 247, 251, 0.92);
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: #DCE7F1 !important;
    }}

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {{
        color: {TEXT} !important;
    }}

    /* Header utama */
    .hero {{
        background: linear-gradient(115deg, {NAVY} 0%, #245C83 58%, {BLUE} 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.13);
    }}

    .hero h1 {{
        color: #FFFFFF;
        font-size: 29px;
        font-weight: 800;
        margin: 0 0 8px 0;
    }}

    .hero p {{
        color: #E4EFF8;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }}

    .section-title {{
        color: {NAVY};
        font-size: 20px;
        font-weight: 750;
        margin: 8px 0 4px 0;
    }}

    .section-subtitle {{
        color: {MUTED};
        font-size: 13px;
        margin-bottom: 14px;
    }}

    /* Kartu metrik */
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #E5EBF2;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(23, 50, 77, 0.045);
    }}

    div[data-testid="stMetricLabel"] p {{
        color: {MUTED} !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-weight: 800 !important;
    }}

    /* Tab */
    div[data-testid="stTabs"] button {{
        font-weight: 650;
        color: #52677B;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {BLUE} !important;
    }}

    /* Kartu informasi */
    .info-card {{
        background: #FFFFFF;
        border: 1px solid #E5EBF2;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 8px;
    }}

    .info-card h4 {{
        color: {NAVY};
        margin: 0 0 8px 0;
        font-size: 15px;
    }}

    .info-card p {{
        color: {MUTED};
        font-size: 13px;
        line-height: 1.65;
        margin: 0;
    }}

    /* Tombol */
    .stButton > button,
    .stDownloadButton > button {{
        background: {BLUE} !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 9px;
        font-weight: 700 !important;
    }}

    .stButton > button *,
    .stDownloadButton > button * {{
        color: #FFFFFF !important;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: {NAVY} !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MEMUAT DATASET LOKAL
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "segmentation data.csv"


@st.cache_data
def load_data():
    """Membaca dataset pelanggan dan memeriksa kolom yang dibutuhkan."""
    data = pd.read_csv(DATA_PATH)

    # Merapikan nama kolom untuk menghindari masalah spasi
    data.columns = data.columns.astype(str).str.strip()

    missing_columns = [column for column in FEATURES if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Kolom berikut tidak ditemukan dalam dataset: "
            + ", ".join(missing_columns)
        )

    # Mengubah fitur menjadi numerik
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Menghapus baris yang memiliki nilai kosong pada fitur yang digunakan
    original_rows = len(data)
    data = data.dropna(subset=FEATURES).copy()
    removed_rows = original_rows - len(data)

    if data.empty:
        raise ValueError("Dataset tidak memiliki baris valid untuk dianalisis.")

    return data, removed_rows


try:
    df, removed_rows = load_data()
except FileNotFoundError:
    st.error(
        "File 'segmentation data.csv' tidak ditemukan. "
        "Letakkan file CSV dalam folder yang sama dengan app_clustering.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Dataset gagal dibaca: {error}")
    st.stop()


# =========================================================
# 4. SIDEBAR: PENGATURAN CLUSTERING
# =========================================================
max_clusters = min(8, len(df) - 1)

if max_clusters < 2:
    st.error("Jumlah data valid terlalu sedikit untuk melakukan clustering.")
    st.stop()

with st.sidebar:
    st.markdown("## 📊 Customer Intelligence")
    st.markdown("Analisis kelompok pelanggan")
    st.divider()

    st.markdown("### Pengaturan model")
    n_clusters = st.slider(
        "Jumlah kelompok pelanggan",
        min_value=2,
        max_value=max_clusters,
        value=min(3, max_clusters),
        help="Pilih jumlah kelompok pelanggan yang ingin dibentuk oleh K-Means.",
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Gunakan angka yang sama agar hasil pengelompokan dapat diulang.",
    )

    st.divider()
    st.markdown("**Informasi dataset**")
    st.caption(f"File: {DATA_PATH.name}")
    st.caption(f"Data valid: {len(df):,} baris")
    st.caption(f"Fitur analisis: {len(FEATURES)} kolom")

    if removed_rows > 0:
        st.caption(f"Baris dengan data kosong/tidak valid diabaikan: {removed_rows:,}")


# =========================================================
# 5. PREPROCESSING DAN MODEL K-MEANS
# =========================================================
features = df[FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

model = KMeans(
    n_clusters=int(n_clusters),
    random_state=int(random_state),
    n_init=10,
)
cluster_labels = model.fit_predict(X_scaled)

# Gunakan salinan data agar data asli tidak berubah
result_df = df.copy()
result_df["Cluster"] = cluster_labels
result_df["Nama Cluster"] = result_df["Cluster"].map(
    lambda value: f"Segmen {value + 1}"
)

cluster_counts = (
    result_df["Cluster"]
    .value_counts()
    .sort_index()
)

# Evaluasi kualitas cluster
if 1 < len(np.unique(cluster_labels)) < len(result_df):
    silhouette = silhouette_score(X_scaled, cluster_labels)
    db_index = davies_bouldin_score(X_scaled, cluster_labels)
else:
    silhouette = np.nan
    db_index = np.nan

# Data profil cluster
profile = result_df.groupby("Cluster")[FEATURES].mean().round(2)


# =========================================================
# 6. HEADER DAN RINGKASAN DASHBOARD
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>Customer Intelligence Platform</h1>
        <p>
            Pahami karakteristik pelanggan melalui pengelompokan K-Means.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Ringkasan Segmentasi</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Gambaran umum hasil pengelompokan pelanggan berdasarkan pengaturan model saat ini.</div>',
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Total pelanggan", f"{len(result_df):,}")
metric_2.metric("Jumlah segmen", f"{n_clusters}")
metric_3.metric(
    "Silhouette Score",
    f"{silhouette:.3f}" if pd.notna(silhouette) else "Tidak tersedia",
    help="Semakin mendekati 1, umumnya semakin baik pemisahan antarsegmen.",
)
metric_4.metric(
    "Davies–Bouldin Index",
    f"{db_index:.3f}" if pd.notna(db_index) else "Tidak tersedia",
    help="Semakin kecil nilainya, umumnya semakin baik pemisahan dan kekompakan cluster.",
)

tabs = st.tabs(
    [
        "Ikhtisar",
        "Visualisasi Segmen",
        "Profil Pelanggan",
        "Data Pelanggan",
    ]
)


# =========================================================
# 7. TAB IKHTISAR
# =========================================================
with tabs[0]:
    st.markdown(
        '<div class="section-title">Gambaran Kelompok Pelanggan</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Lihat jumlah dan proporsi pelanggan pada setiap segmen.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.2], gap="large")

    count_df = cluster_counts.rename_axis("Cluster").reset_index(
        name="Jumlah pelanggan"
    )
    count_df["Segmen"] = count_df["Cluster"].map(
        lambda value: f"Segmen {value + 1}"
    )
    color_map = {
        f"Segmen {index + 1}": CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index in range(n_clusters)
    }

    with left:
        fig = px.pie(
            count_df,
            names="Segmen",
            values="Jumlah pelanggan",
            hole=0.60,
            color="Segmen",
            color_discrete_map=color_map,
            title="Proporsi Pelanggan per Segmen",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hole=0.60,
        )
        fig.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=60, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
            legend_title_text="Kelompok",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            count_df,
            x="Segmen",
            y="Jumlah pelanggan",
            color="Segmen",
            color_discrete_map=color_map,
            text="Jumlah pelanggan",
            title="Jumlah Pelanggan per Segmen",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            height=390,
            showlegend=False,
            margin=dict(l=15, r=15, t=60, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
            xaxis_title="Kelompok pelanggan",
            yaxis_title="Jumlah pelanggan",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Ringkasan jumlah pelanggan")
    display_counts = count_df[["Segmen", "Jumlah pelanggan"]].copy()
    display_counts["Persentase"] = (
        display_counts["Jumlah pelanggan"] / len(result_df) * 100
    ).map(lambda value: f"{value:.1f}%")
    st.dataframe(display_counts, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-card">
            <h4>Apa arti hasil segmentasi?</h4>
            <p>
                K-Means mengelompokkan pelanggan berdasarkan kemiripan nilai pada
                fitur yang dipilih. Setiap segmen berisi pelanggan dengan pola
                karakteristik yang relatif mirip. Nama segmen hanya menunjukkan
                kelompok hasil algoritma.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. TAB VISUALISASI SEGMEN
# =========================================================
with tabs[1]:
    st.markdown(
        '<div class="section-title">Peta Segmentasi Pelanggan</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Bandingkan posisi pelanggan berdasarkan dua karakteristik yang dipilih.</div>',
        unsafe_allow_html=True,
    )

    x_axis = st.selectbox(
        "Karakteristik sumbu horizontal",
        FEATURES,
        index=0,
        format_func=lambda value: FEATURE_LABELS.get(value, value),
    )
    y_axis = st.selectbox(
        "Karakteristik sumbu vertikal",
        FEATURES,
        index=1,
        format_func=lambda value: FEATURE_LABELS.get(value, value),
    )

    if x_axis == y_axis:
        st.warning("Pilih dua karakteristik yang berbeda agar grafik lebih informatif.")
    else:
        scatter_df = result_df.copy()
        scatter_df["Segmen"] = scatter_df["Cluster"].map(
            lambda value: f"Segmen {value + 1}"
        )

        fig = px.scatter(
            scatter_df,
            x=x_axis,
            y=y_axis,
            color="Segmen",
            color_discrete_map=color_map,
            opacity=0.72,
            hover_data=FEATURES,
            labels={
                x_axis: FEATURE_LABELS.get(x_axis, x_axis),
                y_axis: FEATURE_LABELS.get(y_axis, y_axis),
                "Segmen": "Kelompok pelanggan",
            },
            title=(
                f"Persebaran pelanggan: "
                f"{FEATURE_LABELS.get(x_axis, x_axis)} dan "
                f"{FEATURE_LABELS.get(y_axis, y_axis)}"
            ),
        )
        fig.update_layout(
            height=560,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
            margin=dict(l=15, r=15, t=65, b=15),
            legend_title_text="Kelompok pelanggan",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Setiap titik mewakili satu pelanggan. Warna menunjukkan segmen hasil K-Means. "
        "Grafik menampilkan dua fitur saja, sedangkan pembentukan cluster menggunakan seluruh fitur."
    )


# =========================================================
# 9. TAB PROFIL PELANGGAN
# =========================================================
with tabs[2]:
    st.markdown(
        '<div class="section-title">Profil Setiap Segmen</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Bandingkan rata-rata karakteristik pelanggan pada setiap kelompok.</div>',
        unsafe_allow_html=True,
    )

    profile_display = profile.copy()
    profile_display.index = [f"Segmen {index + 1}" for index in profile_display.index]
    profile_display = profile_display.rename(columns=FEATURE_LABELS)
    st.dataframe(profile_display, use_container_width=True)

    st.markdown("#### Perbandingan karakteristik")
    st.caption(
        "Grafik menggunakan nilai rata-rata yang telah distandardisasi. "
        "Nilai di atas 0 berarti relatif di atas rata-rata keseluruhan dataset; "
        "nilai di bawah 0 berarti relatif di bawah rata-rata."
    )

    # Standardisasi profil agar fitur dengan skala berbeda dapat dibandingkan
    profile_scaled_values = scaler.transform(profile[FEATURES])
    profile_scaled = pd.DataFrame(
        profile_scaled_values,
        columns=FEATURES,
        index=profile.index,
    )
    profile_long = (
        profile_scaled.reset_index(names="Cluster")
        .melt(
            id_vars="Cluster",
            var_name="Karakteristik",
            value_name="Nilai relatif",
        )
    )
    profile_long["Segmen"] = profile_long["Cluster"].map(
        lambda value: f"Segmen {value + 1}"
    )
    profile_long["Karakteristik"] = profile_long["Karakteristik"].map(
        FEATURE_LABELS
    )

    fig = px.bar(
        profile_long,
        x="Karakteristik",
        y="Nilai relatif",
        color="Segmen",
        barmode="group",
        color_discrete_map=color_map,
        title="Perbandingan Profil Relatif Antarsegmen",
        labels={"Nilai relatif": "Nilai setelah standardisasi"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#8A99A8")
    fig.update_layout(
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color=TEXT,
        margin=dict(l=15, r=15, t=60, b=15),
        legend_title_text="Kelompok pelanggan",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Gunakan tabel untuk membaca nilai rata-rata asli. Gunakan grafik untuk "
        "membandingkan pola relatif antarfitur yang memiliki skala berbeda."
    )


# =========================================================
# 10. TAB DATA PELANGGAN
# =========================================================
with tabs[3]:
    st.markdown(
        '<div class="section-title">Data Pelanggan</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Telusuri data berdasarkan segmen yang ingin diperiksa.</div>',
        unsafe_allow_html=True,
    )

    available_segments = sorted(result_df["Cluster"].unique())
    selected_segments = st.multiselect(
        "Pilih segmen pelanggan",
        options=available_segments,
        default=available_segments,
        format_func=lambda value: f"Segmen {value + 1}",
    )

    filtered_df = result_df[
        result_df["Cluster"].isin(selected_segments)
    ].copy()

    display_df = filtered_df.rename(
        columns={
            **FEATURE_LABELS,
            "Cluster": "Kode segmen",
            "Nama Cluster": "Nama segmen",
        }
    )

    st.metric("Jumlah pelanggan yang ditampilkan", f"{len(filtered_df):,}")
    st.dataframe(
        display_df,
        use_container_width=True,
        height=430,
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh data hasil segmentasi (CSV)",
        data=csv_data,
        file_name="hasil_segmentasi_pelanggan.csv",
        mime="text/csv",
        use_container_width=False,
    )


# =========================================================
# 11. METODOLOGI DAN BATASAN
# =========================================================
with st.expander("Metode analisis dan hal yang perlu diperhatikan"):
    st.markdown(
        """
        **Tahapan analisis**
        1. Membaca `segmentation data.csv` dari folder aplikasi.
        2. Memeriksa ketersediaan lima fitur yang digunakan.
        3. Mengubah fitur menjadi format numerik dan mengabaikan baris yang tidak valid.
        4. Menyamakan skala fitur menggunakan `StandardScaler`.
        5. Mengelompokkan pelanggan menggunakan algoritma K-Means.
        6. Mengukur kualitas kelompok menggunakan Silhouette Score dan Davies–Bouldin Index.
        7. Menampilkan ukuran, persebaran, dan profil setiap segmen.

        **Cara membaca metrik**
        - Silhouette Score: semakin tinggi nilainya, umumnya semakin baik pemisahan antarcluster.
        - Davies–Bouldin Index: semakin rendah nilainya, umumnya semakin baik kekompakan dan pemisahan cluster.
        """
    )

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7B8A99;
        font-size:12px;
        padding:18px 0 4px 0;
        border-top:1px solid #E2E8F0;">
        Customer Intelligence Platform · Customer Segmentation Project
    </div>
    """,
    unsafe_allow_html=True,
)
