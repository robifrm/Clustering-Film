# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Judul & Deskripsi
# -------------------------
st.title("Film Clustering App")
st.write("""
Aplikasi ini mengelompokkan film berdasarkan **Genre**, **Rating**, dan **Durasi**.
Gunakan slider dan filter untuk menyesuaikan jumlah cluster dan data.
""")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("Upload CSV Data Film", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Asli", df.head())

    # -------------------------
    # Filter & Parameter
    # -------------------------
    k = st.sidebar.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)
    min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=10.0, value=0.0)
    max_rating = st.sidebar.slider("Maksimum Rating", min_value=0.0, max_value=10.0, value=10.0)
    min_duration = st.sidebar.slider("Minimum Durasi (menit)", min_value=0, max_value=300, value=0)
    max_duration = st.sidebar.slider("Maksimum Durasi (menit)", min_value=0, max_value=300, value=300)

    # Apply filter
    df_filtered = df[(df['Rating'] >= min_rating) & (df['Rating'] <= max_rating) &
                     (df['Duration'] >= min_duration) & (df['Duration'] <= max_duration)]

    st.write(f"### Data setelah filter ({len(df_filtered)} film)", df_filtered.head())

    # -------------------------
    # Preprocessing
    # -------------------------
    le = LabelEncoder()
    df_filtered['Genre_encoded'] = le.fit_transform(df_filtered['Genre'])

    features = df_filtered[['Genre_encoded', 'Rating', 'Duration']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # -------------------------
    # Clustering
    # -------------------------
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    df_filtered['Cluster'] = clusters

    st.write("### Data dengan Cluster", df_filtered)

    # -------------------------
    # Plot Cluster
    # -------------------------
    st.write("### Visualisasi Cluster (Rating vs Duration)")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Rating',
        y='Duration',
        hue='Cluster',
        data=df_filtered,
        palette='viridis'
    )
    st.pyplot(plt.gcf())

    # -------------------------
    # Statistik Cluster
    # -------------------------
    st.write("### Statistik Tiap Cluster")
    cluster_summary = df_filtered.groupby('Cluster').agg({
        'Genre': lambda x: x.mode()[0],
        'Rating': 'mean',
        'Duration': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Genre': 'Genre Dominan', 'Rating': 'Rating Rata-rata',
                       'Duration': 'Durasi Rata-rata', 'Cluster': 'Jumlah Film'})
    st.write(cluster_summary)

    # -------------------------
    # Rekomendasi Film Mirip
    # -------------------------
    st.write("### Rekomendasi Film Mirip Berdasarkan Cluster")
    selected_film = st.selectbox("Pilih Film", df_filtered['Movie_Title'])
    selected_cluster = df_filtered[df_filtered['Movie_Title'] == selected_film]['Cluster'].values[0]
    recommended = df_filtered[(df_filtered['Cluster'] == selected_cluster) & (df_filtered['Movie_Title'] != selected_film)]
    st.write(f"Film mirip dengan **{selected_film}**:", recommended[['Movie_Title', 'Genre', 'Rating', 'Duration']])

    # -------------------------
    # Download Hasil
    # -------------------------
    st.write("### Download Data Hasil Clustering")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='film.csv',
        mime='text/csv'
    )

else:
    st.info("Silakan upload file CSV terlebih dahulu.")

