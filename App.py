import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Upload Dataset
# -------------------------
st.title("📊 Film Clustering App with Elbow & Silhouette")

uploaded_file = st.file_uploader("Upload CSV Data Film", type="csv")

if uploaded_file is not None:
    # === Load & Tampilkan ===
    df = pd.read_csv(uploaded_file)
    st.write("### Data Asli", df.head())

    # === Filter ===
    k = st.sidebar.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)
    min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=10.0, value=0.0)
    max_rating = st.sidebar.slider("Maksimum Rating", min_value=0.0, max_value=10.0, value=10.0)
    min_duration = st.sidebar.slider("Minimum Durasi (menit)", min_value=0, max_value=300, value=0)
    max_duration = st.sidebar.slider("Maksimum Durasi (menit)", min_value=0, max_value=300, value=300)

    df_filtered = df[
        (df['Rating'] >= min_rating) & (df['Rating'] <= max_rating) &
        (df['Duration'] >= min_duration) & (df['Duration'] <= max_duration)
    ]
    st.write(f"### Data setelah filter ({len(df_filtered)} film)", df_filtered.head())

    # === Preprocessing ===
    le = LabelEncoder()
    df_filtered['Genre_encoded'] = le.fit_transform(df_filtered['Genre'])
    scaler = StandardScaler()
    features = df_filtered[['Genre_encoded', 'Rating', 'Duration']]
    scaled_features = scaler.fit_transform(features)

    # === Elbow Method ===
    st.write("### 📈 Elbow Method")
    inertia = []
    k_range = range(1, 11)
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(scaled_features)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker='o')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    # === Clustering ===
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    df_filtered['Cluster'] = clusters

    # === Silhouette Score ===
    sil_score = silhouette_score(scaled_features, clusters)
    st.write(f"### 🏷️ Silhouette Score untuk k = {k} : **{sil_score:.4f}**")

    # === Visualisasi Cluster ===
    st.write("### 🎨 Visualisasi Cluster (Rating vs Duration)")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Rating',
        y='Duration',
        hue='Cluster',
        data=df_filtered,
        palette='viridis'
    )
    st.pyplot(plt.gcf())

    # === Statistik Cluster ===
    st.write("### 📊 Statistik Tiap Cluster")
    cluster_summary = df_filtered.groupby('Cluster').agg({
        'Genre': lambda x: x.mode()[0],
        'Rating': 'mean',
        'Duration': 'mean',
        'Cluster': 'count'
    }).rename(columns={
        'Genre': 'Genre Dominan',
        'Rating': 'Rating Rata-rata',
        'Duration': 'Durasi Rata-rata',
        'Cluster': 'Jumlah Film'
    })
    st.write(cluster_summary)

    # === Film Berdasarkan Cluster ===
    st.write("### 📃 Daftar Film Berdasarkan Cluster")
    selected_cluster = st.selectbox(
        "Pilih Nomor Cluster untuk lihat detail filmnya",
        sorted(df_filtered['Cluster'].unique())
    )
    recommended = df_filtered[df_filtered['Cluster'] == selected_cluster]
    st.write(f"**Film di Cluster {selected_cluster}**:")
    st.dataframe(recommended[['Movie_Title', 'Genre', 'Rating', 'Duration']])

    # === Download Button ===
    st.write("### ⬇️ Download Data Hasil Clustering")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='film_clustered.csv',
        mime='text/csv'
    )

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
