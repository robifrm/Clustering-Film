import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🎬 Film Clustering App")
st.write("Upload dataset film, atur parameter, lihat cluster, dan dapatkan list film serupa berdasarkan cluster.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Kolom:", df.columns.tolist())

    # Slider
    k = st.slider("Jumlah Cluster (k)", 2, 10, 3)
    min_rating, max_rating = st.slider("Rentang Rating", 0.0, 10.0, (0.0, 10.0))
    min_duration, max_duration = st.slider("Rentang Durasi (menit)", 0, 300, (0, 300))

    # Filter data
    df_filtered = df[
        (df['Rating'] >= min_rating) &
        (df['Rating'] <= max_rating) &
        (df['Duration'] >= min_duration) &
        (df['Duration'] <= max_duration)
    ]

    if df_filtered.empty:
        st.warning("Data kosong setelah filter! Coba atur slider.")
    else:
        # Encode Genre
        le = LabelEncoder()
        df_filtered['Genre_encoded'] = le.fit_transform(df_filtered['Genre'])

        # Clustering
        features = df_filtered[['Genre_encoded', 'Rating', 'Duration']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(features)
        df_filtered['Cluster'] = clusters

        # Plot
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_filtered,
            x='Rating', y='Duration',
            hue='Cluster', palette='tab10',
            ax=ax
        )
        st.pyplot(fig)

        # Tabel film mirip: Tampilkan semua film di cluster yang sama
        st.subheader("📃 Daftar Film dan Clusternya")
        st.dataframe(df_filtered[['Title', 'Genre', 'Rating', 'Duration', 'Cluster']])

        # Opsi download
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Hasil Clustering",
            data=csv,
            file_name='clustering_result.csv',
            mime='text/csv'
        )
