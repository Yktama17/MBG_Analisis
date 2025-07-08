# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Dashboard Analisis Sentimen Program Makan Bergizi Gratis")

if 'raw_df' in st.session_state:
    st.markdown("---")
    df = st.session_state['raw_df']
    st.subheader("Ringkasan Data yang Telah Diupload")

    total_data = len(df)
    sentiment_counts = df['sentimen'].str.lower().value_counts() ## PERBAIKAN
    positif_count = sentiment_counts.get('positif', 0)
    negatif_count = sentiment_counts.get('negatif', 0)
    netral_count = sentiment_counts.get('netral', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📊 Total Data", value=total_data)
    with col2:
        st.metric(label="😊 Total Sentimen Positif", value=positif_count)
    with col3:
        st.metric(label="😠 Total Sentimen Negatif", value=negatif_count)
    with col4:
        st.metric(label="😐 Total Sentimen Netral", value=netral_count)
        
    st.subheader("Distribusi Sentimen")
    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index, 
        values=sentiment_counts.values,
        hole=.3
    )])
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.markdown("---")
    st.info("ℹ️ Selamat datang! Dashboard akan terisi setelah Anda mengupload dan memproses data di halaman 'Preprocessing'.")
    st.subheader("Ringkasan Data")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📊 Total Data", value="-")
    with col2:
        st.metric(label="😊 Total Sentimen Positif", value="-")
    with col3:
        st.metric(label="😠 Total Sentimen Negatif", value="-")
    with col4:
        st.metric(label="😐 Total Sentimen Netral", value="-")
        
    st.subheader("Distribusi Sentimen")
    st.markdown("Chart akan muncul di sini setelah data diproses.")