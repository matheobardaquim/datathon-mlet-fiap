# dashboard.py (VERSÃO FINAL COM CORREÇÃO DE TIMEZONE)

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Dashboard de Monitoramento de Drift do Modelo")

@st.cache_data
def load_data():
    log_entries = []
    try:
        with open("predictions.log", "r") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('{'):
                    try:
                        log_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        st.error("Arquivo 'predictions.log' não encontrado. Chame a API para gerar logs.")
        return None

    if not log_entries:
        st.warning("Nenhum log de predição válido encontrado.")
        return None

    prod_df = pd.json_normalize(log_entries)
    if 'timestamp' in prod_df.columns:
        prod_df['timestamp'] = pd.to_datetime(prod_df['timestamp'])
    else:
        st.error("O arquivo de log não contém a coluna 'timestamp'. Verifique a configuração do logger na API.")
        return None
        
    return prod_df

prod_data = load_data()

if prod_data is not None:
    st.header("Análise de Tendência e Drift de Previsão")

    # --- Filtro de Data na Barra Lateral ---
    st.sidebar.header("Filtros")
    min_date = prod_data['timestamp'].min().date()
    max_date = prod_data['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Selecione o Período de Análise",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        # --- CORREÇÃO APLICADA AQUI ---
        # Converte as datas do filtro e as TORNA CONSCIENTES do fuso horário UTC
        start_date = pd.to_datetime(date_range[0]).tz_localize('UTC')
        end_date = pd.to_datetime(date_range[1]).replace(hour=23, minute=59).tz_localize('UTC')

        # Agora a comparação entre datas com o mesmo fuso horário funciona
        filtered_data = prod_data[
            (prod_data['timestamp'] >= start_date) & (prod_data['timestamp'] <= end_date)
        ]

        if filtered_data.empty:
            st.warning("Nenhum dado encontrado para o período selecionado.")
        else:
            # --- Gráfico de Tendência ---
            st.subheader("Tendência da Probabilidade Média de 'Match' por Dia")
            daily_avg_prob = filtered_data.set_index('timestamp')['probability_match'].resample('D').mean().dropna()
            st.line_chart(daily_avg_prob)
            st.markdown("""
            **Como interpretar:** Se esta linha mostra uma tendência clara de subida ou descida, 
            isso pode ser um forte indicador de drift.
            """)

            # --- Comparação de Distribuição ---
            st.subheader("Comparação da Distribuição de Probabilidades no Período")
            half_point = len(filtered_data) // 2
            first_half = filtered_data.iloc[:half_point]
            second_half = filtered_data.iloc[half_point:]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Início do Período ({len(first_half)} predições)**")
                if not first_half.empty:
                    hist_values_1 = np.histogram(first_half['probability_match'], bins=20, range=(0,1))[0]
                    st.bar_chart(hist_values_1)

            with col2:
                st.write(f"**Fim do Período ({len(second_half)} predições)**")
                if not second_half.empty:
                    hist_values_2 = np.histogram(second_half['probability_match'], bins=20, range=(0,1))[0]
                    st.bar_chart(hist_values_2)
            
            st.markdown("""
            **Como interpretar:** Se a forma da distribuição for visivelmente diferente 
            entre o início e o fim do período, isso indica que o padrão das previsões do modelo está mudando.
            """)