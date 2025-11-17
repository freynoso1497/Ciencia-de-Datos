import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import math

# ======================================================================
# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE ACTIVOS ---
# ======================================================================
# Carpeta donde está este archivo App.py
BASE_DIR = Path(__file__).resolve().parent

# Paths a los archivos dentro de esa carpeta
MODEL_PATH = BASE_DIR / "modelo_churn_final.joblib"
FEATURES_PATH = BASE_DIR / "features_list.joblib"
DATA_PATH = BASE_DIR / "cust_df_final_for_streamlit.csv"
ADJUSTED_THRESHOLD = 0.30
CAMPAIGN_COST = 100000

# --- Funciones Auxiliares para Carga de Activos ---

@st.cache_data
def load_data(path):
    """Carga el DataFrame de clientes y rellena NaN."""
    try:
        df = pd.read_csv(path)
        if 'cliente_id' in df.columns:
            df = df.fillna(df.median(numeric_only=True))
            df['cliente_id'] = df['cliente_id'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos en {path}. ¡Deteniendo la aplicación!")
        st.stop()
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path, features_path):
    """Carga el modelo y la lista de features."""
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo del modelo o de features. ¡Deteniendo la aplicación!")
        st.stop()
        return None, None

# Cargar Modelo y Datos de Predicción
model, feature_names = load_model(MODEL_PATH, FEATURES_PATH)
cust_df_full = load_data(DATA_PATH)

# ======================================================================
# --- 2. FUNCIONES CENTRALES DEL MODELO ---
# ======================================================================

@st.cache_data
def predict_all_clients(df, _model, feature_names, threshold): 
    """Realiza la predicción de churn para todos los clientes."""
    df_input = df[feature_names]
    probas = _model.predict_proba(df_input)[:, 1]
    
    df_result = df[['cliente_id']].copy()
    pretty_cols = {'Localidad_Moda': 'Localidad', 'Nombre1_Moda': 'Nombre'}
    for col in ['Localidad_Moda', 'Nombre1_Moda']:
        if col in df.columns:
            df_result[col] = df[col]
    df_result = df_result.rename(columns=pretty_cols)
    df_result['Probabilidad_Churn'] = probas
    df_result['Riesgo'] = np.where(probas >= threshold, 'ALTO', 'BAJO')
    
    return df_result


@st.cache_data
def prepare_map_data():
    """Prepara datos agregados por localidad con coordenadas y métricas de churn."""
    # Coordenadas conocidas
    coords = {
        "CRUZ DE PIEDRA": (-33.0311, -68.7761),
        "AGRELO": (-33.1181, -68.8869),
        "MEDIA AGUA": (-31.983333, -68.433333),
        "COQUIMBITO": (-32.966667, -68.75),
        "EUGENIO BUSTOS": (-33.778056, -69.065),
        "TUNUYAN": (-33.566667, -69.016667),
        "GUTIERREZ": (-32.9667, -68.75),
        "JUNIN": (-33.25, -68.716667),
        "TUPUNGATO": (-33.3728, -69.1475),
        "LUZURIAGA": (-32.9431, -68.7939),
        "MENDOZA": (-32.889722222222, -68.84444444444),
        "GUAYMALLEN": (-32.9053, -68.7867),
        "GRAL. ALVEAR": (-34.966666666667, -67.7),
        "LUJAN DE CUYO": (-33.016666666667, -68.866666666667),
        "SAN JOSE": (-32.8897, -68.825),
        "RIVADAVIA": (-33.1833, -68.4667),
        "RAWSON": (-31.56, -68.5589),
        "DORREGO": (-32.8969, -68.8311),
        "SANTA LUCIA": (-31.5406, -68.4989),
        "GODOY CRUZ": (-32.9330, -68.8450),
        "CAPITAL": (-32.916666666667, -68.833333333333),
        "SAN RAFAEL": (-34.6175, -68.335555555556),
        "SAN JUAN": (-31.53726, -68.52568),
        "MAIPU": (-32.966666666667, -68.75),
        "LAS HERAS": (-32.85, -68.816666666667),
    }

    # Predicciones y unión con churn real (si existe)
    df_pred = predict_all_clients(cust_df_full, model, feature_names, ADJUSTED_THRESHOLD)
    base = df_pred.merge(
        cust_df_full[['cliente_id', 'churn']] if 'churn' in cust_df_full.columns else pd.DataFrame(),
        on='cliente_id',
        how='left'
    )
    if 'Localidad' not in base.columns:
        return pd.DataFrame()  # No hay datos de localidad para mapear

    base['Localidad'] = base['Localidad'].astype(str).str.upper()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, lambda1, phi2, lambda2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dphi = phi2 - phi1
        dlambda = lambda2 - lambda1
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    agg = (
        base.groupby('Localidad', observed=True)
        .agg(
            clientes=('cliente_id', 'count'),
            abandonaron=('churn', lambda s: int(np.nansum(s)) if len(s) else 0),
            tasa_abandono=('churn', lambda s: float(np.nanmean(s)) if len(s) else np.nan),
            riesgo_alto=('Riesgo', lambda s: int((s == 'ALTO').sum()))
        )
        .reset_index()
    )

    # Mapear coordenadas y distancia a Maipú
    plot_map = agg[agg['Localidad'].isin(coords.keys())].copy()
    if plot_map.empty:
        return pd.DataFrame()

    plot_map['Latitud'] = plot_map['Localidad'].map(lambda x: coords[x][0])
    plot_map['Longitud'] = plot_map['Localidad'].map(lambda x: coords[x][1])

    if "MAIPU" not in coords:
        return pd.DataFrame()
    maipu_lat, maipu_lon = coords["MAIPU"]
    plot_map['dist_km_maipu'] = plot_map.apply(
        lambda r: haversine_km(maipu_lat, maipu_lon, r['Latitud'], r['Longitud']),
        axis=1
    )

    return plot_map


def build_folium_map(plot_map: pd.DataFrame):
    """Construye un mapa folium mostrando distancias desde Maipú y círculos por localidad."""
    try:
        import folium  # Import local para evitar romper si no está instalada la lib
    except ModuleNotFoundError:
        return None, "Falta instalar la dependencia 'folium' (pip install folium)."

    if plot_map.empty:
        return None, None

    maipu = plot_map.loc[plot_map['Localidad'] == 'MAIPU']
    maipu_lat = maipu['Latitud'].iloc[0] if not maipu.empty else -32.966666666667
    maipu_lon = maipu['Longitud'].iloc[0] if not maipu.empty else -68.75

    center_lat = float(plot_map['Latitud'].mean())
    center_lon = float(plot_map['Longitud'].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    folium.Marker(
        location=[maipu_lat, maipu_lon],
        popup="<b>MAIPU (origen)</b>",
        icon=folium.Icon(color="blue", icon="star")
    ).add_to(m)

    for _, row in plot_map.iterrows():
        color = "red" if row.get("abandonaron", 0) > 1 else "orange"
        radius = 8 + (row.get("tasa_abandono", 0) * 20)  # radio pequeño para marcar solo el centro

        popup_html = (
            f"<b>{row['Localidad']}</b><br>"
            f"Distancia desde Maipú: {row['dist_km_maipu']:.1f} km<br>"
            f"Clientes: {int(row['clientes']) if not math.isnan(row.get('clientes', float('nan'))) else '–'}<br>"
            f"Abandonaron: {int(row['abandonaron']) if not math.isnan(row.get('abandonaron', float('nan'))) else '–'}<br>"
            f"Tasa abandono: {row['tasa_abandono']:.2%}" if not math.isnan(row.get('tasa_abandono', float('nan'))) else ""
        )

        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=1,
            popup=popup_html
        ).add_to(m)

        folium.PolyLine(
            locations=[(maipu_lat, maipu_lon), (row["Latitud"], row["Longitud"])],
            color="#6c6cff", weight=2, opacity=0.6,
            tooltip=f"{row['Localidad']} · {row['dist_km_maipu']:.1f} km"
        ).add_to(m)

    return m, None


@st.cache_data
def build_retention_view(df_base, _model, feature_names, threshold, campaign_cost):
    """Combina predicciones y gasto para evaluar conveniencia de campaña."""
    required_cols = ['monetary_mean']
    if not all(col in df_base.columns for col in required_cols):
        return pd.DataFrame(), "Falta la columna 'monetary_mean' en los datos."

    pred = predict_all_clients(df_base, _model, feature_names, threshold)
    merged = pred.merge(df_base[['cliente_id', 'monetary_mean']], on='cliente_id', how='left')
    merged = merged.rename(columns={'monetary_mean': 'Gasto_mensual_prom'})
    merged['Conviene_retener'] = merged['Gasto_mensual_prom'] >= campaign_cost

    # Ordenar por probabilidad * gasto para priorizar alto valor en riesgo
    merged['Score_valor_riesgo'] = merged['Probabilidad_Churn'] * merged['Gasto_mensual_prom']
    merged = merged.sort_values('Score_valor_riesgo', ascending=False)
    return merged, None


# ======================================================================
# --- 3. FUNCIONES Y LÓGICA DE ALTAIR (Reportes) ---
# ======================================================================

# --- Funciones Auxiliares para Altair ---
ALTAIR_DATA_URL = "https://github.com/freynoso1497/Ciencia-de-Datos/releases/download/release/Dataset"
TIME_WINDOW_DAYS = 540

def minmax(s: pd.Series):
    s = s.astype(float)
    rng = s.max() - s.min()
    if rng == 0 or np.isfinite(rng) is False:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng

@st.cache_data
def create_altair_charts():
    """Carga, transforma los datos (ventas_2025-10-02.csv) y genera SOLO los tres gráficos de Altair requeridos."""
    
    csv_path = ALTAIR_DATA_URL
    if not csv_path:
        return {"error": "No se pudo resolver la URL del dataset remoto para los reportes."}, None, None

    usecols = ["Cliente","CreadoEl","Documento","Localidad","Oficina","Vendedor","ValorNeto","ClaseVenta","DenominacionClase"]
    try:
        df = pd.read_csv(csv_path, usecols=usecols, engine="pyarrow")
    except Exception:
        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as e:
            return {"error": f"No se pudo cargar el dataset remoto desde {csv_path}. Detalle: {e}"}, None, None

    # --- Lógica de Limpieza, Filtrado, Dispersión y Score (Copiada del notebook) ---
    df["CreadoEl"] = pd.to_datetime(df["CreadoEl"], dayfirst=True, errors="coerce", cache=True)
    df = df.dropna(subset=["Cliente","CreadoEl"])
    fecha_max = df["CreadoEl"].max()
    if pd.notna(fecha_max) and TIME_WINDOW_DAYS is not None:
        cutoff = fecha_max - pd.Timedelta(days=TIME_WINDOW_DAYS)
        df = df[df["CreadoEl"] >= cutoff]
    for c in ["Cliente","Localidad","Oficina","Vendedor","ClaseVenta","DenominacionClase"]:
        if c in df.columns: df[c] = df[c].astype("category")
    val = df["ValorNeto"].astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False)
    df["ValorNeto_num"] = pd.to_numeric(val, errors="coerce")
    df_sorted = df.sort_values(["Cliente","CreadoEl"], kind="mergesort")
    gaps = df_sorted.groupby("Cliente", observed=True)["CreadoEl"].diff().dt.days
    disp_cliente = (pd.DataFrame({"Cliente": df_sorted["Cliente"].values, "gap_dias": gaps.values, "Localidad": df_sorted["Localidad"].values})
        .dropna(subset=["gap_dias"]).groupby("Cliente", observed=True)
        .agg(max_gap=("gap_dias","max"), mean_gap=("gap_dias","mean"), compras=("gap_dias","size")).reset_index())
    disp_localidad = (disp_cliente.merge(df_sorted[["Cliente","Localidad"]].drop_duplicates(), on="Cliente", how="left")
        .dropna(subset=["Localidad"]).groupby("Localidad", observed=True)
        .agg(avg_max_gap=("max_gap","mean"), clientes=("Cliente","count")).reset_index())
    df_gap = df.merge(disp_cliente[["Cliente","max_gap"]], on="Cliente", how="left")
    order_col = "DenominacionClase" if "DenominacionClase" in df_gap.columns else "ClaseVenta"
    
    # La lógica del score y recency no es necesaria ya que el heatmap se elimina, pero la dejamos
    # ya que no afecta a los gráficos restantes.
    rec = df.groupby("Cliente", observed=True)["CreadoEl"].max().reset_index(name="ultima_compra")
    rec["recency_dias"] = (fecha_max - rec["ultima_compra"]).dt.days.astype(int)
    risk = disp_cliente.merge(rec[["Cliente","recency_dias"]], on="Cliente", how="left")
    risk["max_gap_norm"] = minmax(risk["max_gap"])
    risk["recency_norm"] = minmax(risk["recency_dias"])
    risk["churn_score"] = 0.6*risk["recency_norm"] + 0.4*risk["max_gap_norm"]

    # --- Generación de Gráficos Altair ---
    alt.data_transformers.disable_max_rows()
    
    gap_thr = alt.param(
        value=float(np.nanpercentile(disp_cliente["max_gap"].dropna(), 75)) if len(disp_cliente) else 90.0,
        bind=alt.binding_range(min=15, max=365, step=15, name="Umbral de alta dispersión (días)")
    )
    
    # (1) Top 20 clientes con mayor dispersión
    chart_clientes = (alt.Chart(disp_cliente).transform_window(rank="rank(max_gap)", sort=[alt.SortField("max_gap", order="descending")])
          .transform_filter("datum.rank <= 20").mark_bar().encode(x=alt.X("Cliente:N", sort="-y", title="Cliente"), y=alt.Y("max_gap:Q", title="Máx gap entre compras (días)"), tooltip=["Cliente","max_gap:Q","mean_gap:Q","compras:Q"])
          .properties(title="Top 20 clientes con mayor dispersión").interactive())
    
    # (2) Top 20 localidades por dispersión
    chart_localidades = (alt.Chart(disp_localidad).transform_window(rank="rank(avg_max_gap)", sort=[alt.SortField("avg_max_gap", order="descending")])
          .transform_filter("datum.rank <= 20").mark_bar().encode(x=alt.X("Localidad:N", sort="-y", title="Localidad"), y=alt.Y("avg_max_gap:Q", title="Promedio del máx gap (días)"), tooltip=["Localidad","avg_max_gap:Q","clientes:Q"])
          .properties(title="Top 20 localidades con mayor dispersión").interactive())
    
    # (3) Tipos de pedido entre clientes de ALTA dispersión (filtrados por slider)
    chart_tipos = (alt.Chart(df_gap).add_params(gap_thr).transform_filter(f"datum.max_gap >= {gap_thr.name}")
          .transform_calculate(tipo=f"datum['{order_col}']").transform_aggregate(n="count()", groupby=["tipo"])
          .transform_joinaggregate(N_total="sum(n)").transform_calculate(pct="datum.n / datum.N_total")
          .transform_window(rank="rank(n)", sort=[alt.SortField("n", order="descending")]).transform_filter("datum.rank <= 20")
          .mark_bar().encode(x=alt.X("tipo:N", sort="-y", title="Tipo de pedido"), y=alt.Y("pct:Q", title="Participación", axis=alt.Axis(format="%")), tooltip=[alt.Tooltip("tipo:N", title="Tipo"), alt.Tooltip("n:Q", title="# Operaciones"), alt.Tooltip("pct:Q", title="% sobre alta dispersión", format=".1%")] )
          .properties(title="Distribución de tipos de pedido en clientes de ALTA dispersión (Top 20)").interactive())
    
    # Retornamos SOLO los tres gráficos principales.
    return chart_clientes, chart_localidades, chart_tipos

# ======================================================================
# --- 4. DISEÑO DE LA APLICACIÓN PRINCIPAL (TABS) ---
# ======================================================================

st.title("Sistema de Predicción y Reportes de Churn")
st.markdown("Navega entre la predicción de clientes y los reportes de diagnóstico.")

# Crear las tarjetas (Tabs)
tab_intro, tab_modelo, tab_reportes, tab_mapa, tab_retencion = st.tabs([
    "ℹ️ Introducción",
    "🚀 Modelo (Predicción de Clientes)",
    "📊 Reportes (Análisis Altair)",
    "🗺️ Mapa",
    "💰 Retención"
])


# --- TAB 0: INTRODUCCIÓN ---
with tab_intro:
    st.header("Sobre esta aplicación")
    st.markdown("""
    Esta app permite explorar el riesgo de churn (abandono) de clientes, visualizar patrones por localidad y tomar decisiones de retención.
    
    **Qué encontrarás:**
    - **Predicción por cliente**: calcula probabilidad de churn con un modelo previamente entrenado (`modelo_churn_final.joblib`).
    - **Reportes Altair**: gráficos de dispersión/recencia y tipos de pedido para entender comportamientos agregados.
    - **Mapa**: visualización geográfica de riesgo y abandono por localidad tomando coordenadas predefinidas.
    - **Retención**: combina gasto mensual (`monetary_mean`) y probabilidad de churn para evaluar si vale la pena invertir en una campaña (costo asumido: 100.000/mes).
    
    **Datos**: se cargan de `cust_df_final_for_streamlit.csv` y se usan las columnas listadas en `features_list.joblib`. El identificador de cliente es `cliente_id`.
    """)


# --- TAB 1: MODELO (PREDICCIÓN) ---
with tab_modelo:
    st.header("Predicción de Churn por Cliente")
    st.markdown(f"Utiliza el umbral ajustado: **{ADJUSTED_THRESHOLD}**")

    # 1. Filtro de Alto Riesgo (Ver todos los clientes)
    st.subheader("Análisis Global de Riesgo")
    if st.button(' Ver Clientes con ALTO Riesgo de Churn'):
        with st.spinner('Calculando riesgos para todos los clientes...'):
            df_predictions = predict_all_clients(cust_df_full, model, feature_names, ADJUSTED_THRESHOLD)
            df_high_risk = df_predictions[df_predictions['Riesgo'] == 'ALTO'].sort_values(
                by='Probabilidad_Churn', ascending=False
            ).reset_index(drop=True)
            
            if df_high_risk.empty:
                st.success("¡Excelente! No se encontraron clientes clasificados con ALTO Riesgo de Churn.")
            else:
                st.warning(f"Se encontraron **{len(df_high_risk)}** clientes con ALTO Riesgo de Churn.")
                st.dataframe(df_high_risk, use_container_width=True)

    st.markdown("---")

    # 1.b Filtro por localidad para ver riesgo de churn
    if 'Localidad_Moda' in cust_df_full.columns:
        st.subheader("Filtrar riesgo por Localidad")
        localidades = ['Todas'] + sorted(cust_df_full['Localidad_Moda'].dropna().unique().tolist())
        selected_loc = st.selectbox('Elige una Localidad:', localidades)

        if st.button('Ver riesgo de Churn para la Localidad seleccionada'):
            with st.spinner('Calculando riesgos para la localidad...'):
                df_predictions = predict_all_clients(cust_df_full, model, feature_names, ADJUSTED_THRESHOLD)
                df_filtered = df_predictions[df_predictions['Riesgo'] == 'ALTO']
                if selected_loc != 'Todas' and 'Localidad' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['Localidad'] == selected_loc]

                if df_filtered.empty:
                    st.info("No se encontraron clientes con ALTO riesgo en la localidad seleccionada.")
                else:
                    st.warning(f"Se encontraron **{len(df_filtered)}** clientes con ALTO riesgo de Churn en la localidad seleccionada.")
                    st.dataframe(df_filtered, use_container_width=True)

    st.markdown("---")

    # 2. Búsqueda Individual
    st.subheader("Búsqueda y Predicción por Cliente ID")
    cliente_ids = cust_df_full['cliente_id'].unique()
    selected_id = st.selectbox('Elige un Cliente ID:', cliente_ids)

    if selected_id:
        client_features = cust_df_full[cust_df_full['cliente_id'] == selected_id]
        df_input = client_features[feature_names] 

        # Mostrar datos descriptivos adicionales del cliente (nuevas columnas en el CSV)
        info_cols = [c for c in ['Localidad_Moda', 'Nombre1_Moda'] if c in client_features.columns]
        pretty_cols = {'Localidad_Moda': 'Localidad', 'Nombre1_Moda': 'Nombre'}
        if info_cols:
            st.markdown("**Información del cliente**")
            st.table(
                client_features[['cliente_id'] + info_cols]
                .rename(columns=pretty_cols)
            )

        # Mostrar los features de entrada del modelo para este cliente
        if not df_input.empty:
            st.markdown("**Features utilizados para la predicción**")
            feature_view = df_input.iloc[0][feature_names].rename("Valor").reset_index()
            feature_view.columns = ["Feature", "Valor"]
            st.dataframe(feature_view, use_container_width=True)

        if st.button('Predecir Churn Solo para este Cliente', key='pred_single_btn'):
            try:
                proba = model.predict_proba(df_input)[:, 1][0]
                
                if proba >= ADJUSTED_THRESHOLD:
                    prediccion = 'ALTO RIESGO de CHURN (Se recomienda intervención)'
                    st.error(f" **Riesgo de Abandono:** {prediccion}")
                else:
                    prediccion = 'BAJO RIESGO de CHURN (Cliente estable)'
                    st.success(f" **Riesgo de Abandono:** {prediccion}")
                    
                st.info(f"Probabilidad de Churn (P(1)): **{proba:.4f}**")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error al realizar la predicción. Error: {e}")


# --- TAB 2: REPORTES (ALTAIR) ---
with tab_reportes:
    st.header("Reportes y Visualizaciones de Churn")
    st.markdown("Diagnóstico de dispersión (gap entre compras) y distribución de riesgo.")
    
    # Botón para generar y mostrar los reportes
    if st.button('Generar y Mostrar Reportes Gráficos (Altair)', key='report_gen_btn'):
        st.info("Generando gráficos. (Optimizados con caché).")
        
        try:
            chart_clientes, chart_localidades, chart_tipos = create_altair_charts()

            if isinstance(chart_clientes, dict) and 'error' in chart_clientes:
                 st.error(chart_clientes['error'])
            else:
                # 1. Gráficos de barra superiores (Clientes y Localidades)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Clientes con Mayor Dispersión")
                    st.altair_chart(chart_clientes, use_container_width=True)
                
                with col2:
                    st.subheader("Top Localidades por Dispersión")
                    st.altair_chart(chart_localidades, use_container_width=True)
                
                st.markdown("---")
                
                # 2. Tercer gráfico (Tipos de pedido en Alta Dispersión)
                # Este gráfico incluye el slider de 'gap_thr' integrado en la definición de Altair
                st.subheader("Distribución de Tipos de Pedido en Alta Dispersión")
                st.altair_chart(chart_tipos, use_container_width=True)


        except Exception as e:
            st.error(f"Ocurrió un error inesperado al generar los gráficos: {e}")
            st.code(e) # Muestra el error para debugging


# --- TAB 3: MAPA ---
with tab_mapa:
    st.header("Mapa de riesgo por Localidad")
    st.markdown("Visualiza distribución de churn y riesgo alto por localidad sobre un mapa de Argentina.")
    try:
        plot_map = prepare_map_data()
        if plot_map.empty:
            st.info("No hay datos de localidad mapeables o no coinciden con el diccionario de coordenadas.")
        else:
            m, map_err = build_folium_map(plot_map)
            if map_err:
                st.error(map_err)
                st.info("Instala la dependencia y recarga la app.")
            elif m is None:
                st.info("No se pudo construir el mapa con los datos disponibles.")
            else:
                st.components.v1.html(m._repr_html_(), height=700, scrolling=False)

            st.markdown("**Detalle de localidades**")
            st.dataframe(
                plot_map[['Localidad', 'dist_km_maipu', 'clientes', 'abandonaron', 'tasa_abandono', 'riesgo_alto']]
                .sort_values('dist_km_maipu'),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Ocurrió un error al preparar el mapa: {e}")
        st.code(e)


# --- TAB 4: RETENCIÓN ---
with tab_retencion:
    st.header("Análisis de Retención y Gasto Mensual")
    st.markdown(f"Evaluamos si conviene una campaña de retención considerando un costo mensual de **{CAMPAIGN_COST:,.0f}**.")

    retention_df, error_msg = build_retention_view(cust_df_full, model, feature_names, ADJUSTED_THRESHOLD, CAMPAIGN_COST)
    if error_msg:
        st.error(error_msg)
    elif retention_df.empty:
        st.info("No hay datos suficientes para calcular gasto mensual promedio.")
    else:
        # KPIs
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Clientes analizados", f"{len(retention_df):,}")
        with col_b:
            st.metric("Clientes donde conviene retener", f"{(retention_df['Conviene_retener']).sum():,}")
        with col_c:
            st.metric("Promedio gasto mensual", f"{retention_df['Gasto_mensual_prom'].mean():,.0f}")

        st.markdown("**Top clientes (valor x riesgo)**")
        st.dataframe(
            retention_df[['cliente_id', 'Gasto_mensual_prom', 'Probabilidad_Churn', 'Riesgo', 'Conviene_retener', 'Score_valor_riesgo']]
            .head(25),
            use_container_width=True
        )

        # Controles de visualización
        col_filt, col_log = st.columns(2)
        with col_filt:
            only_retain = st.checkbox("Mostrar solo clientes donde conviene retener", value=False)
        with col_log:
            use_log_x = st.checkbox("Usar escala logarítmica en X", value=False)

        scatter_df = retention_df.copy()
        if only_retain:
            scatter_df = scatter_df[scatter_df['Conviene_retener']]
        count_points = len(scatter_df)
        st.caption(f"Puntos en el gráfico: {count_points}")

        # Gráfico: dispersión gasto vs probabilidad
        x_field = alt.X(
            'Gasto_mensual_prom:Q',
            title='Gasto mensual promedio',
            scale=alt.Scale(type='log') if use_log_x else alt.Scale(),
        )
        scatter = (alt.Chart(scatter_df)
            .mark_circle(size=80, opacity=0.7)
            .encode(
                x=x_field,
                y=alt.Y('Probabilidad_Churn:Q', title='Probabilidad de churn'),
                color=alt.Color('Conviene_retener:N', title='Conviene retener'),
                tooltip=['cliente_id', 'Gasto_mensual_prom', 'Probabilidad_Churn', 'Riesgo']
            )
            .properties(height=320)
        )

        st.altair_chart(scatter, use_container_width=True)
