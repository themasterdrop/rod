import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.caching import Cache # Importar Cache para memoización
import plotly.express as px
import os
import joblib
import requests
import io
from datetime import datetime
from werkzeug.serving import run_simple

# --- Configuración de la Caché ---
# Es CRÍTICO que la carpeta 'cache-directory' exista y sea escribible
# Puedes crearla manualmente o asegurarte de que el entorno de despliegue la cree.
CACHE_DIRECTORY = './cache-directory'
if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)
    print(f"Directorio de caché creado: {CACHE_DIRECTORY}")

# Configuración del caché
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem', # Opciones: 'redis', 'memcached', 'simple', etc.
    'CACHE_DIR': CACHE_DIRECTORY,
    'CACHE_DEFAULT_TIMEOUT': 300 # Tiempo en segundos (5 minutos) antes de que un resultado expire
}


# --- 1. Carga de los Datos (DataFrame) ---
file_id_data = "1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
url_data = f"https://drive.google.com/uc?export=download&id={file_id_data}"

print("Cargando DataFrame desde Google Drive...")
try:
    df = pd.read_csv(url_data)
    print(f"DataFrame cargado con éxito. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    if df.empty:
        print("ADVERTENCIA: El DataFrame se cargó pero está vacío.")
except Exception as e:
    print(f"Error al cargar el DataFrame: {e}. Se procederá con un DataFrame vacío.")
    df = pd.DataFrame(columns=['EDAD', 'DIFERENCIA_DIAS', 'PRESENCIAL_REMOTO', 'SEGURO', 'SEXO', 'ESPECIALIDAD', 'DIA_SOLICITACITA', 'ATENDIDO'])

# --- 2. Preprocesamiento de Datos para Visualizaciones y Mapeos ---

def clasificar_edad(edad):
    if edad < 13:
        return "Niño"
    elif edad < 19:
        return "Adolescente"
    elif edad < 30:
        return "Joven"
    elif edad < 61:
        return "Adulto"
    elif edad < 200:
        return "Adulto mayor"

if 'EDAD' in df.columns and not df.empty:
    df['Rango de Edad'] = df['EDAD'].apply(clasificar_edad)
else:
    print("ADVERTENCIA: La columna 'EDAD' no se encontró o el DataFrame está vacío. No se creará 'Rango de Edad'.")

def clasificar_dias(dias):
    if dias < 10:
        return "0-9"
    elif dias < 20:
        return "10-19"
    elif dias < 30:
        return "20-29"
    elif dias < 40:
        return "30-39"
    elif dias < 50:
        return "40-49"
    elif dias < 60:
        return "50-59"
    elif dias < 70:
        return "60-69"
    elif dias < 80:
        return "70-79"
    elif dias < 90:
        return "80-89"
    else:
        return "90+"

if 'DIFERENCIA_DIAS' in df.columns and not df.empty:
    df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias)
else:
    print("ADVERTENCIA: La columna 'DIFERENCIA_DIAS' no se encontró o el DataFrame está vacío. No se creará 'RANGO_DIAS'.")

if 'DIA_SOLICITACITA' in df.columns and not df.empty:
    print(f"DEBUG: Columna 'DIA_SOLICITACITA' presente. Total de NaNs antes de conversión: {df['DIA_SOLICITACITA'].isnull().sum()}")
    df['DIA_SOLICITACITA'] = pd.to_datetime(df['DIA_SOLICITACITA'], errors='coerce')
    print(f"DEBUG: Total de NaNs en 'DIA_SOLICITACITA' después de conversión: {df['DIA_SOLICITACITA'].isnull().sum()}")
    
    # Filtrar solo las filas con fechas válidas antes de crear 'MES' y agrupar
    df_valid_dates = df.dropna(subset=['DIA_SOLICITACITA'])
    print(f"DEBUG: Filas con fechas válidas para 'DIA_SOLICITACITA': {df_valid_dates.shape[0]}")

    if not df_valid_dates.empty:
        df_valid_dates['MES'] = df_valid_dates['DIA_SOLICITACITA'].dt.to_period('M').astype(str)
        citas_por_mes = df_valid_dates.groupby('MES').size().reset_index(name='CANTIDAD_CITAS')
        print(f"DEBUG: citas_por_mes DataFrame (primeras 5 filas):\n{citas_por_mes.head()}")
        print(f"DEBUG: citas_por_mes DataFrame (últimas 5 filas):\n{citas_por_mes.tail()}")
        print(f"DEBUG: Total de entradas en citas_por_mes: {citas_por_mes.shape[0]}")
    else:
        citas_por_mes = pd.DataFrame(columns=['MES', 'CANTIDAD_CITAS'])
        print("ADVERTENCIA: No hay fechas válidas en 'DIA_SOLICITACITA'. 'citas_por_mes' estará vacío.")
else:
    citas_por_mes = pd.DataFrame(columns=['MES', 'CANTIDAD_CITAS'])
    print("ADVERTENCIA: La columna 'DIA_SOLICITACITA' no se encontró o el DataFrame está vacío. 'citas_por_mes' estará vacío.")


# --- Creación del mapeo de ESPECIALIDAD a ESPECIALIDAD_cod para el simulador ---
unique_especialidades = []
especialidad_to_cod = {}

if 'ESPECIALIDAD' in df.columns and not df.empty:
    unique_especialidades = sorted(df['ESPECIALIDAD'].unique().tolist())
    especialidad_to_cod = {especialidad: i for i, especialidad in enumerate(unique_especialidades)}
    print(f"Mapeo de especialidades creado a partir del DataFrame: {especialidad_to_cod}")
else:
    print("La columna 'ESPECIALIDAD' no se encontró en el DataFrame o el DataFrame está vacío.")
    print("Usando un mapeo de especialidades predefinido para el simulador.")
    especialidades_cod_a_nombre_predefinidas = {
        17: 'GERIATRIA', 16: 'GASTROENTEROLOGIA', 13: 'ENDOCRINOLOGIA', 51: 'PSIQUIATRIA',
        2: 'CARDIOLOGIA', 61: 'UROLOGIA', 50: 'PSICOLOGIA', 6: 'CIRUGIA GENERAL',
        34: 'NEUROLOGIA', 20: 'HEMATOLOGIA', 26: 'MEDICINA INTERNA', 42: 'OFTALMOLOGIA',
        54: 'REUMATOLOGIA', 4: 'CIRUGIA PLASTICA Y QUEMADOS', 33: 'NEUROCIRUGIA',
        48: 'PEDIATRIA GENERAL', 27: 'NEFROLOGIA', 35: 'NEUROLOGIA PEDIATRICA',
        40: 'OBSTETRICIA', 29: 'NEUMOLOGIA', 43: 'ONCOLOGIA GINECOLOGIA',
        28: 'NEONATOLOGIA', 21: 'INFECTOLOGIA', 0: 'ADOLESCENTE', 18: 'GINECOLOGIA',
        10: 'DERMATOLOGIA', 8: 'CIRUGIA PEDIATRICA', 56: 'TRAUMATOLOGIA',
        47: 'PATOLOGIA MAMARIA', 46: 'OTORRINOLARINGOLOGIA', 12: 'ECOGRAFIA',
        25: 'MEDICINA FÍSICA Y REHABILITACIÓN', 31: 'NEUMOLOGIA PEDIATRICA',
        44: 'ONCOLOGIA MEDICA', 5: 'CIRUGIA CABEZA Y CUELLO', 7: 'CIRUGIA MAXILO-FACIAL',
        19: 'GINECOLOGIA DE ALTO RIESGO', 36: 'NEUROPSICOLOGIA', 52: 'PUERPERIO',
        59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS', 3: 'CARDIOLOGIA PEDIATRICA',
        41: 'ODONTOLOGIA', 53: 'RADIOTERAPIA', 9: 'CIRUGIA TORAXICA',
        37: 'NUTRICION - ENDOCRINOLOGIA', 57: 'TUBERCULOSIS', 38: 'NUTRICION - MEDICINA',
        22: 'INFECTOLOGIA PEDIATRICA', 30: 'NEUMOLOGIA FUNCION RESPIRATORIA',
        39: 'NUTRICION - PEDIATRICA', 14: 'ENDOCRINOLOGIA PEDIATRICA',
        55: 'SALUD MENTAL ', 23: 'INFERTILIDAD', 45: 'ONCOLOGIA QUIRURGICA',
        32: 'NEUMOLOGIA TEST DE CAMINATA', 49: 'PLANIFICACION FAMILIAR',
        24: 'MEDICINA ALTERNATIVA', 1: 'ANESTESIOLOGIA', 11: 'DERMATOLOGIA PEDIATRICA',
        58: 'TUBERCULOSIS PEDIATRICA', 62: 'ZPRUEBA', 60: 'URODINAMIA',
        15: 'ENDOCRINOLOGIA TUBERCULOSIS'
    }
    especialidad_to_cod = {nombre: codigo for codigo, nombre in especialidades_cod_a_nombre_predefinidas.items()}
    unique_especialidades = sorted(list(especialidad_to_cod.keys()))

especialidades_cod_a_nombre = {
    17: 'GERIATRIA', 16: 'GASTROENTEROLOGIA', 13: 'ENDOCRINOLOGIA', 51: 'PSIQUIATRIA',
    2: 'CARDIOLOGIA', 61: 'UROLOGIA', 50: 'PSICOLOGIA', 6: 'CIRUGIA GENERAL',
    34: 'NEUROLOGIA', 20: 'HEMATOLOGIA', 26: 'MEDICINA INTERNA', 42: 'OFTALMOLOGIA',
    54: 'REUMATOLOGIA', 4: 'CIRUGIA PLASTICA Y QUEMADOS', 33: 'NEUROCIRUGIA',
    48: 'PEDIATRIA GENERAL', 27: 'NEFROLOGIA', 35: 'NEUROLOGIA PEDIATRICA',
    40: 'OBSTETRICIA', 29: 'NEUMOLOGIA', 43: 'ONCOLOGIA GINECOLOGIA',
    28: 'NEONATOLOGIA', 21: 'INFECTOLOGIA', 0: 'ADOLESCENTE', 18: 'GINECOLOGIA',
    10: 'DERMATOLOGIA', 8: 'CIRUGIA PEDIATRICA', 56: 'TRAUMATOLOGIA',
    47: 'PATOLOGIA MAMARIA', 46: 'OTORRINOLARINGOLOGIA', 12: 'ECOGRAFIA',
    25: 'MEDICINA FÍSICA Y REHABILITACIÓN', 31: 'NEUMOLOGIA PEDIATRICA',
    44: 'ONCOLOGIA MEDICA', 5: 'CIRUGIA CABEZA Y CUELLO', 7: 'CIRUGIA MAXILO-FACIAL',
    19: 'GINECOLOGIA DE ALTO RIESGO', 36: 'NEUROPSICOLOGIA', 52: 'PUERPERIO',
    59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS', 3: 'CARDIOLOGIA PEDIATRICA',
    41: 'ODONTOLOGIA', 53: 'RADIOTERAPIA', 9: 'CIRUGIA TORAXICA',
    37: 'NUTRICION - ENDOCRINOLOGIA', 57: 'TUBERCULOSIS', 38: 'NUTRICION - MEDICINA',
    22: 'INFECTOLOGIA PEDIATRICA', 30: 'NEUMOLOGIA FUNCION RESPIRATORIA',
    39: 'NUTRICION - PEDIATRICA', 14: 'ENDOCRINOLOGIA PEDIATRICA',
    55: 'SALUD MENTAL ', 23: 'INFERTILIDAD', 45: 'ONCOLOGIA QUIRURGICA',
    32: 'NEUMOLOGIA TEST DE CAMINATA', 49: 'PLANIFICACION FAMILIAR',
    24: 'MEDICINA ALTERNATIVA', 1: 'ANESTESIOLOGIA', 11: 'DERMATOLOGIA PEDIATRICA',
    58: 'TUBERCULOSIS PEDIATRICA', 62: 'ZPRUEBA', 60: 'URODINAMIA',
    15: 'ENDOCRINOLOGIA TUBERCULOSIS'
}


# --- 3. Carga del Modelo de Machine Learning ---
HF_MODEL_URL = "https://huggingface.co/themasterdrop/simulador_citas_modelo/resolve/main/modelo_forest.pkl?download=true"

print("Descargando modelo desde Hugging Face...")
modelo_forest = None

try:
    response = requests.get(HF_MODEL_URL)
    response.raise_for_status()

    model_bytes = io.BytesIO(response.content)
    modelo_forest = joblib.load(model_bytes)
    print("¡Modelo cargado con éxito usando joblib!")
    if hasattr(modelo_forest, 'feature_names_in_'):
        print(f"Características esperadas por el modelo: {modelo_forest.feature_names_in_}")
    else:
        print("El modelo no tiene 'feature_names_in_'. Asegúrate de que las columnas de entrada para la predicción coincidan con el entrenamiento.")

except requests.exceptions.RequestException as e:
    print(f"ERROR al descargar el modelo desde Hugging Face: {e}")
except Exception as e:
    print(f"ERROR inesperado al cargar el modelo con joblib: {e}")
    print("Asegúrate de que el archivo .pkl fue guardado correctamente con joblib y es compatible con el entorno.")


# --- 4. Configuración del Servidor Flask y las Aplicaciones Dash ---

server = Flask(__name__)

# Inicializar la caché con el servidor Flask
cache = Cache(server, config=CACHE_CONFIG)

@server.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(server.root_path, 'static'), filename)

@server.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Bienvenido</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f6f8;
                text-align: center;
                padding: 50px;
                color: #333;
            }
            h2 {
                color: #2c3e50;
            }
            .logo {
                width: 80px;
                height: auto;
                margin-bottom: 20px;
            }
            .container {
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                display: inline-block;
                max-width: 600px;
                width: 100%;
                animation: fadeIn 1s ease-in-out;
            }
            .links {
                margin-top: 30px;
            }
            a {
                display: inline-block;
                margin: 10px;
                margin-bottom: 15px;
                padding: 12px 24px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            a:hover {
                background-color: #2980b9;
                transform: scale(1.05);
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/logo.png" alt="Logo de la Institución" class="logo">
            <h2>Bienvenido</h2>
            <p>Explora las siguientes visualizaciones:</p>
            <div class="links">
                <a href="/edad/">Distribución por Edad</a>
                <a href="/espera/">Tiempos de Espera</a>
                <a href="/modalidad/">Modalidad de Atención</a>
                <a href="/asegurados/">Estado del Seguro</a>
                <a href="/tiempo/">Línea de Tiempo</a>
                <a href="/simulador/">Simulador de Citas</a>
            </div>
        </div>
    </body>
    </html>
    """)

# --- 5. Definición y Registro de Aplicaciones Dash ---

# App 1: Distribución por Rango de Edad
app_edad = dash.Dash(__name__, server=server, url_base_pathname='/edad/')
app_edad.layout = html.Div([
    html.H1("Distribución por Rango de Edad"),
    dcc.Graph(id='histogram-edad', figure=px.histogram(
        df,
        x='Rango de Edad',
        category_orders={'Rango de Edad': ["Niño", "Adolescente", "Joven", "Adulto", "Adulto mayor"]},
        title='Distribución de edades de los pacientes',
        labels={'Rango de Edad': 'Rango de Edad'},
        template='plotly_white'
    )),
    dcc.Graph(id='pie-chart-edad', figure=px.pie(
        names=[], values=[], title="Seleccione una barra en el histograma"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
])

@app_edad.callback(
    Output('pie-chart-edad', 'figure'),
    Input('histogram-edad', 'clickData')
)
@cache.memoize() # Decorador para cachear los resultados de esta función
def update_pie_chart_edad(clickData):
    print(f"DEBUG: update_pie_chart_edad - clickData: {clickData}")
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    print(f"DEBUG: update_pie_chart_edad - selected_range: {selected_range}")

    if 'Rango de Edad' not in df.columns or df.empty:
        print("DEBUG: update_pie_chart_edad - 'Rango de Edad' no está en df.columns o df está vacío.")
        return px.pie(names=[], values=[], title="Datos no disponibles para rango de edad", height=500)

    filtered_df = df[df['Rango de Edad'] == selected_range].copy()
    print(f"DEBUG: update_pie_chart_edad - filtered_df shape: {filtered_df.shape}")

    if filtered_df.empty or 'ESPECIALIDAD' not in filtered_df.columns:
        print(f"DEBUG: update_pie_chart_edad - filtered_df vacío o sin columna 'ESPECIALIDAD' para {selected_range}")
        return px.pie(names=[], values=[], title=f"No hay datos de especialidad para '{selected_range}'", height=500)

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    print(f"DEBUG: update_pie_chart_edad - Top especialidades: {top_especialidades.index.tolist()}")

    if not top_especialidades.empty:
        filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
            lambda x: x if x in top_especialidades.index else 'Otras'
        )
        grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
        grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    else:
        grouped = pd.DataFrame(columns=['ESPECIALIDAD', 'CUENTA'])

    print(f"DEBUG: update_pie_chart_edad - Grouped data: {grouped.to_dict('records')}")

    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de edad '{selected_range}'",
        height=600
    )

# App 2: Distribución por Tiempo de Espera
app_espera = dash.Dash(__name__, server=server, url_base_pathname='/espera/')
app_espera.layout = html.Div([
    html.H1("Distribución por Tiempo de Espera"),
    dcc.Graph(id='histogram-espera', figure=px.histogram(
        df,
        x='RANGO_DIAS',
        category_orders={'RANGO_DIAS': ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]},
        title='Distribución de la Cantidad de Pacientes según su Tiempo de Espera',
        labels={'RANGO_DIAS': 'Rango de Días'},
        template='plotly_white'
    )),
    dcc.Graph(id='pie-chart-espera', figure=px.pie(
        names=[], values=[], title="Seleccione una barra en el histograma"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
])

@app_espera.callback(
    Output('pie-chart-espera', 'figure'),
    Input('histogram-espera', 'clickData')
)
@cache.memoize() # Decorador para cachear los resultados de esta función
def update_pie_chart_espera(clickData):
    print(f"DEBUG: update_pie_chart_espera - clickData: {clickData}")
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    print(f"DEBUG: update_pie_chart_espera - selected_range: {selected_range}")

    if 'RANGO_DIAS' not in df.columns or df.empty:
        print("DEBUG: update_pie_chart_espera - 'RANGO_DIAS' no está en df.columns o df está vacío.")
        return px.pie(names=[], values=[], title="Datos no disponibles para rango de días", height=500)

    filtered_df = df[df['RANGO_DIAS'] == selected_range].copy()
    print(f"DEBUG: update_pie_chart_espera - filtered_df shape: {filtered_df.shape}")

    if filtered_df.empty or 'ESPECIALIDAD' not in filtered_df.columns:
        print(f"DEBUG: update_pie_chart_espera - filtered_df vacío o sin columna 'ESPECIALIDAD' para {selected_range}")
        return px.pie(names=[], values=[], title=f"No hay datos de especialidad para '{selected_range}' días", height=500)

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    print(f"DEBUG: update_pie_chart_espera - Top especialidades: {top_especialidades.index.tolist()}")

    if not top_especialidades.empty:
        filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
            lambda x: x if x in top_especialidades.index else 'Otras'
        )
        grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
        grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    else:
        grouped = pd.DataFrame(columns=['ESPECIALIDAD', 'CUENTA'])

    print(f"DEBUG: update_pie_chart_espera - Grouped data: {grouped.to_dict('records')}")

    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de espera '{selected_range}' días",
        height=600
    )

# App 3: Distribución por Modalidad de Cita
app_modalidad = dash.Dash(__name__, server=server, url_base_pathname='/modalidad/')
app_modalidad.layout = html.Div([
    html.H1("Distribución por Modalidad de Cita"),
    dcc.Graph(id='pie-modalidad', figure=px.pie(
        df,
        names='PRESENCIAL_REMOTO',
        title='Distribución de Citas: Remotas vs Presenciales',
        template='plotly_white'
    )),
    dcc.Graph(id='bar-especialidad-modalidad', figure=px.bar(
        pd.DataFrame(columns=['ESPECIALIDAD', 'DIFERENCIA_DIAS']),
        x='ESPECIALIDAD',
        y='DIFERENCIA_DIAS',
        title="Seleccione una modalidad en el gráfico de pastel"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
])

@app_modalidad.callback(
    Output('bar-especialidad-modalidad', 'figure'),
    Input('pie-modalidad', 'clickData')
)
@cache.memoize() # Decorador para cachear los resultados de esta función
def update_bar_modalidad(clickData):
    print(f"DEBUG: update_bar_modalidad - clickData: {clickData}")
    if clickData is None:
        return px.bar(x=[], y=[], title="Seleccione una modalidad en el gráfico de pastel")

    modalidad = clickData['points'][0]['label']
    print(f"DEBUG: update_bar_modalidad - selected modalidad: {modalidad}")

    if 'PRESENCIAL_REMOTO' not in df.columns or df.empty:
        print("DEBUG: update_bar_modalidad - 'PRESENCIAL_REMOTO' no está en df.columns o df está vacío.")
        return px.bar(x=[], y=[], title="Datos no disponibles para modalidad", height=500)

    filtered_df = df[df['PRESENCIAL_REMOTO'] == modalidad]
    print(f"DEBUG: update_bar_modalidad - filtered_df shape: {filtered_df.shape}")

    if filtered_df.empty or 'ESPECIALIDAD' not in filtered_df.columns or 'DIFERENCIA_DIAS' not in filtered_df.columns:
        print(f"DEBUG: update_bar_modalidad - filtered_df vacío o sin columnas necesarias para {modalidad}")
        return px.bar(x=[], y=[], title=f"No hay datos de espera por especialidad para '{modalidad}'", height=500)

    mean_wait = filtered_df.groupby('ESPECIALIDAD')['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)
    print(f"DEBUG: update_bar_modalidad - Mean wait data: {mean_wait.to_dict('records')}")

    return px.bar(
        mean_wait,
        x='ESPECIALIDAD',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por Especialidad ({modalidad})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
    )


# App 4: Distribución por Estado del Seguro (Corregido nombre de la instancia a app_seguro)
app_seguro = dash.Dash(__name__, server=server, url_base_pathname='/asegurados/')
app_seguro.layout = html.Div([
    html.H1("Distribución por Estado del Seguro"),
    dcc.Graph(id='pie-seguro', figure=px.pie(
        df.dropna(subset=['SEGURO']) if 'SEGURO' in df.columns and not df.empty else pd.DataFrame(columns=['SEGURO']), # Asegurarse de manejar NaNs si los hay en esta columna
        names='SEGURO',
        title='Distribución de Pacientes: Asegurados vs No Asegurados',
        template='plotly_white'
    )),
    dcc.Graph(id='bar-espera-seguro', figure=px.bar(
        pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title="Seleccione una categoría en el gráfico de pastel"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
])

@app_seguro.callback( # Callback asociada a app_seguro
    Output('bar-espera-seguro', 'figure'),
    Input('pie-seguro', 'clickData')
)
@cache.memoize() # Decorador para cachear los resultados de esta función
def update_bar_seguro(clickData):
    print(f"DEBUG: update_bar_seguro - clickData: {clickData}")
    if clickData is None:
        return px.bar(x=[], y=[], title="Seleccione una categoría en el gráfico de pastel")

    seguro = clickData['points'][0]['label']
    print(f"DEBUG: update_bar_seguro - selected seguro: {seguro}")

    if 'SEGURO' not in df.columns or df.empty:
        print("DEBUG: update_bar_seguro - 'SEGURO' no está en df.columns o df está vacío.")
        return px.bar(x=[], y=[], title="Datos no disponibles para seguro", height=500)

    filtered_df = df[df['SEGURO'] == seguro]
    print(f"DEBUG: update_bar_seguro - filtered_df shape: {filtered_df.shape}")

    if filtered_df.empty or 'SEXO' not in filtered_df.columns or 'DIFERENCIA_DIAS' not in filtered_df.columns:
        print(f"DEBUG: update_bar_seguro - filtered_df vacío o sin columnas necesarias para {seguro}")
        return px.bar(x=[], y=[], title=f"No hay datos de espera por sexo para '{seguro}'", height=500)

    mean_wait = filtered_df.groupby('SEXO')['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)
    print(f"DEBUG: update_bar_seguro - Mean wait data: {mean_wait.to_dict('records')}")

    fig = px.bar(
        mean_wait,
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por SEXO ({seguro})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
    )
    fig.update_yaxes(range=[mean_wait['DIFERENCIA_DIAS'].min() - 1, mean_wait['DIFERENCIA_DIAS'].max() + 1] if not mean_wait.empty else [0, 1])
    return fig


# App 5: Línea de Tiempo (Citas Agendadas por Mes - Corregido nombre de la instancia a app_tiempo)
app_tiempo = dash.Dash(__name__, server=server, url_base_pathname='/tiempo/')
app_tiempo.layout = html.Div([
    html.H1("Citas Agendadas por Mes"),
    dcc.Graph(
        id='grafico-lineal',
        figure=px.line(citas_por_mes, x='MES', y='CANTIDAD_CITAS', markers=True,
                       title='Cantidad de Citas por Mes')
    ),
    html.Div([
        dcc.Graph(id='grafico-pie-especialidades'),
        dcc.Graph(id='grafico-pie-atencion')
    ]),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
])

@app_tiempo.callback(
    [Output('grafico-pie-especialidades', 'figure'),
     Output('grafico-pie-atencion', 'figure')],
    [Input('grafico-lineal', 'clickData')]
)
@cache.memoize() # Decorador para cachear los resultados de esta función
def actualizar_graficos(clickData):
    print(f"DEBUG: actualizar_graficos - clickData: {clickData}")
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione un mes"), \
               px.pie(names=[], values=[], title="Seleccione un mes")

    mes_seleccionado = clickData['points'][0]['x'] # mes_seleccionado ya viene como string del eje X
    print(f"DEBUG: actualizar_graficos - selected mes: {mes_seleccionado}")

    if 'MES' not in df.columns or df.empty:
        print("DEBUG: actualizar_graficos - 'MES' no está en df.columns o df está vacío.")
        return px.pie(names=[], values=[], title="Datos no disponibles para el mes", height=500), \
               px.pie(names=[], values=[], title="Datos no disponibles para el mes", height=500)


    df_mes = df[df['MES'] == mes_seleccionado]
    print(f"DEBUG: actualizar_graficos - df_mes shape: {df_mes.shape}")

    if df_mes.empty:
        print(f"DEBUG: actualizar_graficos - df_mes está vacío para el mes: {mes_seleccionado}")
        return px.pie(names=[], values=[], title=f"No hay datos para {mes_seleccionado}", height=500), \
               px.pie(names=[], values=[], title=f"No hay datos para {mes_seleccionado}", height=500)

    fig_especialidades = px.pie(names=[], values=[], title=f"No hay datos de especialidad para {mes_seleccionado}", height=500)
    if 'ESPECIALIDAD' in df_mes.columns:
        top_especialidades = df_mes['ESPECIALIDAD'].value_counts().nlargest(5)
        if not top_especialidades.empty:
            df_mes_copy_esp = df_mes.copy()
            df_mes_copy_esp['ESPECIALIDAD_AGRUPADA'] = df_mes_copy_esp['ESPECIALIDAD'].apply(
                lambda x: x if x in top_especialidades.index else 'Otras'
            )
            grouped = df_mes_copy_esp['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
            grouped.columns = ['ESPECIALIDAD', 'CUENTA']
            grouped = grouped.sort_values(by='CUENTA', ascending=False)
            fig_especialidades = px.pie(grouped, names='ESPECIALIDAD', values="CUENTA", title=f'Distribución de Especialidades en {mes_seleccionado}')
        else:
            print(f"DEBUG: actualizar_graficos - top_especialidades vacío para {mes_seleccionado}")
    else:
        print("DEBUG: actualizar_graficos - Columna 'ESPECIALIDAD' no encontrada en df_mes.")

    fig_atencion = px.pie(names=[], values=[], title=f"No hay datos de atención para {mes_seleccionado}", height=500)
    if 'ATENDIDO' in df_mes.columns:
        if not df_mes['ATENDIDO'].empty:
            fig_atencion = px.pie(df_mes, names='ATENDIDO', title=f'Estado de Atención en {mes_seleccionado}')
        else:
            print(f"DEBUG: actualizar_graficos - df_mes['ATENDIDO'] vacío para {mes_seleccionado}")
    else:
        print("DEBUG: actualizar_graficos - Columna 'ATENDIDO' no encontrada en df_mes.")

    return fig_especialidades, fig_atencion


# --- App 6: Simulador de Tiempo de Espera Estimado (Regresión) ---
app_simulador = dash.Dash(__name__, server=server, url_base_pathname='/simulador/')
app_simulador.layout = html.Div([
    html.H1("Simulador de Tiempo de Espera Estimado", className="text-3xl font-bold mb-6 text-blue-800"),
    html.Div([
        html.Label("Edad:", className="block text-gray-700 text-sm font-bold mb-2"),
        dcc.Input(id='sim-input-edad', type='number', value=30, min=0, max=120,
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mb-4"),

        html.Label("Especialidad:", className="block text-gray-700 text-sm font-bold mb-2"),
        dcc.Dropdown(
            id='sim-input-especialidad',
            options=[{'label': nombre, 'value': codigo} for nombre, codigo in especialidad_to_cod.items()],
            value=list(especialidad_to_cod.values())[0] if especialidad_to_cod else None,
            placeholder="Selecciona una especialidad",
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mb-6"
        ),

        html.Button(
            'Predecir Tiempo de Espera',
            id='sim-predict-button',
            n_clicks=0,
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg focus:outline-none focus:shadow-outline w-full transform transition-all duration-200 hover:scale-105"
        ),

        html.Div(
            id='sim-output-prediction',
            className="mt-6 p-4 bg-blue-100 border-l-4 border-blue-500 text-blue-700 text-center font-semibold rounded-lg text-xl"
        )
    ], className="bg-white p-8 rounded-lg shadow-xl max-w-lg mx-auto"),

    html.Div(dcc.Link(
        'Volver a la Página Principal',
        href='/',
        className="inline-block mt-8 py-3 px-6 bg-yellow-600 hover:bg-yellow-700 text-white font-bold rounded-lg text-lg transition-all duration-200 hover:scale-105"
    ))
], className="min-h-screen bg-gray-100 flex flex-col items-center justify-center py-10 px-4")


@app_simulador.callback(
    Output('sim-output-prediction', 'children'),
    Input('sim-predict-button', 'n_clicks'),
    Input('sim-input-edad', 'value'),
    Input('sim-input-especialidad', 'value'),
    prevent_initial_call=True
)
@cache.memoize() # También puedes cachear este callback si la predicción es costosa
def predict_dias_espera(n_clicks, edad, especialidad_cod_input):
    print(f"DEBUG: predict_dias_espera - n_clicks: {n_clicks}, edad: {edad}, especialidad_cod_input: {especialidad_cod_input}")
    if n_clicks is None or n_clicks == 0:
        return ""

    if modelo_forest is None:
        print("DEBUG: predict_dias_espera - modelo_forest is None.")
        return html.Div("Error: El modelo de predicción no se pudo cargar. No se puede realizar la predicción.", className="text-red-600 font-bold")

    if edad is None or not (0 <= edad <= 120):
        return html.Div("Error: Edad no válida. Por favor, ingrese una edad entre 0 y 120.", className="text-red-600 font-bold")
    if especialidad_cod_input is None:
        return html.Div("Error: Por favor, seleccione una especialidad.", className="text-red-600 font-bold")

    today = datetime.now()
    dia = today.day
    semana_del_año = today.isocalendar()[1]
    print(f"DEBUG: predict_dias_espera - día: {dia}, semana_del_año: {semana_del_año}")

    nombre_especialidad_para_mostrar = especialidades_cod_a_nombre.get(especialidad_cod_input, "Especialidad Desconocida")
    print(f"DEBUG: predict_dias_espera - nombre_especialidad_para_mostrar: {nombre_especialidad_para_mostrar}")

    input_data = pd.DataFrame([[
        especialidad_cod_input,
        edad,
        dia,
        semana_del_año
    ]],
    columns=[
        'ESPECIALIDAD_cod',
        'EDAD',
        'día',
        'semana_del_año'
    ])
    print(f"DEBUG: predict_dias_espera - input_data for model: \n{input_data}")

    try:
        predicted_days = modelo_forest.predict(input_data)[0]
        predicted_days_rounded = max(0, round(predicted_days))
        print(f"DEBUG: predict_dias_espera - Predicted days: {predicted_days_rounded}")

        return html.Div([
            html.P(f"Para la especialidad de ", className="inline"),
            html.Span(f"{nombre_especialidad_para_mostrar}", className="font-bold text-blue-800"),
            html.P(f", el tiempo de espera estimado es de:", className="inline"),
            html.P(f"{predicted_days_rounded} días", className="text-4xl font-extrabold text-green-700 mt-2")
        ], className="prediction-output-text")
    except Exception as e:
        print(f"ERROR: predict_dias_espera - Error during prediction: {e}")
        return html.Div(f"Error al realizar la predicción: {e}. Asegúrate de que los datos de entrada coincidan con lo que el modelo espera.", className="text-red-600 font-bold")

application = server

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    print(f"La aplicación se ejecutará en http://0.0.0.0:{port}/")
    print(f"El simulador estará disponible en http://0.0.0.0:{port}/simulador/")
    run_simple('0.0.0.0', port, application, use_reloader=True)

