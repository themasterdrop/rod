import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import joblib  # Para cargar el modelo guardado con joblib
import requests # Para descargar el modelo desde la URL
import io # Para manejar el contenido binario del modelo en memoria
from datetime import datetime # Para obtener la fecha actual (día y semana_del_año)
import os # Para trabajar con rutas de archivos, especialmente para la carpeta 'static'

print("--- Iniciando carga y preprocesamiento de datos y modelo (multi_app.py) ---")

# URLs de los recursos
HF_DATA_URL = "https://drive.google.com/uc?export=download&id=1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
HF_MODEL_URL = "https://huggingface.co/themasterdrop/simulador_citas_modelo/resolve/main/modelo_forest.pkl?download=true"

# Diccionario de especialidades (tal cual lo proporcionaste)
# Las claves son los códigos numéricos, los valores son los nombres de las especialidades.
especialidades_dic = {
    17: 'GERIATRIA', 16: 'GASTROENTEROLOGIA', 13: 'ENDOCRINOLOGIA',
    51: 'PSIQUIATRIA', 2: 'CARDIOLOGIA', 61: 'UROLOGIA', 50: 'PSICOLOGIA',
    6: 'CIRUGIA GENERAL', 34: 'NEUROLOGIA', 20: 'HEMATOLOGIA',
    26: 'MEDICINA INTERNA', 42: 'OFTALMOLOGIA', 54: 'REUMATOLOGIA',
    4: 'CIRUGIA PLASTICA Y QUEMADOS', 33: 'NEUROCIRUGIA',
    48: 'PEDIATRIA GENERAL', 27: 'NEFROLOGIA',
    35: 'NEUROLOGIA PEDIATRICA', 40: 'OBSTETRICIA', 29: 'NEUMOLOGIA',
    43: 'ONCOLOGIA GINECOLOGIA', 28: 'NEONATOLOGIA', 21: 'INFECTOLOGIA',
    0: 'ADOLESCENTE', 18: 'GINECOLOGIA', 10: 'DERMATOLOGIA',
    8: 'CIRUGIA PEDIATRICA', 56: 'TRAUMATOLOGIA', 47: 'PATOLOGIA MAMARIA',
    46: 'OTORRINOLARINGOLOGIA', 12: 'ECOGRAFIA',
    25: 'MEDICINA FÍSICA Y REHABILITACIÓN', 31: 'NEUMOLOGIA PEDIATRICA',
    44: 'ONCOLOGIA MEDICA', 5: 'CIRUGIA CABEZA Y CUELLO',
    7: 'CIRUGIA MAXILO-FACIAL', 19: 'GINECOLOGIA DE ALTO RIESGO',
    36: 'NEUROPSICOLOGIA', 52: 'PUERPERIO',
    59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS', 3: 'CARDIOLOGIA PEDIATRICA',
    41: 'ODONTOLOGIA', 53: 'RADIOTERAPIA', 9: 'CIRUGIA TORAXICA',
    37: 'NUTRICION - ENDOCRINOLOGIA', 57: 'TUBERCULOSIS',
    38: 'NUTRICION - MEDICINA', 22: 'INFECTOLOGIA PEDIATRICA',
    30: 'NEUMOLOGIA FUNCION RESPIRATORIA', 39: 'NUTRICION - PEDIATRICA',
    14: 'ENDOCRINOLOGIA PEDIATRICA', 55: 'SALUD MENTAL ', 23: 'INFERTILIDAD',
    45: 'ONCOLOGIA QUIRURGICA', 32: 'NEUMOLOGIA TEST DE CAMINATA',
    49: 'PLANIFICACION FAMILIAR', 24: 'MEDICINA ALTERNATIVA',
    1: 'ANESTESIOLOGIA', 11: 'DERMATOLOGIA PEDIATRICA',
    58: 'TUBERCULOSIS PEDIATRICA', 62: 'ZPRUEBA',
    60: 'URODINAMIA', 15: 'ENDOCRINOLOGIA TUBERCULOSIS'
}


# --- Carga y Preprocesamiento de Datos del DataFrame ---
df = pd.DataFrame() # Inicializa un DataFrame vacío para manejar posibles errores
citas_por_mes = pd.DataFrame(columns=['MES', 'CANTIDAD_CITAS']) # Inicializa vacío

try:
    df = pd.read_csv(HF_DATA_URL)
    print("DataFrame descargado y cargado con éxito.")

    # Convertir 'DIA_SOLICITACITA' a datetime y extraer 'MES'
    df['DIA_SOLICITACITA'] = pd.to_datetime(df['DIA_SOLICITACITA'], errors='coerce')
    df['MES'] = df['DIA_SOLICITACITA'].dt.to_period('M').astype(str)

    # Clasificación por edad
    def clasificar_edad(edad):
        if edad < 13: return "Niño"
        elif edad < 19: return "Adolescente"
        elif edad < 30: return "Joven"
        elif edad < 61: return "Adulto"
        else: return "Adulto mayor"
    df['Rango de Edad'] = df['EDAD'].apply(clasificar_edad)

    # Clasificación por días de espera
    def clasificar_dias_visualizacion(dias):
        if dias < 10: return "0-9"
        elif dias < 20: return "10-19"
        elif dias < 30: return "20-29"
        elif dias < 40: return "30-39"
        elif dias < 50: return "40-49"
        elif dias < 60: return "50-59"
        elif dias < 70: return "60-69"
        elif dias < 80: return "70-79"
        elif dias < 90: return "80-89"
        else: return "90+"
    df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias_visualizacion)

    # Datos agrupados para la visualización de línea de tiempo
    citas_por_mes = df.groupby('MES').size().reset_index(name='CANTIDAD_CITAS')

except Exception as e:
    print(f"ERROR FATAL al cargar o preprocesar el DataFrame: {e}")
    # Define un DataFrame vacío con las columnas esperadas para evitar errores en las apps
    df = pd.DataFrame(columns=['DIA_SOLICITACITA', 'MES', 'EDAD', 'Rango de Edad', 'DIFERENCIA_DIAS', 'RANGO_DIAS', 'ESPECIALIDAD', 'PRESENCIAL_REMOTO', 'SEGURO', 'ATENDIDO', 'SEXO'])


# --- Carga del Modelo de Machine Learning (joblib) ---
print("--- Iniciando descarga y carga del modelo (multi_app.py) ---")
modelo_forest = None # Inicializar a None en caso de error

try:
    response = requests.get(HF_MODEL_URL)
    response.raise_for_status() # Lanza una excepción para errores HTTP

    model_bytes = io.BytesIO(response.content)
    modelo_forest = joblib.load(model_bytes) # Carga el modelo con joblib
    print("¡Modelo cargado con éxito usando joblib!")

except requests.exceptions.RequestException as e:
    print(f"ERROR al descargar el modelo desde Hugging Face: {e}")
except Exception as e:
    print(f"ERROR inesperado al cargar el modelo con joblib: {e}")
    print("Asegúrate de que el archivo .pkl fue guardado correctamente con joblib y es compatible.")

print("-" * 40)


# --- Configuración del Servidor Flask Compartido ---
server = Flask(__name__)

# Ruta para servir archivos estáticos (como el logo.png)
@server.route('/static/<path:filename>')
def static_files(filename):
    # Asume que 'static' está en la misma raíz que 'multi_app.py'
    return send_from_directory(os.path.join(server.root_path, 'static'), filename)

# Ruta raíz con enlaces a todas las aplicaciones Dash
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

# --- App 1: Por Rango de Edad ---
app_edad = dash.Dash(__name__, server=server, url_base_pathname='/edad/')
app_edad.layout = html.Div([
    html.H1("Distribución por Rango de Edad", style={'color': '#2c3e50'}),
    dcc.Graph(id='histogram-edad', figure=px.histogram(
        df,
        x='Rango de Edad',
        category_orders={'Rango de Edad': ["Niño", "Adolescente", "Joven", "Adulto", "Adulto mayor"]},
        title='Distribución de edades de los pacientes del hospital María Auxiliadora',
        labels={'Rango de Edad': 'Rango de Edad'},
        template='plotly_white'
    )),
    dcc.Graph(id='pie-chart-edad', figure=px.pie(
        names=[], values=[], title="Seleccione una barra en el histograma"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px', 'minHeight': '100vh'})

@app_edad.callback(
    Output('pie-chart-edad', 'figure'),
    Input('histogram-edad', 'clickData')
)
def update_pie_chart_edad(clickData):
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    filtered_df = df[df['Rango de Edad'] == selected_range].copy()

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
    grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de edad '{selected_range}'",
        height=600
    )

# --- App 2: Por Rango de Días de Espera ---
app_espera = dash.Dash(__name__, server=server, url_base_pathname='/espera/')
app_espera.layout = html.Div([
    html.H1("Distribución por Tiempo de Espera", style={'color': '#2c3e50'}),
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
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px', 'minHeight': '100vh'})

@app_espera.callback(
    Output('pie-chart-espera', 'figure'),
    Input('histogram-espera', 'clickData')
)
def update_pie_chart_espera(clickData):
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    filtered_df = df[df['RANGO_DIAS'] == selected_range].copy()

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
    grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de espera '{selected_range}' días",
        height=600
    )

# --- App 3: Por Modalidad de Cita ---
app_modalidad = dash.Dash(__name__, server=server, url_base_pathname='/modalidad/')
app_modalidad.layout = html.Div([
    html.H1("Distribución por Modalidad de Cita", style={'color': '#2c3e50'}),
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
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px', 'minHeight': '100vh'})

@app_modalidad.callback(
    Output('bar-especialidad-modalidad', 'figure'),
    Input('pie-modalidad', 'clickData')
)
def update_bar_modalidad(clickData):
    if clickData is None:
        return px.bar(x=[], y=[], title="Seleccione una modalidad en el gráfico de pastel")

    modalidad = clickData['points'][0]['label']
    filtered_df = df[df['PRESENCIAL_REMOTO'] == modalidad]
    mean_wait = filtered_df.groupby('ESPECIALIDAD')['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)

    return px.bar(
        mean_wait,
        x='ESPECIALIDAD',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por Especialidad ({modalidad})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
   )


# --- App 4: Por Estado de Seguro (Renombrada para evitar conflicto con app_modalidad original) ---
app_asegurados = dash.Dash(__name__, server=server, url_base_pathname='/asegurados/')
app_asegurados.layout = html.Div([
    html.H1("Distribución por Estado del Seguro", style={'color': '#2c3e50'}),
    dcc.Graph(id='pie-seguro', figure=px.pie(
        df.dropna(subset=['SEGURO']) if 'SEGURO' in df.columns else pd.DataFrame(columns=['SEGURO']),
        names='SEGURO',
        title='Distribución de Pacientes: Asegurados vs No Asegurados',
        template='plotly_white'
    )),
    dcc.Graph(id='bar-espera-seguro', figure=px.bar(
        pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title="Seleccione una opción en el gráfico de pastel"
    )),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px', 'minHeight': '100vh'})

@app_asegurados.callback(
    Output('bar-espera-seguro', 'figure'),
    Input('pie-seguro', 'clickData')
)
def update_bar_seguro(clickData):
    if clickData is None:
        return px.bar(x=[], y=[], title="Seleccione una opción en el gráfico de pastel")

    seguro = clickData['points'][0]['label']
    filtered_df = df[df['SEGURO'] == seguro]
    mean_wait = filtered_df.groupby('SEXO')['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)

    fig = px.bar(
        mean_wait,
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por SEXO ({seguro})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
    )
    # Ajustar rango dinámicamente o dejar fijo
    fig.update_yaxes(range=[0, mean_wait['DIFERENCIA_DIAS'].max() + 2] if not mean_wait.empty else [0, 1])
    return fig


# --- App 5: Línea de Tiempo ---
app_tiempo = dash.Dash(__name__, server=server, url_base_pathname='/tiempo/')
app_tiempo.layout = html.Div([
    html.H1("Citas Agendadas por Mes", style={'color': '#2c3e50'}),
    dcc.Graph(
        id='grafico-lineal',
        figure=px.line(citas_por_mes, x='MES', y='CANTIDAD_CITAS', markers=True,
                       title='Cantidad de Citas por Mes')
    ),
    html.Div([
        dcc.Graph(id='grafico-pie-especialidades'),
        dcc.Graph(id='grafico-pie-atencion')
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px', 'minHeight': '100vh'})

@app_tiempo.callback(
    [Output('grafico-pie-especialidades', 'figure'),
     Output('grafico-pie-atencion', 'figure')],
    [Input('grafico-lineal', 'clickData')]
)
def actualizar_graficos(clickData):
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione un mes"), px.pie(names=[], values=[], title="Seleccione un mes")

    mes_seleccionado = pd.to_datetime(clickData['points'][0]['x']).to_period('M').strftime('%Y-%m')
    df_mes = df[df['MES'] == mes_seleccionado].copy()

    top_especialidades = df_mes['ESPECIALIDAD'].value_counts().nlargest(5)
    df_mes['ESPECIALIDAD_AGRUPADA'] = df_mes['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    grouped = df_mes['ESPECIALIDAD_AGRUPADA'].value_counts().reset_index()
    grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    grouped = grouped.sort_values(by='CUENTA', ascending=False)

    fig_especialidades = px.pie(grouped, names='ESPECIALIDAD', values="CUENTA", title=f'Distribución de Especialidades en {mes_seleccionado}')
    fig_atencion = px.pie(df_mes, names='ATENDIDO', title=f'Estado de Atención en {mes_seleccionado}')

    return fig_especialidades, fig_atencion


# --- App 6: Simulador de Tiempo de Espera (NUEVA APP) ---
simulador_app = dash.Dash(__name__, server=server, url_base_pathname='/simulador/')
simulador_app.layout = html.Div([
    html.H1("Simulador de Tiempo de Espera Estimado", style={'color': '#2c3e50', 'marginBottom': '30px'}),
    html.Div([
        html.Label("Edad:", style={'display': 'block', 'marginBottom': '5px', 'fontWeight': 'bold'}),
        dcc.Input(id='sim-input-edad', type='number', value=30, min=0, max=120, className="input-field", style={'width': 'calc(100% - 20px)', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd', 'marginBottom': '15px'}),
        
        html.Label("Especialidad:", style={'display': 'block', 'marginBottom': '5px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='sim-input-especialidad',
            options=[{'label': v, 'value': k} for k, v in especialidades_dic.items()], # Aquí se usa especialidades_dic
            value=17, # Valor por defecto (GERIATRIA) o el que prefieras
            placeholder="Selecciona una especialidad",
            className="dropdown-field",
            style={'marginBottom': '20px'}
        ),
        
        html.Button('Predecir Tiempo de Espera', id='sim-predict-button', n_clicks=0, className="button-predict",
                    style={'backgroundColor': '#28a745', 'color': 'white', 'padding': '12px 25px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px', 'transition': 'background-color 0.3s ease', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        html.Div(id='sim-output-prediction', style={'marginTop': '30px', 'fontSize': '22px', 'fontWeight': 'bold', 'color': '#007bff'})
    ], style={'padding': '30px', 'border': '1px solid #e0e0e0', 'borderRadius': '10px', 'maxWidth': '550px', 'margin': '40px auto', 'backgroundColor': '#ffffff', 'boxShadow': '0 5px 15px rgba(0,0,0,0.08)'}),
    
    html.Br(),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '30px', 'padding': '12px 25px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px', 'fontSize': '16px', 'transition': 'background-color 0.3s ease'}))
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '40px 20px', 'minHeight': '100vh', 'boxSizing': 'border-box'})


@simulador_app.callback(
    Output('sim-output-prediction', 'children'),
    Input('sim-predict-button', 'n_clicks'),
    Input('sim-input-edad', 'value'),
    Input('sim-input-especialidad', 'value'), # Este es el código numérico
    prevent_initial_call=True
)
def predecir(n_clicks, edad, especialidad_cod_input): # Renombrado a especialidad_cod_input
    if n_clicks is None or n_clicks == 0:
        return ""

    if modelo_forest is None:
        return "❌ Error: El modelo de predicción no se pudo cargar. Contacta al administrador."
    if edad is None or especialidad_cod_input is None:
        return "⚠️ Por favor, ingrese la edad y seleccione una especialidad para la predicción."

    # Obtener día y semana_del_año de la fecha actual
    today = datetime.now()
    dia = today.day
    semana_del_año = today.isocalendar()[1]

    # Crear el DataFrame de entrada para el modelo
    # Las columnas y su orden DEBEN coincidir con el entrenamiento del modelo:
    # ['ESPECIALIDAD_cod', 'EDAD', 'día', 'semana_del_año']
    input_data = pd.DataFrame([[
        especialidad_cod_input, # Usamos el código numérico directamente
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

    try:
        predicted_days = modelo_forest.predict(input_data)[0]
        predicted_days_rounded = max(0, round(predicted_days))

        # Recuperar el nombre de la especialidad para mostrarlo en el resultado
        nombre_especialidad = especialidades_dic.get(especialidad_cod_input, "Especialidad Desconocida")

        return f"Especialidad: {nombre_especialidad} — Tiempo estimado de espera: ➡️ **{predicted_days_rounded} días**."
    except Exception as e:
        return f"❌ Error al realizar la predicción: {e}. Asegúrate de que los datos de entrada coincidan con lo que el modelo espera."


# --- Punto de Entrada para Gunicorn y Desarrollo Local ---
# 'application' es el nombre que Gunicorn buscará para iniciar tu app en Render.
application = server

if __name__ == '__main__':
    # Este bloque solo se ejecuta cuando corres el script localmente (python multi_app.py)
    # No se ejecuta en el entorno de Render cuando Gunicorn lo inicia.
    port = int(os.environ.get("PORT", 8050)) # Render asigna un puerto, localmente usa 8050
    server.run(host='0.0.0.0', port=port, debug=True)
