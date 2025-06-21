import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os
import joblib  # Importamos joblib para cargar el modelo
import requests
import io  # Necesario para manejar el contenido binario del modelo desde la URL
from datetime import datetime # Necesario para 'día' y 'semana_del_año' en el simulador

# --- 1. Carga de los Datos (DataFrame) ---
# Se mantiene tu URL actual de Google Drive ya que has confirmado que funciona en Render.
file_id_data = "1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
url_data = f"https://drive.google.com/uc?export=download&id={file_id_data}"

print("Cargando DataFrame desde Google Drive...")
try:
    df = pd.read_csv(url_data)
    print("DataFrame cargado con éxito.")
except Exception as e:
    print(f"Error al cargar el DataFrame: {e}. Se procederá con un DataFrame vacío.")
    # Crea un DataFrame vacío en caso de error para evitar que la aplicación se detenga
    df = pd.DataFrame(columns=['EDAD', 'DIFERENCIA_DIAS', 'PRESENCIAL_REMOTO', 'SEGURO', 'SEXO', 'ESPECIALIDAD', 'DIA_SOLICITACITA', 'ATENDIDO'])

# --- 2. Preprocesamiento de Datos para Visualizaciones y Mapeos ---

# Clasificación por edad
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

if 'EDAD' in df.columns: # Asegurarse de que la columna exista
    df['Rango de Edad'] = df['EDAD'].apply(clasificar_edad)

# Clasificación por días de espera
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

if 'DIFERENCIA_DIAS' in df.columns: # Asegurarse de que la columna exista
    df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias)

# Transformación para Línea de Tiempo
if 'DIA_SOLICITACITA' in df.columns: # Asegurarse de que la columna exista
    df['DIA_SOLICITACITA'] = pd.to_datetime(df['DIA_SOLICITACITA'], errors='coerce')
    df['MES'] = df['DIA_SOLICITACITA'].dt.to_period('M').astype(str)
    citas_por_mes = df.groupby('MES').size().reset_index(name='CANTIDAD_CITAS')
else:
    citas_por_mes = pd.DataFrame(columns=['MES', 'CANTIDAD_CITAS'])


# --- Creación del mapeo de ESPECIALIDAD a ESPECIALIDAD_cod para el simulador ---
# CRÍTICO: Este mapeo DEBE ser consistente con cómo se codificó 'ESPECIALIDAD'
# cuando se entrenó el modelo. Si usaste un LabelEncoder y lo guardaste, cárgalo.
# De lo contrario, ordenar alfabéticamente es una buena aproximación para consistencia.
unique_especialidades = []
especialidad_to_cod = {}
if 'ESPECIALIDAD' in df.columns:
    unique_especialidades = sorted(df['ESPECIALIDAD'].unique().tolist())
    especialidad_to_cod = {especialidad: i for i, especialidad in enumerate(unique_especialidades)}
    print(f"Mapeo de especialidades creado: {especialidad_to_cod}")
else:
    print("La columna 'ESPECIALIDAD' no se encontró en el DataFrame. El simulador podría no funcionar correctamente.")


# --- 3. Carga del Modelo de Machine Learning ---
# URL para el modelo en Hugging Face
HF_MODEL_URL = "https://huggingface.co/themasterdrop/simulador_citas_modelo/resolve/main/modelo_forest.pkl?download=true"

print("Descargando modelo desde Hugging Face...")
modelo_forest = None # Inicializar a None en caso de error

try:
    response = requests.get(HF_MODEL_URL)
    response.raise_for_status() # Lanza una excepción si la descarga no es exitosa

    # Usar io.BytesIO para tratar el contenido binario de la respuesta como un archivo en memoria
    model_bytes = io.BytesIO(response.content)
    modelo_forest = joblib.load(model_bytes) # Carga el modelo con joblib
    print("¡Modelo cargado con éxito usando joblib!")
    # Opcional: imprimir las características que el modelo espera, si el modelo tiene este atributo
    if hasattr(modelo_forest, 'feature_names_in_'):
        print(f"Características esperadas por el modelo: {modelo_forest.feature_names_in_}")
    else:
        print("El modelo no tiene 'feature_names_in_'. Asegúrate de que las columnas de entrada para la predicción coincidan con el entrenamiento.")

except requests.exceptions.RequestException as e:
    print(f"ERROR al descargar el modelo desde Hugging Face: {e}")
except Exception as e: # Captura cualquier otro error, incluyendo errores específicos de joblib.load
    print(f"ERROR inesperado al cargar el modelo con joblib: {e}")
    print("Asegúrate de que el archivo .pkl fue guardado correctamente con joblib y es compatible con el entorno.")


# --- 4. Configuración del Servidor Flask y las Aplicaciones Dash ---

server = Flask(__name__)

# Ruta para servir archivos estáticos (como el logo.png)
@server.route('/static/<path:filename>')
def static_files(filename):
    # Asegúrate de que la carpeta 'static' existe en la raíz de tu proyecto
    return send_from_directory(os.path.join(server.root_path, 'static'), filename)

# Página principal de Flask con enlaces a las apps Dash
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
                <a href="/simulador/">Simulador de Citas</a> <!-- Enlace al nuevo simulador -->
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


# App 4: Distribución por Estado del Seguro (Corregido nombre de la instancia a app_seguro)
app_seguro = dash.Dash(__name__, server=server, url_base_pathname='/asegurados/')
app_seguro.layout = html.Div([
    html.H1("Distribución por Estado del Seguro"),
    dcc.Graph(id='pie-seguro', figure=px.pie(
        df.dropna(), # Asegurarse de manejar NaNs si los hay en esta columna
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
def update_bar_seguro(clickData):
    if clickData is None:
        return px.bar(x=[], y=[], title="Seleccione una categoría en el gráfico de pastel")

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
    # Considera ajustar este rango dinámicamente si tus datos varían mucho
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
def actualizar_graficos(clickData):
    if clickData is None:
        # Crea figuras vacías para el estado inicial o cuando no hay selección
        return px.pie(names=[], values=[], title="Seleccione un mes"), \
               px.pie(names=[], values=[], title="Seleccione un mes")

    mes_seleccionado = pd.to_datetime(clickData['points'][0]['x']).to_period('M').strftime('%Y-%m')
    df_mes = df[df['MES'] == mes_seleccionado]

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


# --- App 6: Simulador de Tiempo de Espera Estimado (Regresión) ---
app_simulador = dash.Dash(__name__, server=server, url_base_pathname='/simulador/')
app_simulador.layout = html.Div([
    html.H1("Simulador de Tiempo de Espera Estimado"),
    html.Div([
        html.Label("Edad:"),
        dcc.Input(id='sim-input-edad', type='number', value=30, min=0, max=120, className="input-field"),
        html.Br(), # Salto de línea para mejor diseño
        html.Label("Especialidad:"),
        dcc.Dropdown(
            id='sim-input-especialidad',
            options=[{'label': i, 'value': i} for i in unique_especialidades],
            value=unique_especialidades[0] if unique_especialidades else None, # Establecer un valor por defecto si hay especialidades
            placeholder="Selecciona una especialidad",
            className="dropdown-field"
        ),
        html.Br(), # Salto de línea para mejor diseño
        html.Button('Predecir Tiempo de Espera', id='sim-predict-button', n_clicks=0, className="button-predict"),
        html.Div(id='sim-output-prediction', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#3498db'})
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'maxWidth': '500px', 'margin': '20px auto', 'backgroundColor': '#fff', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    html.Br(), # Salto de línea
    # Enlace para volver al inicio
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={'display': 'inline-block', 'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 'textDecoration': 'none', 'borderRadius': '5px'}))
], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px'})


@app_simulador.callback(
    Output('sim-output-prediction', 'children'),
    Input('sim-predict-button', 'n_clicks'),
    Input('sim-input-edad', 'value'),
    Input('sim-input-especialidad', 'value'),
    prevent_initial_call=True
)
def predict_dias_espera(n_clicks, edad, especialidad):
    if n_clicks is None or n_clicks == 0:
        return "" # Estado inicial: no mostrar nada

    if modelo_forest is None:
        return "Error: El modelo de predicción no se pudo cargar. No se puede realizar la predicción."
    if edad is None or especialidad is None:
        return "Por favor, ingrese todos los valores para la predicción."

    # Obtener día y semana_del_año de la fecha actual para la predicción
    # Tu modelo usa 'día' y 'semana_del_año'
    today = datetime.now()
    dia = today.day
    semana_del_año = today.isocalendar()[1]

    # Codificar la especialidad usando el mapeo que creamos
    especialidad_cod = especialidad_to_cod.get(especialidad)
    if especialidad_cod is None:
        return f"Error: La especialidad '{especialidad}' no está en la lista de especialidades conocidas para la predicción."

    # Crear el DataFrame de entrada para el modelo
    # Las columnas deben coincidir *exactamente* con el entrenamiento:
    # ['ESPECIALIDAD_cod', 'EDAD', 'día', 'semana_del_año']
    input_data = pd.DataFrame([[
        especialidad_cod,
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
        # El modelo predice DIFERENCIA_DIAS (regresión)
        predicted_days = modelo_forest.predict(input_data)[0]
        # Redondear el resultado para una presentación más amigable
        predicted_days_rounded = max(0, round(predicted_days)) # Asegurar que no sea negativo

        return f"El tiempo de espera estimado para esta cita es de **{predicted_days_rounded} días**."
    except Exception as e:
        return f"Error al realizar la predicción: {e}. Asegúrate de que los datos de entrada coincidan con lo que el modelo espera."

# --- 6. Ejecutar el servidor (Para Render, Gunicorn ejecutará 'application') ---
# 'application' es el nombre que Gunicorn buscará para iniciar tu app en Render.
# Necesita ser la instancia de Flask que contiene todas tus Dash apps.
application = server

if __name__ == '__main__':
    # Este bloque solo se ejecuta cuando corres el script localmente (python multi_app.py)
    # No se ejecuta en el entorno de Render cuando Gunicorn lo inicia.
    port = int(os.environ.get("PORT", 8050)) # Render asigna un puerto, localmente usa 8050
    server.run(host='0.0.0.0', port=port, debug=True)
