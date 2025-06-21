import pandas as pd
from flask import Flask, render_template_string
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import requests
import io
import os
from datetime import datetime
import numpy as np
import pickle

hoy = datetime.today()
dia_actual = hoy.day
semana_anio = hoy.isocalendar()[1]

model_download_url = "https://huggingface.co/themasterdrop/simulador_citas_modelo/resolve/main/modelo_forest.pkl?download=true"
modelo_path = "modelo_forest.pkl"

if not os.path.exists(modelo_path):
    print("Descargando modelo desde Hugging Face...")
    try:
        r = requests.get(model_download_url, stream=True)
        r.raise_for_status() # Lanza un error para códigos de estado HTTP erróneos (4xx o 5xx)
        with open(modelo_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Modelo descargado.")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el modelo desde Hugging Face: {e}")
        raise # Re-lanza la excepción para que el despliegue falle

try:
    with open(modelo_path, 'rb') as file:
        modelo_forest = pickle.load(file)
    print("¡Modelo cargado con éxito usando pickle!")
except Exception as e:
    print(f"Error al cargar el modelo con pickle: {e}")
    raise
    

# Cargar los datos (DataFrame)
file_id_data = "1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
url_data = f"https://drive.google.com/uc?export=download&id={file_id_data}"
df = pd.read_csv(url_data)

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

df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias)

# Transformación para Línea de Tiempo
df['DIA_SOLICITACITA'] = pd.to_datetime(df['DIA_SOLICITACITA'], errors='coerce')
df['MES'] = df['DIA_SOLICITACITA'].dt.to_period('M').astype(str)
citas_por_mes = df.groupby('MES').size().reset_index(name='CANTIDAD_CITAS')

# Especialidades para el simulador
especialidades = {17: 'GERIATRIA',
 16: 'GASTROENTEROLOGIA',
 13: 'ENDOCRINOLOGIA',
 51: 'PSIQUIATRIA',
 2: 'CARDIOLOGIA',
 61: 'UROLOGIA',
 50: 'PSICOLOGIA',
 6: 'CIRUGIA GENERAL',
 34: 'NEUROLOGIA',
 20: 'HEMATOLOGIA',
 26: 'MEDICINA INTERNA',
 42: 'OFTALMOLOGIA',
 54: 'REUMATOLOGIA',
 4: 'CIRUGIA PLASTICA Y QUEMADOS',
 33: 'NEUROCIRUGIA',
 48: 'PEDIATRIA GENERAL',
 27: 'NEFROLOGIA',
 35: 'NEUROLOGIA PEDIATRICA',
 40: 'OBSTETRICIA',
 29: 'NEUMOLOGIA',
 43: 'ONCOLOGIA GINECOLOGIA',
 28: 'NEONATOLOGIA',
 21: 'INFECTOLOGIA',
 0: 'ADOLESCENTE',
 18: 'GINECOLOGIA',
 10: 'DERMATOLOGIA',
 8: 'CIRUGIA PEDIATRICA',
 56: 'TRAUMATOLOGIA',
 47: 'PATOLOGIA MAMARIA',
 46: 'OTORRINOLARINGOLOGIA',
 12: 'ECOGRAFIA',
 25: 'MEDICINA FÍSICA Y REHABILITACIÓN',
 31: 'NEUMOLOGIA PEDIATRICA',
 44: 'ONCOLOGIA MEDICA',
 5: 'CIRUGIA CABEZA Y CUELLO',
 7: 'CIRUGIA MAXILO-FACIAL',
 19: 'GINECOLOGIA DE ALTO RIESGO',
 36: 'NEUROPSICOLOGIA',
 52: 'PUERPERIO',
 59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS',
 3: 'CARDIOLOGIA PEDIATRICA',
 41: 'ODONTOLOGIA',
 53: 'RADIOTERAPIA',
 9: 'CIRUGIA TORAXICA',
 37: 'NUTRICION - ENDOCRINOLOGIA',
 57: 'TUBERCULOSIS',
 38: 'NUTRICION - MEDICINA',
 22: 'INFECTOLOGIA PEDIATRICA',
 30: 'NEUMOLOGIA FUNCION RESPONSATORIA',
 39: 'NUTRICION - PEDIATRICA',
 14: 'ENDOCRINOLOGIA PEDIATRICA',
 55: 'SALUD MENTAL ',
 23: 'INFERTILIDAD',
 45: 'ONCOLOGIA QUIRURGICA',
 32: 'NEUMOLOGIA TEST DE CAMINATA',
 49: 'PLANIFICACION FAMILIAR',
 24: 'MEDICINA ALTERNATIVA',
 1: 'ANESTESIOLOGIA',
 11: 'DERMATOLOGIA PEDIATRICA',
 58: 'TUBERCULOSIS PEDIATRICA',
 62: 'ZPRUEBA',
 60: 'URODINAMIA',
 15: 'ENDOCRINOLOGIA TUBERCULOSIS'}

# --- 2. Servidor Flask y Aplicación Dash Principal ---

server = Flask(__name__)
# Inicializa la aplicación Dash principal con use_pages=True
app = dash.Dash(__name__, server=server, use_pages=True, pages_folder="", suppress_callback_exceptions=True)

# Ruta raíz para el menú principal
@server.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Bienvenido</title>
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
                <a href="/simulador/">Simulador de Tiempo de Espera</a>
            </div>
        </div>
    </body>
    </html>
    """)

# El "layout" principal de la aplicación Dash que contendrá el contenido de la página actual
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=dash.page_container) # Aquí se renderizarán las páginas
])


# --- 3. Registro de Páginas Dash con sus Layouts y Callbacks ---

# App 1: Por Rango de Edad
dash.register_page(
    "edad_page",
    path="/edad/",
    name="Distribución por Edad",
    layout=html.Div([
        html.H1("Distribución por Rango de Edad"),
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
        ))
    ])
)

@app.callback( # Las callbacks ahora pertenecen a la única 'app'
    Output('pie-chart-edad', 'figure'),
    Input('histogram-edad', 'clickData'),
    prevent_initial_call=True # Importante para evitar callbacks al cargar la página
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


# App 2: Por Rango de Días de Espera
dash.register_page(
    "espera_page",
    path="/espera/",
    name="Tiempos de Espera",
    layout=html.Div([
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
        ))
    ])
)

@app.callback(
    Output('pie-chart-espera', 'figure'),
    Input('histogram-espera', 'clickData'),
    prevent_initial_call=True
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


# App 3: Por Modalidad de Cita
dash.register_page(
    "modalidad_page",
    path="/modalidad/",
    name="Modalidad de Atención",
    layout=html.Div([
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
        ))
    ])
)

@app.callback(
    Output('bar-especialidad-modalidad', 'figure'),
    Input('pie-modalidad', 'clickData'),
    prevent_initial_call=True
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


# App 4: Por Estado de Seguro
dash.register_page(
    "seguro_page",
    path="/asegurados/",
    name="Estado del Seguro",
    layout=html.Div([
        html.H1("Distribución por Estado del Seguro"),
        dcc.Graph(id='pie-seguro', figure=px.pie(
            df.dropna(),
            names='SEGURO',
            title='Distribución de Pacientes: Asegurados vs No Asegurados',
            template='plotly_white'
        )),
        dcc.Graph(id='bar-espera-seguro', figure=px.bar(
            pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
            x='SEXO',
            y='DIFERENCIA_DIAS',
            title="Seleccione una modalidad en el gráfico de pastel"
        ))
    ])
)

@app.callback(
    Output('bar-espera-seguro', 'figure'),
    Input('pie-seguro', 'clickData'),
    prevent_initial_call=True
)
def update_bar_seguro(clickData):
    if clickData is None or 'label' not in clickData['points'][0]:
        return px.bar(
            pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
            x='SEXO',
            y='DIFERENCIA_DIAS',
            title="Seleccione una modalidad en el gráfico de pastel"
        )

    seguro = clickData['points'][0]['label']
    filtered_df = df[df['SEGURO'] == seguro]

    if filtered_df.empty:
        return px.bar(
            pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
            x='SEXO',
            y='DIFERENCIA_DIAS',
            title=f"No hay datos para {seguro}"
        )

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

    y_min = mean_wait['DIFERENCIA_DIAS'].min() - 1
    y_max = mean_wait['DIFERENCIA_DIAS'].max() + 1
    fig.update_yaxes(range=[y_min, y_max])

    return fig


# App 5: Línea de Tiempo
dash.register_page(
    "tiempo_page",
    path="/tiempo/",
    name="Línea de Tiempo",
    layout=html.Div([
        html.H1("Citas Agendadas por Mes"),
        dcc.Graph(
            id='grafico-lineal',
            figure=px.line(citas_por_mes, x='MES', y='CANTIDAD_CITAS', markers=True,
                             title='Cantidad de Citas por Mes')
        ),
        html.Div([
            dcc.Graph(id='grafico-pie-especialidades'),
            dcc.Graph(id='grafico-pie-atencion')
        ])
    ])
)

@app.callback(
    [Output('grafico-pie-especialidades', 'figure'),
     Output('grafico-pie-atencion', 'figure')],
    [Input('grafico-lineal', 'clickData')],
    prevent_initial_call=True
)
def actualizar_graficos(clickData):
    if clickData is None:
        return px.pie(names=[], values=[], title="Seleccione un mes"), px.pie(names=[], values=[], title="Seleccione un mes")

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


# App 6: Simulador de Tiempo de Espera
dash.register_page(
    "simulador_page",
    path="/simulador/",
    name="Simulador de Tiempo de Espera",
    layout=html.Div([
        html.H2("Simulador de Tiempo de Espera de Citas"),
        html.Label("Especialidad:"),
        dcc.Dropdown(
            id='input-especialidad',
            options=[{'label': k, 'value': v} for k, v in especialidades.items()],
            value=1,
            placeholder="Selecciona una especialidad"
        ),
        html.Label("Edad:"),
        dcc.Input(id='input-edad', type='number', value=30),
        html.Label("Día:"),
        dcc.Input(id='input-dia', type='number', value=dia_actual),
        html.Label("Semana del año:"),
        dcc.Input(id='input-semana_anio', type='number', value=semana_anio),
        html.Br(),
        html.Button("Predecir", id='btn-predecir', n_clicks=0),
        html.Div(id='output-prediccion')
    ])
)

@app.callback(
    Output('output-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    Input('input-especialidad', 'value'),
    Input('input-edad', 'value'),
    Input('input-dia', 'value'),
    Input('input-semana_anio', 'value'),
    prevent_initial_call=True
)
def predecir(n_clicks, especialidad, edad, dia, semana_anio):
    if n_clicks > 0:
        if edad is None or edad < 0 or edad > 120:
            return "Edad no válida."
        entrada = [[
            especialidad, edad, dia, semana_anio
        ]]
        prediccion = modelo_forest.predict(entrada)[0]
        nombre_especialidad = especialidades.get(especialidad, "Desconocida")
        return f"Especialidad: {nombre_especialidad} — Tiempo estimado de espera: {prediccion:.2f} días"
    return ""


# --- 4. Ejecución de la Aplicación ---

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port, debug=True) # Para desarrollo local

# Para Render, Gunicorn ejecutará app.server
application = app.server
