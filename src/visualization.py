import os
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input

from helpers import get_datasets, load_time_series

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Cutting Force Data Visualization'),
    dcc.Dropdown(
        options=[{"label": f.split("/")[-1], "value": f} for f in get_datasets()],
        value=get_datasets()[0],
        id="dataset-dropdown",
    ),
    dcc.Graph(id="cutting-force-in-time-domain"),
    dcc.Graph(id="Fx-in-time-domain"),
    dcc.Graph(id="Fy-in-time-domain"),
    dcc.Graph(id="Fz-in-time-domain")
])

@app.callback(
    Output("cutting-force-in-time-domain", "figure"),
    Output("Fx-in-time-domain", "figure"),
    Output("Fy-in-time-domain", "figure"),
    Output("Fz-in-time-domain", "figure"),
    Input("dataset-dropdown", "value")
)
def update_graph(selected_dataset):
    expanded = os.path.expanduser(selected_dataset)
    df = load_time_series(expanded)

    time_column = df["Time"]

    fig_total = go.Figure()
    color_map = {"Fx": "blue", "Fy": "green", "Fz": "red"}

    for fcol in ["Fx", "Fy", "Fz"]:
        fig_total.add_trace(go.Scatter(x=time_column, y=df[fcol], mode="lines", name=fcol, line=dict(color=color_map[fcol])))
    fig_total.update_layout(title="Cutting Forces Over Time", xaxis_title="Time", yaxis_title="Force")

    # Individual force plots
    fig_fx = px.line(df, x="Time", y="Fx", title="Fx over Time")
    fig_fx.update_traces(line_color="blue")

    fig_fy = px.line(df, x="Time", y="Fy", title="Fy over Time")
    fig_fy.update_traces(line_color="green")

    fig_fz = px.line(df, x="Time", y="Fz", title="Fz over Time")
    fig_fz.update_traces(line_color="red")

    return fig_total, fig_fx, fig_fy, fig_fz

if __name__ == "__main__":
    app.run(debug=True)