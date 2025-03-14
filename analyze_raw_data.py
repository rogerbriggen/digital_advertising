import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

# Define the feature columns
feature_columns = [
    "competitiveness", "difficulty_score", "organic_rank", "organic_clicks", 
    "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", 
    "ad_roas", "conversion_rate", "cost_per_click"
]

def read_and_organize_csv(file_path):
    """Reads and organizes the CSV data"""
    df = pd.read_csv(file_path)
    if 'step' in df.columns:
        df = df.drop(columns=['step'])
    return df

def calculate_percentage_changes(df):
    """Calculate percentage changes for each keyword and feature"""
    keywords = df['keyword'].unique()
    result = pd.DataFrame()
    for keyword in keywords:
        keyword_df = df[df['keyword'] == keyword].reset_index(drop=True)
        for feature in feature_columns:
            keyword_df[f'{feature}_pct_change'] = keyword_df[feature].pct_change() * 100
        result = pd.concat([result, keyword_df])
    return result

def create_visualization_app(df):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Use index_string only for CSS and head elements; remove overlay Divs from here.
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .fullscreen-overlay {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 100vh;
                    background-color: white;
                    z-index: 9999;
                    overflow: hidden;
                    box-sizing: border-box;
                }
                .fullscreen-overlay.active {
                    display: block;
                }
                .overlay-header {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-bottom: 1px solid #dee2e6;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .overlay-body {
                    height: calc(100vh - 50px);
                    overflow: hidden;
                }
                .overlay-body .fullscreen-graph {
                    width: 100%;
                    height: 100%;
                }
                .user-select-none {
                    padding: 10px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Get unique keywords
    keywords = df['keyword'].unique()
    
    # Add overlay Divs inside the layout so they exist in the component tree.
    overlay_divs = html.Div([
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Time Series Data (Fullscreen)"),
                        html.Button("Close", id="close-fullscreen-ts", className="btn btn-outline-secondary btn-sm")
                    ],
                    className="overlay-header"
                ),
                html.Div(
                    html.Div(id="fullscreen-ts-graph-container", className="fullscreen-graph"),
                    className="overlay-body"
                )
            ],
            id="fullscreen-ts-overlay",
            className="fullscreen-overlay"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Percentage Change (Fullscreen)"),
                        html.Button("Close", id="close-fullscreen-pct", className="btn btn-outline-secondary btn-sm")
                    ],
                    className="overlay-header"
                ),
                html.Div(
                    html.Div(id="fullscreen-pct-graph-container", className="fullscreen-graph"),
                    className="overlay-body"
                )
            ],
            id="fullscreen-pct-overlay",
            className="fullscreen-overlay"
        )
    ])
    
    # Main layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Digital Advertising Data Visualization"),
                html.P("Select features and keywords to visualize")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Select Features:"),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": col, "value": col} for col in feature_columns],
                    value=feature_columns[0],
                    multi=False
                )
            ], width=6),
            dbc.Col([
                html.Label("Select Keywords:"),
                dcc.Dropdown(
                    id="keyword-dropdown",
                    options=[{"label": k, "value": k} for k in keywords],
                    value=list(keywords[:5]),
                    multi=True
                )
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("Time Series Data", style={"display": "inline-block"}),
                    html.Button("Fullscreen", id="fullscreen-button-ts", 
                                className="btn btn-outline-secondary btn-sm ms-2",
                                style={"marginLeft": "10px", "verticalAlign": "middle"})
                ]),
                dcc.Graph(id="time-series-graph", style={"height": "600px"})
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("Percentage Change", style={"display": "inline-block"}),
                    html.Button("Fullscreen", id="fullscreen-button-pct", 
                                className="btn btn-outline-secondary btn-sm ms-2",
                                style={"marginLeft": "10px", "verticalAlign": "middle"})
                ]),
                dcc.Graph(id="pct-change-graph", style={"height": "600px"})
            ], width=12)
        ]),
        overlay_divs  # Append overlay Divs to layout so they are part of the component tree.
    ], fluid=True)
    
    @app.callback(
        [Output("time-series-graph", "figure"),
         Output("pct-change-graph", "figure")],
        [Input("feature-dropdown", "value"),
         Input("keyword-dropdown", "value")]
    )
    def update_graphs(selected_feature, selected_keywords):
        if not selected_keywords:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No keywords selected")
            return empty_fig, empty_fig
        filtered_df = df[df["keyword"].isin(selected_keywords)]
        colors = px.colors.qualitative.Bold
        color_mapping = {keyword: colors[i % len(colors)] for i, keyword in enumerate(selected_keywords)}
        
        fig_ts = go.Figure()
        for keyword in selected_keywords:
            kdf = filtered_df[filtered_df["keyword"] == keyword]
            fig_ts.add_trace(go.Scatter(
                x=kdf.index,
                y=kdf[selected_feature],
                name=keyword,
                line={"color": color_mapping[keyword], "width": 2},
                mode="lines+markers",
                marker={"size": 8}
            ))
        fig_ts.update_layout(
            title=f"Time Series: {selected_feature}",
            xaxis_title="Time Step",
            yaxis_title=selected_feature,
            plot_bgcolor="rgba(240,240,240,0.9)",
            legend={"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "black", "borderwidth": 1}
        )
        
        pct_change_col = f"{selected_feature}_pct_change"
        fig_pct = go.Figure()
        for keyword in selected_keywords:
            kdf = filtered_df[filtered_df["keyword"] == keyword]
            fig_pct.add_trace(go.Scatter(
                x=kdf.index,
                y=kdf[pct_change_col],
                name=keyword,
                line={"color": color_mapping[keyword], "width": 2},
                mode="lines+markers",
                marker={"size": 8}
            ))
        if len(filtered_df) > 0:
            fig_pct.add_shape(
                type="line",
                x0=filtered_df.index.min(),
                y0=0,
                x1=filtered_df.index.max(),
                y1=0,
                line={"color": "gray", "width": 1, "dash": "dash"}
            )
        fig_pct.update_layout(
            title=f"Percentage Change: {selected_feature}",
            xaxis_title="Time Step",
            yaxis_title="Percentage Change (%)",
            plot_bgcolor="rgba(240,240,240,0.9)",
            legend={"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "black", "borderwidth": 1}
        )
        return fig_ts, fig_pct

    # Callback to toggle Time Series overlay
    @app.callback(
        Output("fullscreen-ts-overlay", "className"),
        [Input("fullscreen-button-ts", "n_clicks"),
         Input("close-fullscreen-ts", "n_clicks")],
        [State("fullscreen-ts-overlay", "className")]
    )
    def toggle_ts_overlay(n_open, n_close, current_class):
        ctx = callback_context
        if not ctx.triggered:
            return current_class or "fullscreen-overlay"
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "fullscreen-button-ts":
            return "fullscreen-overlay active"
        elif trigger == "close-fullscreen-ts":
            return "fullscreen-overlay"
        return current_class

    # Callback to render TS graph in overlay
    @app.callback(
        Output("fullscreen-ts-graph-container", "children"),
        [Input("fullscreen-ts-overlay", "className")],
        [State("time-series-graph", "figure")]
    )
    def render_ts_overlay(className, ts_fig):
        if "active" in className and ts_fig:
            return dcc.Graph(figure=ts_fig, className="fullscreen-graph")
        return no_update

    # Callback to toggle Percentage Change overlay
    @app.callback(
        Output("fullscreen-pct-overlay", "className"),
        [Input("fullscreen-button-pct", "n_clicks"),
         Input("close-fullscreen-pct", "n_clicks")],
        [State("fullscreen-pct-overlay", "className")]
    )
    def toggle_pct_overlay(n_open, n_close, current_class):
        ctx = callback_context
        if not ctx.triggered:
            return current_class or "fullscreen-overlay"
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "fullscreen-button-pct":
            return "fullscreen-overlay active"
        elif trigger == "close-fullscreen-pct":
            return "fullscreen-overlay"
        return current_class

    # Callback to render Percentage Change graph in overlay
    @app.callback(
        Output("fullscreen-pct-graph-container", "children"),
        [Input("fullscreen-pct-overlay", "className")],
        [State("pct-change-graph", "figure")]
    )
    def render_pct_overlay(className, pct_fig):
        if "active" in className and pct_fig:
            return dcc.Graph(figure=pct_fig, className="fullscreen-graph")
        return no_update

    return app

def main():
    print("Loading dataset...")
    dataset_path = "data/organized_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} not found!")
        return
    df = read_and_organize_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print("Calculating percentage changes...")
    df_with_pct_change = calculate_percentage_changes(df)
    print("Creating visualization app...")
    app = create_visualization_app(df_with_pct_change)
    print("Running app - visit http://127.0.0.1:8050/ in your web browser")
    app.run_server(debug=True)

if __name__ == '__main__':
    main()