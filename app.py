import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime, timedelta
import pandas as pd
import apply_historical as ah

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.Div([
        dcc.Input(id='symbol', type='text', placeholder='Symbol'),
        dcc.Input(id='buy_query', type='text', placeholder='Buy Query'),
        dcc.Input(id='sell_query', type='text', placeholder='Sell Query'),
        dcc.Input(id='start_date', type='text', placeholder='Start Date (Ex. 2020-01)'),
        dcc.Input(id='end_date', type='text', placeholder='End Date (Ex. 2020-12)'),
        html.Button(id='trade_button', n_clicks=0, children='Long'),
        html.Button(id='run_button', n_clicks=0, children='Run'),
    ]),
    dcc.Graph(id='output-graph'),
    dcc.DatePickerRange(
        id='date-range-picker',
        display_format='YYYY-MM-DD',
        start_date='2020-01-01',
        end_date='2020-12-31',
    ),
    dcc.Store(id='trade_sign', storage_type='memory'),
    dcc.Store(id='trade_data', storage_type='memory'),
    dcc.Store(id='start_date_table', storage_type='memory'),
    dcc.Store(id='end_date_table', storage_type='memory'),
])

stock_data = pd.read_pickle('alphavantage.pkl')

@app.callback(
    Output('trade_data', 'data'),
    Input('run_button', 'n_clicks'),
    State('symbol', 'value'),
    State('buy_query', 'value'),
    State('sell_query', 'value'),
    State('start_date', 'value'),
    State('end_date', 'value'),
)
def update_trade_data(run_click, symbol, buy_query, sell_query, start_date, end_date):
    if run_click:
        df = ah.execute(symbol, buy_query, sell_query, start_date, end_date)
        return df.to_json()
    
    return None

# Define a callback to handle "Trade" button clicks and update button text
@app.callback(
    [Output('trade_button', 'children'), Output('trade_sign', 'data')],
    Input('trade_button', 'n_clicks'),
)
def toggle_trade_button(n_clicks):
    # Toggle button state between Long and Short
    if n_clicks % 2 == 0:
        return 'Long', 1
    else:
        return 'Short', -1

# Create a callback to update start_date and end_date when the date range picker changes
@app.callback(
    [Output('start_date_table', 'data'), Output('end_date_table', 'data')],
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_date_range(start_date, end_date):
    return start_date, end_date

# Create a callback to update the graph based on trade_data, trade_sign, and date range
@app.callback(
    Output('output-graph', 'figure'),
    Input('trade_data', 'data'),
    Input('trade_sign', 'data'),
    Input('start_date_table', 'data'),
    Input('end_date_table', 'data'),
    Input('run_button', 'n_clicks'),
)
def update_graph(trade_data, trade_sign, start_date, end_date, run_clicks):
    if run_clicks is not None and trade_data is not None and trade_sign is not None:
        df = pd.read_json(trade_data)
        df['pct_change'] = df['pct_change'] * trade_sign
        
        if start_date is not None and end_date is not None:
            df['close_time'] = pd.to_datetime(df['close_time'])
            df = df[(df['close_time'] >= start_date) & (df['close_time'] <= end_date)]
        
        figure = {
            'data': [
                {'x': df['close_time'], 'y': df['pct_change'], 'type': 'line', 'name': 'Performance',
                    'connectgaps': True},
            ],
            'layout': {
                'xaxis': {'title': 'time'},
                'yaxis': {'title': 'Profit'},
            }
        }
        return figure
    else:
        return {'data': [], 'layout': {}}

if __name__ == '__main__':
    app.run_server(debug=True)
