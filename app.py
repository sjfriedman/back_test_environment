import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime, timedelta

import pandas as pd

import apply_historical as ah

app = dash.Dash(__name__)

#layout
app.layout = html.Div([
    html.Div([
        dcc.Input(id='symbol', type='text', placeholder='Symbol'),
        dcc.Input(id='buy_query', type='text', placeholder='Buy Query'),
        dcc.Input(id='sell_query', type='text', placeholder='Sell Query'),
        dcc.Input(id='stop_loss', type='text', placeholder='Stop Loss(Null if None)'),
        dcc.Input(id='start_date', type='text', placeholder='Start Date (Ex. 2020-01)'),
        dcc.Input(id='end_date', type='text', placeholder='End Date (Ex. 2020-12)'),
        html.Button(id='trade_button', n_clicks=0, children='Long'),
        html.Button(id='close_at_end_day_button', n_clicks=0, children='Close At End Day'),
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
    dcc.Store(id='close_at_end_day_bool', storage_type='memory'),
])

#get trade data
@app.callback(
    Output('trade_data', 'data'),
    Input('run_button', 'n_clicks'),
    Input('close_at_end_day_bool', 'data'),
    State('symbol', 'value'),
    State('buy_query', 'value'),
    State('sell_query', 'value'),
    State('stop_loss', 'value'),
    State('start_date', 'value'),
    State('end_date', 'value'),
)
def update_trade_data(run_click, close_at_end_day_bool, symbol, buy_query, sell_query, stop_loss, start_date, end_date):
    #if run, run queries
    if run_click:
        df = ah.execute(symbol.upper(), buy_query, sell_query, start_date, end_date, close_at_end_day_bool, stop_loss)
        print(df)
        return df.to_json()
    
    return None

#toggle trade button between long and short
@app.callback(
    [Output('trade_button', 'children'), Output('trade_sign', 'data')],
    Input('trade_button', 'n_clicks'),
)
def toggle_trade_button(n_clicks):
    #toggle button state between Long and Short
    if n_clicks % 2 == 0:
        return 'Long', 1
    else:
        return 'Short', -1

#toggle close at end of day
@app.callback(
    [Output('close_at_end_day_button', 'children'), Output('close_at_end_day_bool', 'data')],
    Input('close_at_end_day_button', 'n_clicks'),
)
def toggle_close_at_end_of_day_button(n_clicks):
    #toggle button state between close at end of day and not close
    if n_clicks % 2 == 0:
        return 'Close At End Day', True
    else:
        return 'Don\'t Close At End Day', False

#date range update
@app.callback(
    [Output('start_date_table', 'data'), Output('end_date_table', 'data')],
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_date_range(start_date, end_date):
    #updates corresponding vars
    return start_date, end_date

#update graph
@app.callback(
    Output('output-graph', 'figure'),
    Input('trade_data', 'data'),
    Input('trade_sign', 'data'),
    Input('start_date_table', 'data'),
    Input('end_date_table', 'data'),
    Input('run_button', 'n_clicks'),
)
def update_graph(trade_data, trade_sign, start_date, end_date, run_clicks):
    #if click on something
    if run_clicks is not None and trade_data is not None and trade_sign is not None:
        #read data and multiply by trade_sign(long or short)
        df = pd.read_json(trade_data)
        df['pct_change'] = df['pct_change'] * trade_sign
        
        #for graph show range
        if start_date is not None and end_date is not None:
            df = df[(df['close_time'] >= start_date) & (df['close_time'] <= end_date)]
        
        #graph figure
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
    #if nada
    return {'data': [], 'layout': {}}

if __name__ == '__main__':
    app.run_server(debug=True)
