import GenerateAndSerialise as gs
import LoadAndQuery as lq
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import matplotlib.pyplot as plt

min_bars_df = gs.pd.read_csv("30_min_bars.csv")
min_bars_df = min_bars_df.rename(columns={'ldn_time': 'time'})
#no missing values
#print(min_bars_df.isnull().sum())
min_bars_df['time'] = min_bars_df['time'].str.split('+').str[0]
#note the above split is because of the time zone change
#We don't care if it's +01:00 or +00:00 so we remove it
min_bars_df['time'] = gs.pd.to_datetime(min_bars_df['time'])


def check_dict_values(date_spcfic_dict):
    """
    :param date_spcfic_dict: used to filter the min_bars_df using keys which are
    the input dates; values which give the corresponding security(s)
    :return: dict of {dates: securities} for which data is available
    """
    unique_specifics = min_bars_df['specific'].unique()
    unique_specifics_set = set(unique_specifics)
    for value in date_spcfic_dict.values():
        if value not in unique_specifics_set:
            date_spcfic_dict = None
            break
    return date_spcfic_dict

def create_specific_dict(security, date_range_str):
    """
    :param security: the product in specific form
    :param date_range_str: range of dates over which we can potentially assign
    the security
    :return: dict of {dates: specific} where specific is available for that
    date, otherwise None value
    """
    start_date_str, end_date_str = date_range_str.split('-')
    start_date = gs.pd.to_datetime(start_date_str)
    end_date = gs.pd.to_datetime(end_date_str)
    num_days = (end_date - start_date).days
    date_range = [start_date + lq.timedelta(days=x) for x in range(num_days + 1)]
    cutoff_date = gs.pd.to_datetime('2025-01-20')
    date_specific_dict = {date.strftime('%Y-%m-%d'): security if date
        <= cutoff_date else None for date in date_range}
    return date_specific_dict


def get_filtered_prices(security, security_type, date_range_str):
    """
    :param security: the security we filter for, of any of the 3 forms
    :param security_type: the form of the security e.g. specific/G/MG
    :param date_range_str: filter for days we want on which we do intraday plots
    :return: the filtered dataset, the input security and the date on which
    historical calculations are produced
    """
    filtered_data = gs.pd.DataFrame()
    if security_type == 'generic':
        date_specific_dict, _ =\
            lq.querying_interface(security, security_type, date_range_str)
        date_specific_dict = check_dict_values(date_specific_dict)
    elif security_type == 'monthly generic':
        date_specific_dict, _ =\
            lq.querying_interface(security, security_type, date_range_str)
        date_specific_dict = check_dict_values(date_specific_dict)
    elif security_type == 'specific':
        date_specific_dict = create_specific_dict(security, date_range_str)
        date_specific_dict = check_dict_values(date_specific_dict)
    else:
        date_specific_dict = None

    for date_str, specific_code in date_specific_dict.items():
        if specific_code is not None:
            date = gs.pd.to_datetime(
                date_str)

            daily_data = min_bars_df[
                (min_bars_df['time'].dt.date == date.date()) &
                (min_bars_df['specific'] == specific_code)]

            filtered_data = gs.pd.concat([filtered_data, daily_data])
    filtered_data.sort_values(by='time', inplace=True)
    first_input_date = date_range_str[:10]
    return filtered_data, security, first_input_date


def calculate_intraday_value(filtered_data):
    """
    :param filtered_data: data for the specified product and date range
    :return: dataframe of rows being times, columns being dates in the date
    range, contents being cumulative price change (value)
    """
    grouped = filtered_data.groupby(filtered_data['time'].dt.date)
    full_day_changes = gs.pd.DataFrame()

    for date, group in grouped:
        intraday_changes = group['OPEN'].diff().iloc[1:] #intraday
        times = group['time'].dt.strftime(f'%H:%M:00').iloc[1:]

        #end_of_day
        daily_open = group.iloc[-1]['OPEN']
        daily_close = group.iloc[-1]['CLOSE']
        end_of_day_change = daily_close - daily_open

        all_changes = gs.pd.concat(
            [intraday_changes, gs.pd.Series(end_of_day_change)], ignore_index=True)

        end_of_day_time = gs.pd.Series(
            [group['time'].iloc[-1].strftime(f'19:30:00')])
        all_times = gs.pd.concat([times, end_of_day_time], ignore_index=True)

        #append to the full day changes DataFrame
        full_day_changes[date] = gs.pd.Series(all_changes.values, index=all_times)
    cumulative_price_changes = full_day_changes.cumsum()
    return cumulative_price_changes


def get_volumes_df(grouped_df):
    """
    :param grouped_df: filtered dataset grouped by time
    :return: dataframe of cumulative intraday volumes with rows as times,
    columns as single dates
    """
    day_volumes_intra = gs.pd.DataFrame()
    for date, group in grouped_df:
        intraday_cum_volumes = group['VOLUME'].cumsum()
        times = group['time'].dt.strftime(f'%H:%M:00')
        day_volumes_intra[date] = gs.pd.Series(
            intraday_cum_volumes.values, index=times)
    return day_volumes_intra


def intraday_vs_hist_avg_volumes(
        filtered_data, security, security_type, last_hist_date):
    """
    :param filtered_data: filtered data set for the security in the time range
    :param security: security of any form which we will get volumes for
    :param security_type: form of security
    :param last_hist_date: date used for historical calculation
    :return: intraday cumulative volumes df with historical intraday cumulative
    volumes to compare to
    """
    grouped_intra = filtered_data.groupby(filtered_data['time'].dt.date)
    prev_OHLCdf, _, _ = get_filtered_prices(
        security, security_type, '2024/08/01'+'-'+last_hist_date)
    prev_OHLCdf['time'] = prev_OHLCdf['time'].astype(str)
    prev_OHLCdf['date'] = prev_OHLCdf['time'].str[
                          :10]
    prev_OHLCdf['time'] = prev_OHLCdf['time'].str[11:]
    prev_OHLCdf['CUMVOLUME'] = prev_OHLCdf.groupby('date')['VOLUME'].transform(
        'cumsum')
    prev_OHLCdf = prev_OHLCdf.drop(columns=['VOLUME'])
    time_avg_cumvolume = prev_OHLCdf.groupby('time')['CUMVOLUME'].mean()
    average_cumvolume_df = time_avg_cumvolume.reset_index()
    average_cumvolume_df.columns = ['time', 'Average Cumulative Volume']
    average_cumvolume_df.set_index('time', inplace=True)
    full_day_volumes = get_volumes_df(grouped_intra)
    full_day_volumes['hist average'] = average_cumvolume_df['Average Cumulative Volume']
    return full_day_volumes

contractOHLC_df, mon_gen_code, last_hist = get_filtered_prices(
    'COMAR1', 'monthly generic', '2024/10/30-2024/11/01')
intraday_volume_df = intraday_vs_hist_avg_volumes(
    contractOHLC_df, mon_gen_code, 'monthly generic', last_hist)

def plot_intraday_cumvolumes(intraday_volumes_df, security):
    """
    Plots the intraday volumes dataframe columns along with cumulative
    historical avg.
    Done in matplotlib due to time constraint.
    """
    plt.figure(figsize=(14, 7))

    for column in intraday_volumes_df.columns:
        plt.plot(intraday_volumes_df.index, intraday_volumes_df[column],
                 label=str(column))

    plt.title(f'Intraday Value Changes for {security}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Volumes')

    specific_times = ["09:00:00", "12:00:00", "15:00:00", "18:00:00"]
    x_labels = [label if any(time in label for time in specific_times) else ""
                for label in intraday_volumes_df.index]
    plt.xticks(intraday_volumes_df.index, x_labels)
    plt.legend(title='date', loc='upper right')
    plt.grid(True)
    plt.show()

#uncomment to plot
#plot_intraday_cumvolumes(intraday_volume_df, 'COMAR1')


# Define the Dash application
app = dash.Dash(__name__)

# Application layout
app.layout = html.Div([
    html.Div([
        dcc.Input(id='security-input', type='text', value='COMAR1',
                  placeholder='Enter Security Code'),
        dcc.Dropdown(
            id='security-type-dropdown',
            options=[
                {'label': 'Generic', 'value': 'generic'},
                {'label': 'Monthly Generic', 'value': 'monthly generic'},
                {'label': 'Specific', 'value': 'specific'}
            ],
            value='monthly generic'
        ),
        dcc.Input(id='date-range-input', type='text',
                  value='2025/01/14-2025/01/20',
                  placeholder='Enter Date Range'),
        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    dcc.Graph(id='intraday-changes-graph')
])


# Callback to update graph based on user inputs
@app.callback(
    Output('intraday-changes-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    [State('security-input', 'value'),
     State('security-type-dropdown', 'value'),
     State('date-range-input', 'value')]
)
def update_graph(n_clicks, security, security_type, date_range):
    """
    Used to update the Dash app with new inputs
    """
    if n_clicks > 0:
        if not (security and security_type and date_range):
            return px.scatter(title="Please enter all required fields and click Submit.")

        filtered_data, _, _ = get_filtered_prices(security, security_type, date_range)
        if filtered_data.empty:
            return px.scatter(title="No data available for the given inputs.")

        cumulative_price_changes = calculate_intraday_value(filtered_data)
        fig = px.line(cumulative_price_changes,
                      title=f'Intraday Value Changes for {security}')
        fig.update_layout(xaxis_title='Time of Day',
                          yaxis_title='Value',
                          plot_bgcolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
        return fig

    return px.scatter(title="Please enter all fields and click Submit to see the data.")


if __name__ == '__main__':
    app.run_server(debug=True)
