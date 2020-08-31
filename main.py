# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import pandas as pd


def transform_raw_data(data):
    data = pd.read_csv(
        'data/malfunction_compressor_elishar000000000000.csv'
        #                    parse_dates=['ts'],
        #                     date_parser= lambda x:  pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S.%f %Z')

    )
    data.drop(['device_name', 'indicator_id', 'metric', 'indicator_name'], axis=1, inplace=True)

    data['ts'] = pd.to_datetime(data['ts'], format='%Y-%m-%d %H:%M:%S.%f %Z', errors='coerce')
    data['ts'] = pd.to_datetime(data['ts'].dt.strftime('%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S',
                                errors='coerce')

    len_original_df = len(data)
    print(f'len before {len_original_df}')
    data.loc[data['value'] > 4, 'value'] = np.nan
    data.dropna(how='any', inplace=True)
    print(f'len only value < 4 {len(data) / len_original_df}')
    data = data.groupby(['ts', 'device_id'], as_index=False)['value'].min()
    print(f'len min only {len(data) / len_original_df}')
    return data


def transform_data(data):
    df = data.copy()

    df.set_index('ts', inplace=True)
    df.loc[~df['compressor_status'].isin([0, 0.5, 1, 2, 3, 4]), 'compressor_status'] = np.nan
    df['compressor_status'].fillna(method='ffill', inplace=True)
    df['compressor_status'].fillna(method='bfill', inplace=True)
    df['compressor_status'].fillna(1, inplace=True)
    df.drop(['RN'], axis=1, inplace=True)
    df.drop(['ind_shutdown_mal_24H_next', 'ind_shutdown_mal_3D_next', 'ind_shutdown_mal_7D_next'], axis=1, inplace=True)
    print(len(set(df['device_id'].values)))

    device_filiere_list = df.groupby('device_id')['ind_shutdown_mal'].mean()
    device_filiere_list = device_filiere_list[device_filiere_list > 0].sort_values(ascending=True)
    devices_with_faileures = device_filiere_list
    df_filtered = df[df['device_id'].isin(devices_with_faileures.index)].copy()

    print(len(set(df_filtered['device_id'].values)))
    print(device_filiere_list)
    df_hourly = df_filtered.groupby('device_id').resample('30min').agg(
        {'compressor_status': 'mean', 'ind_shutdown_mal': 'max'})

    return df_filtered, df_hourly, devices_with_faileures


def code_des(df, lim):
    status_description_dict = {
        0: 'off',  ## on
        1: 'energy_consuming',
        2: 'compressing'
    }
    df['compressor_status_des'] = df['compressor_status_code'].map(status_description_dict)
    info = df.reset_index().groupby('device_id').agg(
        {'ts': ['min', 'max', 'count'], 'was_na_mean': 'mean', 'shutdown_mal_code': 'mean'})
    info[('', 'high malfunctions')] = 0
    info.loc[info[('shutdown_mal_code', 'mean')] > lim, ('', 'high malfunctions')] = 1

    high_malfunctions = info.loc[info[('', 'high malfunctions')] == 1, :].index
    df = df.loc[~df.index.get_level_values(0).isin(high_malfunctions), :]
    return df, info


def resample_period(df, period):
    pp = df.pivot_table(index=['device_id', 'ts'], columns='compressor_status_des', values='compressor_status_code',
                        aggfunc='count', fill_value=0, observed=True).reset_index().set_index('ts').groupby(
        'device_id').resample(period).sum().drop('device_id', axis=1).join(df['shutdown_mal_code'], how='left').fillna(
        0)
    stamps = (pp.index.droplevel(0) - pd.Timestamp("1970-01-01")) / np.timedelta64(1, period)  # // pd.Timedelta('H')
    pp['stamps'] = stamps - stamps.min()
    pp['shutdown_mal_code'].clip(upper=1, inplace=True)
    return pp


def shorten_label(df1):
    df_tmp = df1.copy()
    tmp = df_tmp[['shutdown_mal_code']].query('shutdown_mal_code==1').reset_index()
    df_tmp.drop('shutdown_mal_code', axis=1, inplace=True)
    first_ocurence = tmp['ts'].diff().astype('timedelta64[D]').fillna(0)
    tmp.loc[first_ocurence == 0, 'shutdown_mal_code'] = 2
    print(tmp.loc[first_ocurence > 0, ['device_id', 'ts']])
    tmp = tmp.set_index(['device_id', 'ts'])
    df_tmp = df_tmp.join(tmp, how='left')
    df_tmp.fillna(0, inplace=True)
    return df_tmp


def minimum_timespan(df, period):
    tmp = df.query('shutdown_mal_code==1').copy()
    minimal_time_span = tmp.index.to_frame()['ts'] + pd.Timedelta(hours=period)
    minimal_time_span = minimal_time_span.rename('ts_end').to_frame().reset_index()
    minimal_time_span['ts'] = minimal_time_span['ts'] + pd.Timedelta(minutes=1)

    tmp = minimal_time_span.apply(lambda x: pd.date_range(start=x['ts'], end=x['ts_end'], freq='H'), axis=1)
    tmp.index = minimal_time_span['device_id'].values  #
    period_to_na = tmp.explode().to_frame().reset_index()
    period_to_na.columns = ['device_id', 'ts']
    period_to_na = period_to_na.set_index(['device_id', 'ts'])
    period_to_na['shutdown_mal_code'] = 2
    df.update(period_to_na)


def series_to_supervised(data, n_in=1, n_out=1, dropna=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j + 1}(t-{i})' for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j + 1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j + 1}(t+{i})' for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropna:
        agg.dropna(inplace=True)
    return agg


# check version
import tensorflow as tf

print(tf.__version__)

# example of a model defined with the sequential api
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
