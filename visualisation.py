import glob
import pandas as pd

# url = 'https://storage.cloud.google.com/ecoplant_gstat/malfunction_compressor000000000000.csv'
# uri = 'gs://ecoplant_gstat/malfunction_compressor000000000000.csv'
# df = pd.read_csv(uri)


def open_from_bucket():
    data = pd.concat(map(pd.read_csv, glob.glob('data/bucket/*.csv')))
    data['ts'] = pd.to_datetime(data['ts'], format='%Y-%m-%d %H:%M:%S %Z')
    data.sort_values(['device_id', 'ts'], inplace=True)
    data.to_csv('data/data.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
    # data.to_csv('data2.csv', index=False)
    # data.sample(5000).to_csv('sample1.csv', index=False)
    # data.sample(5000).to_csv('sample2.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
    return data

def visu(id):
    df_tmp = df.reset_index().query('device_id==@id')
    df_tmp['shutdown_mal_code'] = df_tmp['shutdown_mal_code'].astype('str')
    fig = px.scatter(df_tmp,
                     x='ts',
                     y='compressor_status_des',
                     color='shutdown_mal_code',
                     #                      size='size_shutdown_mal'
                     #                      hover_name = 'dict_compressor_status_des'
                     )
    fig.write_html(f"output/scater_{id}_tmp.html")
