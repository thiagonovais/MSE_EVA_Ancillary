import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as datetime
import sys
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
register_matplotlib_converters()


def create_csv():
        xls = './Netzfrequenz_Sekundenwerte_2012_KW37.xlsx'
        tab_with_data = 'Netzfrequenz'

        df_data = pd.read_excel(xls, tab_with_data,encoding=sys.getfilesystemencoding())
        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=False)


def read_csv():
        df_data = pd.read_csv('./Data_in_CSV.csv', sep=';')
        # df_data.drop(columns={'Unnamed: 0'}, inplace= True)
        df_data['Date and Time'] = pd.to_datetime(df_data['Date and Time'])
        df_data.set_index('Date and Time', inplace=True)
        # df_data['Date and Time'] = pd.to_datetime()
        print(list(df_data))
        return df_data

#create_csv()
df_data = read_csv()
data_column = 'Frequency in Hz'

def Assignment1_3_a(df_data, data_column):
        print(str(df_data[data_column].median()))
        print(str(df_data[data_column].mean()))
        print(str(df_data[data_column].std()))

        df_data[data_column].plot()
        plt.title('Frequency over time')
        plt.ylabel('Frequency in Hz')
        plt.xlabel('Time')
        plt.show()


def Assignment1_3_b(df_data, data_column):
        sync_time_in_seconds = 1./50.
        df_data['measured_time_in_seconds'] = 1./df_data[data_column]
        df_data['time_deviation_per_second'] = sync_time_in_seconds - df_data['measured_time_in_seconds']
        df_data['sync_time_deviation'] = df_data['time_deviation_per_second'].cumsum()

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_b_Plot(df_data):
        df_data['sync_time_deviation'].plot()

        plt.title('Sync time deviation')
        plt.ylabel('Seconds')
        plt.xlabel('Time')
        plt.show()

def Characteristic(frequency):
        nominal_frequency = 50.

        delta_frequency = nominal_frequency - frequency

        absolute_delta = abs(delta_frequency)

        slope = (1.-0.)/(200./1000.-10./1000.) # (y-y0) = r.(x-x0) ... r = (y-y0)/(x-x0) ... r = (-1 - 0) / (200./1000.-10./1000.)
        if absolute_delta < 10./1000. : #+-10 mHz
                return 0
        elif delta_frequency > 0.:
                return -30000.*slope*absolute_delta
        elif delta_frequency < 0.:
                return 30000.*slope*absolute_delta
        else:
                print('outside the borders buddy')
                return 0

def Assignment1_3_c(df_data, data_column):
        print('calculating FCR')
        df_data['FCR'] = df_data[data_column].apply(Characteristic)
        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_c_Plot(df_data):
        print('generating plot')

        df_data['FCR'].plot()
        plt.title('FCR')
        plt.ylabel('Power in MW')
        plt.xlabel('Time')
        plt.show()


def Assignment1_3_d(df_data):
        print('filtering positive FCR')
        df_data['FCR_positive'] = df_data['FCR']
        df_data['FCR_positive'].values[df_data['FCR_positive'].values < 0.] = 0.
        print('filtering negative FCR')
        df_data['FCR_negative'] = df_data['FCR']
        df_data['FCR_negative'].values[df_data['FCR_negative'].values > 0.] = 0.

        print('cumulative_sum_of_pos_fcr')
        df_data['FCR_pos_cumsum'] = df_data['FCR_positive']*(1./(60.*60.))
        df_data['FCR_pos_cumsum'] = df_data['FCR_pos_cumsum'].cumsum()

        print('cumulative_sum_of_neg_fcr')
        df_data['FCR_neg_cumsum'] = df_data['FCR_negative']*(1./(60.*60.))
        df_data['FCR_neg_cumsum'] = df_data['FCR_neg_cumsum'].cumsum()
        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_d_Plot(df_data):
        df_data['FCR_pos_cumsum'].plot(label='Positive Net Energy')
        df_data['FCR_neg_cumsum'].plot(label='Negative Net Energy')
        #plt.fill_between(df_data.index.values, df_data['FCR_pos_cumsum'], df_data['FCR_neg_cumsum'], color='green', alpha=0.2)
        dates = df_data.index
        #plt.fill_between(dates, df_data['FCR_pos_cumsum'], df_data['FCR_neg_cumsum'],
        #                 where=df_data['FCR_pos_cumsum'] >= df_data['FCR_neg_cumsum'],
        #                 facecolor='green',alpha=0.2,interpolate=True)
        plt.title('FCR Net Energy')
        plt.ylabel('FCR Net Energy in MWh')
        plt.xlabel('Time (1 second sampling size)')
        plt.show()


def Assignment1_3_e(df_data):
        df_data_resampled_15min = df_data.resample(timedelta(minutes=15)).mean()
        df_data_resampled_15min = df_data_resampled_15min.resample(timedelta(seconds=1)).pad()

        df_data_resampled_30min = df_data.resample(timedelta(minutes=30)).mean()
        df_data_resampled_30min = df_data_resampled_30min.resample(timedelta(seconds=1)).pad()

        df_data_resampled_60min = df_data.resample(timedelta(minutes=60)).mean()
        df_data_resampled_60min = df_data_resampled_60min.resample(timedelta(seconds=1)).pad()
        #df_data['FCR'].plot(label='1 sec')
        df_data_resampled_15min['FCR'].plot(label='15 min')
        df_data_resampled_30min['FCR'].plot(label='30 min')
        df_data_resampled_60min['FCR'].plot(label='60 min')
        #plt.fill_between(df_data_resampled_15min.index.values,0., df_data_resampled_15min['FCR'], where=df_data_resampled_15min['FCR'] >0., color='green', alpha=0.2)
        #plt.fill_between(df_data_resampled_15min.index.values, df_data_resampled_15min['FCR'],0., where=df_data_resampled_15min['FCR'] <0., color='red', alpha=0.2)

        plt.legend()
        plt.title('FCR')
        plt.ylabel('FCR MW')
        plt.xlabel('Time (sampling size on label)')

        plt.show()

        df_data_resampled_15min['FCR_cumsum'] = df_data_resampled_15min['FCR'] * (1. / (60. * 60.))
        df_data_resampled_15min['FCR_cumsum'] = df_data_resampled_15min['FCR_cumsum'].cumsum()
        df_data_resampled_30min['FCR_cumsum'] = df_data_resampled_30min['FCR']*(1./(60.*60.))
        df_data_resampled_30min['FCR_cumsum'] = df_data_resampled_30min['FCR_cumsum'].cumsum()
        df_data_resampled_60min['FCR_cumsum'] = df_data_resampled_60min['FCR']*(1./(60.*60.))
        df_data_resampled_60min['FCR_cumsum'] = df_data_resampled_60min['FCR_cumsum'].cumsum()

        df_data_resampled_15min['FCR_cumsum'].plot(label='15 min')
        df_data_resampled_30min['FCR_cumsum'].plot(label='30 min')
        df_data_resampled_60min['FCR_cumsum'].plot(label='60 min')

        plt.title('FCR Net Energy')
        plt.ylabel('FCR Net Energy in MWh')
        plt.xlabel('Time (sampling size on label)')
        plt.legend()
        plt.show()


def Assignment1_3_f(df_data, data_column):
        Switzerland_Activation_power = 74. #MW
        Total_activation_power = 30000. #MW
        Swiss_ratio_of_power = Switzerland_Activation_power/Total_activation_power

        df_data['Swiss FCR'] = df_data[data_column].apply(Characteristic)
        df_data['Swiss FCR'] = df_data['Swiss FCR']*Swiss_ratio_of_power

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment_3_f_Plot(df_data):
        df_data['Swiss FCR'].plot()


        plt.title('Swiss FCR ')
        plt.ylabel('FCR in MW')
        plt.xlabel('Time')
        plt.show()


#Assignment1_3_a(df_data,data_column)

#Assignment1_3_b(df_data,data_column)
#Assignment1_3_b_Plot(df_data)

#Assignment1_3_c(df_data,data_column)
#Assignment1_3_c_Plot(df_data)

#Assignment1_3_d(df_data)
#Assignment1_3_d_Plot(df_data)

#Assignment1_3_e(df_data)

#Assignment1_3_f(df_data,data_column)
#Assignment_3_f_Plot(df_data)

def Assignment1_3_g(df_data,data_column):
        #df_data_resampled_15min = df_data.resample(timedelta(minutes=15)).mean()
        #df_data_resampled_15min = df_data_resampled_15min.resample(timedelta(seconds=1)).pad()

        #df_data_resampled_30min = df_data.resample(timedelta(minutes=30)).mean()
        #df_data_resampled_30min = df_data_resampled_30min.resample(timedelta(seconds=1)).pad()

        #df_data_resampled_60min = df_data.resample(timedelta(minutes=60)).mean()
        #df_data_resampled_60min = df_data_resampled_60min.resample(timedelta(seconds=1)).pad()
        #df_data_resampled_15min[data_column].plot(label='15 min')
        #df_data_resampled_30min[data_column].plot(label='30 min')
        #df_data_resampled_60min[data_column].plot(label='60 min')
        #plt.fill_between(df_data_resampled_15min.index.values,0., df_data_resampled_15min['FCR'], where=df_data_resampled_15min['FCR'] >0., color='green', alpha=0.2)
        #plt.fill_between(df_data_resampled_15min.index.values, df_data_resampled_15min['FCR'],0., where=df_data_resampled_15min['FCR'] <0., color='red', alpha=0.2)
        df_data = df_data.resample(timedelta(minutes=30)).mean()
        df_data['Day'] = df_data.index.weekday
        #df_data['fake_index'] = df_data.index.strftime('%H:%M:%S')
        df_data['fake_index'] = df_data.index.time
        pivot_table = pd.pivot_table(df_data, index = df_data['fake_index'],columns=['Day'], values=[data_column])
        fig, ax = plt.subplots()
        pivot_table.plot(ax=ax)
        #df_data['FCR'].plot(label='1 sec')

        print(df_data['fake_index'])
        #plt.grid(which='both')
        plt.legend()
        plt.title('Frequency ')
        plt.ylabel('Frequency in Hz')
        plt.xlabel('Time (sampling size on label)')
        hours = mdates.MinuteLocator(byminute=range(60))
        minorFMT = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_minor_locator(hours)
        ax.xaxis.set_minor_formatter(minorFMT)
        plt.grid(which='both', axis='both')

        plt.setp(ax.xaxis.get_minorticklabels(),rotation=90)
        plt.show()

Assignment1_3_g(df_data,data_column)