import pandas as pd
import numpy as np
from datetime import timedelta
import sys
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
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
        #print(list(df_data))
        return df_data

def Assignment1_3_a(df_data, data_column):
        print('1.3 a) Calculation')
        print('Median: '+str(df_data[data_column].median()))
        print('Mean: '+str(df_data[data_column].mean()))
        print('Standard Deviation: '+str(df_data[data_column].std()))

def Assignment1_3_a_Plot(df_data, data_column):
        print('1.3 a) Plot')
        fig, ax = plt.subplots()

        df_data[data_column].plot(ax=ax, linewidth=1)
        plt.title('Frequency over time')
        plt.ylabel('Frequency in Hz')
        plt.xlabel('Time')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)
        fig.savefig('./Day1/1_3a.png', bbox_inches='tight')
        plt.close(fig)

def Assignment1_3_b(df_data, data_column):
        print('1.3 b) Calculation')
        sync_time_in_seconds = 1./50.
        df_data['measured_time_in_seconds'] = 1./df_data[data_column]
        df_data['time_deviation_per_second'] = (sync_time_in_seconds - df_data['measured_time_in_seconds'])*df_data[data_column]
        df_data['sync_time_deviation'] = df_data['time_deviation_per_second'].cumsum()

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_b_Plot(df_data):
        print('1.3 b) Plot')
        fig, ax = plt.subplots()
        df_data['sync_time_deviation'].plot(ax=ax, linewidth = 1)

        plt.title('Sync time deviation')
        plt.ylabel('Seconds')
        plt.xlabel('Time')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)
        fig.savefig('./Day1/1_3b.png', bbox_inches='tight')
        plt.close(fig)

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
        print('1.3 c) Calculation')
        #print('calculating FCR')
        df_data['FCR'] = df_data[data_column].apply(Characteristic)

        df_data['FCR_positive'] = df_data['FCR']
        df_data['FCR_positive'].values[df_data['FCR_positive'].values < 0.] = 0.

        df_data['FCR_negative'] = df_data['FCR']
        df_data['FCR_negative'].values[df_data['FCR_negative'].values > 0.] = 0.

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_c_Plot(df_data):
        print('1.3 c) Plot')

        fig, ax = plt.subplots()
        df_data['FCR_positive'].plot(ax=ax, linewidth = 1)
        df_data['FCR_negative'].plot(ax=ax, linewidth = 1)
        plt.title('FCR')
        plt.ylabel('Power in MW')
        plt.xlabel('Time')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)
        fig.savefig('./Day1/1_3c.png', bbox_inches='tight')
        plt.close(fig)

def Assignment1_3_d(df_data):
        print('1.3 d) Calculation')
        df_data['FCR_positive'] = df_data['FCR']
        df_data['FCR_positive'].values[df_data['FCR_positive'].values < 0.] = 0.

        df_data['FCR_negative'] = df_data['FCR']
        df_data['FCR_negative'].values[df_data['FCR_negative'].values > 0.] = 0.

        df_data['FCR_pos_cumsum'] = df_data['FCR_positive']*(1./(60.*60.))
        df_data['FCR_pos_cumsum'] = df_data['FCR_pos_cumsum'].cumsum()

        df_data['FCR_neg_cumsum'] = df_data['FCR_negative']*(1./(60.*60.))
        df_data['FCR_neg_cumsum'] = df_data['FCR_neg_cumsum'].cumsum()

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment1_3_d_Plot(df_data):

        print('1.3 d) Plot')
        fig,ax=plt.subplots()

        df_data['FCR_pos_cumsum'].plot(label='Positive Net Energy', ax=ax, linewidth = 1)
        df_data['FCR_neg_cumsum'].plot(label='Negative Net Energy', ax=ax, linewidth = 1)
        plt.fill_between(df_data.index.values, df_data['FCR_pos_cumsum'], y2=0,color='green', alpha=0.2)
        plt.fill_between(df_data.index.values, df_data['FCR_neg_cumsum'], y2=0,  color='red', alpha=0.2)
        dates = df_data.index
        #plt.fill_between(dates, df_data['FCR_pos_cumsum'], df_data['FCR_neg_cumsum'],
        #                 where=df_data['FCR_pos_cumsum'] >= df_data['FCR_neg_cumsum'],
        #                 facecolor='green',alpha=0.2,interpolate=True)
        plt.title('FCR Net Energy')
        plt.ylabel('FCR Net Energy in MWh')
        plt.xlabel('Time (1 second sampling size)')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)
        fig.savefig('./Day1/1_3d.png', bbox_inches='tight')
        plt.close(fig)

def Assignment1_3_e(df_data):

        print('1.3 e) Calculation')
        df_data_resampled_15min = df_data.resample(timedelta(minutes=15)).mean()
        df_data_resampled_15min = df_data_resampled_15min.resample(timedelta(seconds=1)).pad()

        df_data_resampled_30min = df_data.resample(timedelta(minutes=30)).mean()
        df_data_resampled_30min = df_data_resampled_30min.resample(timedelta(seconds=1)).pad()

        df_data_resampled_60min = df_data.resample(timedelta(minutes=60)).mean()
        df_data_resampled_60min = df_data_resampled_60min.resample(timedelta(seconds=1)).pad()

        print('1.3 e) Plot (FCR)')

        fig, ax = plt.subplots()
        df_data_resampled_15min['FCR'].plot(label='15 min',ax=ax, linewidth = 1)
        df_data_resampled_30min['FCR'].plot(label='30 min',ax=ax, linewidth = 1)
        df_data_resampled_60min['FCR'].plot(label='60 min',ax=ax, linewidth = 1)
        #plt.fill_between(df_data_resampled_15min.index.values,0., df_data_resampled_15min['FCR'], where=df_data_resampled_15min['FCR'] >0., color='green', alpha=0.2)
        #plt.fill_between(df_data_resampled_15min.index.values, df_data_resampled_15min['FCR'],0., where=df_data_resampled_15min['FCR'] <0., color='red', alpha=0.2)

        plt.legend()
        plt.title('FCR')
        plt.ylabel('FCR MW')
        plt.xlabel('Time (sampling size on label)')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)

        df_data_resampled_15min['FCR_cumsum'] = df_data_resampled_15min['FCR'] * (1. / (60. * 60.))
        df_data_resampled_15min['FCR_cumsum'] = df_data_resampled_15min['FCR_cumsum'].cumsum()
        df_data_resampled_30min['FCR_cumsum'] = df_data_resampled_30min['FCR']*(1./(60.*60.))
        df_data_resampled_30min['FCR_cumsum'] = df_data_resampled_30min['FCR_cumsum'].cumsum()
        df_data_resampled_60min['FCR_cumsum'] = df_data_resampled_60min['FCR']*(1./(60.*60.))
        df_data_resampled_60min['FCR_cumsum'] = df_data_resampled_60min['FCR_cumsum'].cumsum()

        fig.savefig('./Day1/1_3e_FCR_MW.png', bbox_inches='tight')
        plt.close(fig)
        print('1.3 e) Plot (FCR Net Energy)')

        fig, ax = plt.subplots()
        df_data_resampled_15min['FCR_cumsum'].plot(label='15 min',ax=ax, linewidth = 1)
        df_data_resampled_30min['FCR_cumsum'].plot(label='30 min',ax=ax, linewidth = 1)
        df_data_resampled_60min['FCR_cumsum'].plot(label='60 min',ax=ax, linewidth = 1)

        plt.title('FCR Net Energy')
        plt.ylabel('FCR Net Energy in MWh')
        plt.xlabel('Time (sampling size on label)')
        plt.legend()
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)
        fig.savefig('./Day1/1_3e_FCR_NET_Energy.png', bbox_inches='tight')
        plt.close(fig)

def Assignment1_3_f(df_data, data_column):

        print('1.3 f) Calculation')
        Switzerland_Activation_power = 74. #MW
        Total_activation_power = 30000. #MW
        Swiss_ratio_of_power = Switzerland_Activation_power/Total_activation_power

        df_data['Swiss FCR'] = df_data[data_column].apply(Characteristic)
        df_data['Swiss FCR'] = df_data['Swiss FCR']*Swiss_ratio_of_power

        df_data.to_csv('./Data_in_CSV.csv', sep=';', index=True)

def Assignment_3_f_Plot(df_data):
        print('1.3 f) Plot')

        fig, ax = plt.subplots()

        df_data['Swiss FCR'].plot(ax=ax, linewidth = 1)
        plt.title('Swiss FCR (Switzerland Activation power = 73 MW, Grid = 30000 MW')
        plt.ylabel('FCR in MW')
        plt.xlabel('Time')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor', axis='both', alpha=0.4)

        fig.savefig('./Day1/1_3f_FCR_SWISS.png', bbox_inches='tight')
        plt.close(fig)

def Assignment1_3_g(df_data,data_column):


        print('1.3 g) Calculation')
        df_data = df_data.resample(timedelta(minutes=5)).mean()
        df_data['Day'] = df_data.index.strftime('%a')#weekday
        df_data['fake_index'] = df_data.index.time
        pivot_table = pd.pivot_table(df_data, index = df_data['fake_index'],columns=['Day'], values=[data_column])

        print('1.3 g) Plot')

        fig, ax = plt.subplots()
        pivot_table.plot(ax=ax, linewidth = 1)
        #df_data['FCR'].plot(label='1 sec')
        plt.grid(which='major', axis='both', alpha=0.6)
        plt.grid(which='minor',axis='both', alpha=0.4)
        plt.legend(loc='upper right')
        plt.title('Frequency ')
        plt.ylabel('Frequency in Hz')
        plt.xlabel('Time (sampling size on label)')
        lista = np.arange(0,24).tolist()
        for num in range(len(lista)):
                lista[num] *= (4*900)
        ax.set_xticks(lista)

        plt.setp(ax.xaxis.get_majorticklabels(),rotation=90)
        fig.savefig('./Day1/1_3g_Freq.png', bbox_inches='tight')
        plt.close(fig)


def Assignment1_3_g_ANOTHER_VIEW(df_data,data_column):


        print('1.3 g) Calculation')
        df_data = df_data.resample(timedelta(minutes=5)).mean()
        df_data['Day'] = df_data.index.strftime('%a')#weekday
        df_data['fake_index'] = df_data.index.time
        pivot_table = pd.pivot_table(df_data, index = df_data['fake_index'],columns=['Day'], values=[data_column])

        print('1.3 g) Plot')
        list_of_days = df_data['Day'].unique().tolist()
        fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,sharex='all', figsize=[12,10])
        list_of_axes = [ax1,ax2,ax3,ax4,ax5,ax6]
        list_of_colors = ['blue','orange','green','red','purple','brown','pink']
        counter=0

        ax1.set_title('Frequency ')
        plt.ylabel('Frequency in Hz')
        for ax in list_of_axes:
                filtered_df = df_data[df_data['Day']==list_of_days[counter]]
                ax.plot(filtered_df['fake_index'],filtered_df[data_column], color=list_of_colors[counter], label=list_of_days[counter], linewidth=1)
                ax.grid(which='major', axis='both', alpha=0.6)
                ax.grid(which='minor',axis='both', alpha=0.4)
                ax.legend(loc='upper right')
                ax.set_xlim(left=df_data['fake_index'][0], right=df_data['fake_index'][-1])
                ax.set_ylim(bottom=49.9, top=50.1)
                counter+=1
        #plt.grid(which='major', axis='both', alpha=0.6)
        #plt.grid(which='minor',axis='both', alpha=0.4)
        plt.legend(loc='upper right')
        plt.xlabel('Time (sampling size on label)')
        lista = np.arange(0,24).tolist()
        for num in range(len(lista)):
                lista[num] *= (4*900)
        ax.set_xticks(lista)

        plt.setp(ax.xaxis.get_majorticklabels(),rotation=90)

        fig.savefig('./Day1/1_3g_Freq_separate.png', bbox_inches='tight')
        plt.close(fig)
#GENERATE THE Data_in_CSV by running the create_csv() method.
#create_csv()
df_data = read_csv()
data_column = 'Frequency in Hz'


#Assignment1_3_a(df_data,data_column)
#Assignment1_3_a_Plot(df_data,data_column)

#Assignment1_3_b(df_data,data_column)
#Assignment1_3_b_Plot(df_data)

#Assignment1_3_c(df_data,data_column)
#Assignment1_3_c_Plot(df_data)

#Assignment1_3_d(df_data)
#Assignment1_3_d_Plot(df_data)

#Assignment1_3_e(df_data)

#Assignment1_3_f(df_data,data_column)
#Assignment_3_f_Plot(df_data)

#Assignment1_3_g(df_data,data_column)
Assignment1_3_g_ANOTHER_VIEW(df_data,data_column)
#plt.show()