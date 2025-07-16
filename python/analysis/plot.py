import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import warnings
import scipy
import seaborn as sns


def get_img_data(img, img_name):
    results = [img['results'][algorithm] for algorithm in algorithms if 'results' in img]
    return [img_name] + results    


def get_column_names():
    return ['img_name'] + algorithms


input_file_mri_specific_training = './results_json/saida_mri.json'
input_file_mri_non_specific_training = './results_json/saida_mri_general_purpose_training.json'
input_file_astronomy_specific_training = './results_json/saida_astronomy.json'
input_file_astronomy_non_specific_training = './results_json/saida_astronomy_general_purpose_training.json'

algorithms = ['mse', 'rmse', 'psnr', 'uqi', 'ergas', 'scc', 'rase', 'sam', 'vifp']

input_file_mri_specific_training_content = open(input_file_mri_specific_training, 'r').read()
input_file_mri_non_specific_training_content = open(input_file_mri_non_specific_training, 'r').read()
input_file_astronomy_specific_training_content = open(input_file_astronomy_specific_training, 'r').read()
input_file_astronomy_non_specific_training_content = open(input_file_astronomy_non_specific_training, 'r').read()

input_file_mri_specific_training_dict = json.loads(input_file_mri_specific_training_content)
input_file_mri_non_specific_training_dict = json.loads(input_file_mri_non_specific_training_content)
input_file_astronomy_specific_training_dict = json.loads(input_file_astronomy_specific_training_content)
input_file_astronomy_non_specific_training_dict = json.loads(input_file_astronomy_non_specific_training_content)

mri_specific_training_dict = input_file_mri_specific_training_dict['mri']
mri_non_specific_training_dict = input_file_mri_non_specific_training_dict['mri-general-training']
astronomy_specific_training_dict = input_file_astronomy_specific_training_dict['astronomy']
astronomy_non_specific_training_dict = input_file_astronomy_non_specific_training_dict['astronomy-general-training']

df_mri_specific_training_data = [get_img_data(mri_specific_training_dict[img_name], img_name) for img_name in mri_specific_training_dict]
df_mri_non_specific_training_data = [get_img_data(mri_non_specific_training_dict[img_name], img_name) for img_name in mri_non_specific_training_dict]
df_astronomy_specific_training_data = [get_img_data(astronomy_specific_training_dict[img_name], img_name) for img_name in astronomy_specific_training_dict]
df_astronomy_non_specific_training_data = [get_img_data(astronomy_non_specific_training_dict[img_name], img_name) for img_name in astronomy_non_specific_training_dict]

df_mri_specific_training = pd.DataFrame(df_mri_specific_training_data, columns = get_column_names())
df_mri_non_specific_training = pd.DataFrame(df_mri_non_specific_training_data, columns = get_column_names())
df_astronomy_specific_training = pd.DataFrame(df_astronomy_specific_training_data, columns = get_column_names())
df_astronomy_non_specific_training = pd.DataFrame(df_astronomy_non_specific_training_data, columns = get_column_names())

df_mri_specific_training.replace(np.nan, 0, regex=True)
df_mri_non_specific_training.replace(np.nan, 0, regex=True)
df_astronomy_specific_training = df_astronomy_specific_training.replace(np.nan, 0, regex=True)
df_astronomy_non_specific_training = df_astronomy_non_specific_training.replace(np.nan, 0, regex=True)

# df_mri_specific_training.dropna()
# df_mri_non_specific_training.dropna()
df_astronomy_specific_training = df_astronomy_specific_training.dropna()
df_astronomy_non_specific_training = df_astronomy_non_specific_training.dropna()

for algorithm in algorithms:
    df_mri_specific_training[algorithm] = df_mri_specific_training[algorithm].replace('nan', np.nan)
    df_mri_non_specific_training[algorithm] = df_mri_non_specific_training[algorithm].replace('nan', np.nan)
    df_astronomy_specific_training[algorithm] = df_astronomy_specific_training[algorithm].replace('nan', np.nan)
    df_astronomy_non_specific_training[algorithm] = df_astronomy_non_specific_training[algorithm].replace('nan', np.nan)

for algorithm in algorithms:
    df_mri_specific_training[algorithm] = pd.to_numeric(df_mri_specific_training[algorithm], downcast="float")
    df_mri_non_specific_training[algorithm] = pd.to_numeric(df_mri_non_specific_training[algorithm], downcast="float")
    df_astronomy_specific_training[algorithm] = pd.to_numeric(df_astronomy_specific_training[algorithm], downcast="float")
    df_astronomy_non_specific_training[algorithm] = pd.to_numeric(df_astronomy_non_specific_training[algorithm], downcast="float")

df_mri_specific_training['index'] = range(1, len(df_mri_specific_training) + 1)
df_mri_non_specific_training['index'] = range(1, len(df_mri_non_specific_training) + 1)
df_astronomy_specific_training['index'] = range(1, len(df_astronomy_specific_training) + 1)
df_astronomy_non_specific_training['index'] = range(1, len(df_astronomy_non_specific_training) + 1)

if len(df_mri_specific_training) > len(df_mri_non_specific_training):
    df_mri_specific_training = df_mri_specific_training.iloc[:len(df_mri_non_specific_training)]
elif len(df_mri_specific_training) < len(df_mri_non_specific_training):
    df_mri_non_specific_training = df_mri_non_specific_training.iloc[:len(df_mri_specific_training)]

if len(df_astronomy_specific_training) > len(df_astronomy_non_specific_training):
    df_astronomy_specific_training = df_astronomy_specific_training.iloc[:len(df_astronomy_non_specific_training)]
elif len(df_astronomy_specific_training) < len(df_astronomy_non_specific_training):
    df_astronomy_non_specific_training = df_astronomy_non_specific_training.iloc[:len(df_astronomy_specific_training)]


###########################################################################################################################################################################
###########################################################################################################################################################################
# MRI
###########################################################################################################################################################################
###########################################################################################################################################################################

def mse_mri():
    # MSE - MRI

    key = 'mse'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#9467bd'
    color_histogram_non_specific_training = '#2ca02c'


    fig = plt.figure(figsize=(15,12))


    # fig.suptitle('MSE - Base de dados de Ressonância Magnética', y=0.92)
    fig.suptitle('MSE - Base de dados de Ressonância Magnética', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])

    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_mri_specific_training['index'], df_mri_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='MSE - t. específico')
    ax1.scatter(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='MSE - t. genérico')

    min_x_specific_training = np.argmin(df_mri_specific_training[key]) + 1
    min_y_specific_training = np.min(df_mri_specific_training[key])

    min_x_non_specific_training = np.argmin(df_mri_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_mri_non_specific_training[key])

    max_x_specific_training = np.argmax(df_mri_specific_training[key]) + 1
    max_y_specific_training = np.max(df_mri_specific_training[key])

    max_x_non_specific_training = np.argmax(df_mri_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_mri_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='MSE - Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='MSE - Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='MSE - Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='MSE - Máx. t. genérico')

    m_mri_specific_training, b_mri_specific_training = np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 1)
    m_mri_non_specific_training, b_mri_non_specific_training = np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_mri_specific_training = np.poly1d(np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 40))
        polynomium_mri_non_specific_training = np.poly1d(np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 40))

    ax1.plot(df_mri_specific_training['index'], m_mri_specific_training*df_mri_specific_training['index'] + b_mri_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='MSE - t. específico - regressão em 1º grau')
    ax1.plot(df_mri_non_specific_training['index'], m_mri_non_specific_training*df_mri_non_specific_training['index'] + b_mri_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='MSE - t. genérico - regressão em 1º grau')

    ax1.plot(df_mri_specific_training['index'], polynomium_mri_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='MSE - t. específico - regressão em 40º grau')
    ax1.plot(df_mri_specific_training['index'], polynomium_mri_non_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='MSE - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    # ax1.legend(loc="lower right")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_mri_specific_training[key], bins='auto', color=color_histogram_specific_training, label='MSE - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_mri_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='MSE - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()




    plt.savefig('mse_mri_compound.svg')
    # plt.savefig('mse_mri_compound.pdf')

    plt.show()


def rmse_mri():
    # RMSE - MRI

    key = 'rmse'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#9467bd'
    color_histogram_non_specific_training = '#2ca02c'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('RMSE - Base de dados de Ressonância Magnética', y=0.92)
    fig.suptitle('RMSE - Base de dados de Ressonância Magnética', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_mri_specific_training['index'], df_mri_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='RMSE - t. específico')
    ax1.scatter(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='RMSE - t. genérico')

    min_x_specific_training = np.argmin(df_mri_specific_training[key]) + 1
    min_y_specific_training = np.min(df_mri_specific_training[key])

    min_x_non_specific_training = np.argmin(df_mri_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_mri_non_specific_training[key])

    max_x_specific_training = np.argmax(df_mri_specific_training[key]) + 1
    max_y_specific_training = np.max(df_mri_specific_training[key])

    max_x_non_specific_training = np.argmax(df_mri_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_mri_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='RMSE - Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='RMSE - Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='RMSE - Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='RMSE - Máx. t. genérico')

    m_mri_specific_training, b_mri_specific_training = np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 1)
    m_mri_non_specific_training, b_mri_non_specific_training = np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_mri_specific_training = np.poly1d(np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 40))
        polynomium_mri_non_specific_training = np.poly1d(np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 40))

    ax1.plot(df_mri_specific_training['index'], m_mri_specific_training*df_mri_specific_training['index'] + b_mri_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='RMSE - t. específico - regressão em 1º grau')
    ax1.plot(df_mri_non_specific_training['index'], m_mri_non_specific_training*df_mri_non_specific_training['index'] + b_mri_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='RMSE - t. genérico - regressão em 1º grau')

    ax1.plot(df_mri_specific_training['index'], polynomium_mri_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='RMSE - t. específico - regressão em 40º grau')
    ax1.plot(df_mri_specific_training['index'], polynomium_mri_non_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='RMSE - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    # ax1.legend(loc="lower right")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_mri_specific_training[key], bins='auto', color=color_histogram_specific_training, label='RMSE - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_mri_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='RMSE - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()




    plt.savefig('rmse_mri_compound.svg')
    # plt.savefig('rmse_mri_compound.pdf')

    plt.show()


def psnr_mri():
    # PSNR - MRI

    key = 'psnr'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#9467bd'
    color_histogram_non_specific_training = '#2ca02c'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('PSNR - Base de dados de Ressonância Magnética', y=0.92)
    fig.suptitle('PSNR - Base de dados de Ressonância Magnética', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_mri_specific_training['index'], df_mri_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='PSNR - t. específico')
    ax1.scatter(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='PSNR - t. genérico')

    min_x_specific_training = np.argmin(df_mri_specific_training[key]) + 1
    min_y_specific_training = np.min(df_mri_specific_training[key])

    min_x_non_specific_training = np.argmin(df_mri_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_mri_non_specific_training[key])

    max_x_specific_training = np.argmax(df_mri_specific_training[key]) + 1
    max_y_specific_training = np.max(df_mri_specific_training[key])

    max_x_non_specific_training = np.argmax(df_mri_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_mri_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='PSNR - Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='PSNR - Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='PSNR - Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='PSNR - Máx. t. genérico')

    m_mri_specific_training, b_mri_specific_training = np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 1)
    m_mri_non_specific_training, b_mri_non_specific_training = np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_mri_specific_training = np.poly1d(np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 40))
        polynomium_mri_non_specific_training = np.poly1d(np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 40))

    ax1.plot(df_mri_specific_training['index'], m_mri_specific_training*df_mri_specific_training['index'] + b_mri_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='PSNR - t. específico - regressão em 1º grau')
    ax1.plot(df_mri_non_specific_training['index'], m_mri_non_specific_training*df_mri_non_specific_training['index'] + b_mri_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='PSNR - t. genérico - regressão em 1º grau')

    ax1.plot(df_mri_specific_training['index'], polynomium_mri_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='PSNR - t. específico - regressão em 40º grau')
    ax1.plot(df_mri_specific_training['index'], polynomium_mri_non_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='PSNR - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    # ax1.legend(loc="lower right")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_mri_specific_training[key], bins='auto', color=color_histogram_specific_training, label='PSNR - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_mri_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='PSNR - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()



    plt.savefig('psnr_mri_compound.svg')
    # plt.savefig('psnr_mri_compound.pdf')

    plt.show()


def ergas_mri():
    # ERGAS - MRI

    key = 'ergas'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#9467bd'
    color_histogram_non_specific_training = '#2ca02c'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('ERGAS - Base de dados de Ressonância Magnética', y=0.92)
    fig.suptitle('ERGAS - Base de dados de Ressonância Magnética', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_mri_specific_training['index'], df_mri_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='ERGAS - t. específico')
    ax1.scatter(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='ERGAS - t. genérico')

    min_x_specific_training = np.argmin(df_mri_specific_training[key]) + 1
    min_y_specific_training = np.min(df_mri_specific_training[key])

    min_x_non_specific_training = np.argmin(df_mri_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_mri_non_specific_training[key])

    max_x_specific_training = np.argmax(df_mri_specific_training[key]) + 1
    max_y_specific_training = np.max(df_mri_specific_training[key])

    max_x_non_specific_training = np.argmax(df_mri_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_mri_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='ERGAS - Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='ERGAS - Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='ERGAS - Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='ERGAS - Máx. t. genérico')

    m_mri_specific_training, b_mri_specific_training = np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 1)
    m_mri_non_specific_training, b_mri_non_specific_training = np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_mri_specific_training = np.poly1d(np.polyfit(df_mri_specific_training['index'], df_mri_specific_training[key], 40))
        polynomium_mri_non_specific_training = np.poly1d(np.polyfit(df_mri_non_specific_training['index'], df_mri_non_specific_training[key], 40))

    ax1.plot(df_mri_specific_training['index'], m_mri_specific_training*df_mri_specific_training['index'] + b_mri_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='ERGAS - t. específico - regressão em 1º grau')
    ax1.plot(df_mri_non_specific_training['index'], m_mri_non_specific_training*df_mri_non_specific_training['index'] + b_mri_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='ERGAS - t. genérico - regressão em 1º grau')

    ax1.plot(df_mri_specific_training['index'], polynomium_mri_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='ERGAS - t. específico - regressão em 40º grau')
    ax1.plot(df_mri_specific_training['index'], polynomium_mri_non_specific_training(df_mri_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='ERGAS - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_mri_specific_training[key], bins='auto', color=color_histogram_specific_training, label='ERGAS - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_mri_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='ERGAS - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()  



    plt.savefig('ergas_mri_compound.svg')
    # plt.savefig('ergas_mri_compound.pdf')

    plt.show()

###########################################################################################################################################################################
###########################################################################################################################################################################
# ASTRONOMY
###########################################################################################################################################################################
###########################################################################################################################################################################

def mse_astronomy():
    # MSE - Astronomy

    key = 'mse'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#42d1f5'
    color_histogram_non_specific_training = '#51f542'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('MSE - Base de dados de Astronomia', y=0.95)
    fig.suptitle('MSE - Base de dados de Astronomia', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='MSE - t. específico')
    ax1.scatter(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='MSE - t. genérico')

    min_x_specific_training = np.argmin(df_astronomy_specific_training[key]) + 1
    min_y_specific_training = np.min(df_astronomy_specific_training[key])

    min_x_non_specific_training = np.argmin(df_astronomy_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_astronomy_non_specific_training[key])

    max_x_specific_training = np.argmax(df_astronomy_specific_training[key]) + 1
    max_y_specific_training = np.max(df_astronomy_specific_training[key])

    max_x_non_specific_training = np.argmax(df_astronomy_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_astronomy_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='Máx. t. genérico')

    m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
    m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

    # print(m_astronomy_specific_training, b_astronomy_specific_training)
    # print(df_astronomy_specific_training['mse'].describe())
    # print(df_astronomy_specific_training['mse'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 40))
        polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 40))

    ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='MSE - t. específico - regressão em 1º grau')
    ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='MSE - t. genérico - regressão em 1º grau')

    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='MSE - t. específico - regressão em 40º grau')
    ax1.plot(df_astronomy_non_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_non_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='MSE - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='MSE - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='MSE - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()



    plt.savefig('mse_astronomy_compound.svg')
    # plt.savefig('mse_astronomy_compound.pdf')

    plt.show()


def rmse_astronomy():
    # RMSE - Astronomy

    key = 'rmse'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#42d1f5'
    color_histogram_non_specific_training = '#51f542'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('RMSE - Base de dados de Astronomia', y=0.92)
    fig.suptitle('RMSE - Base de dados de Astronomia', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='RMSE - t. específico')
    ax1.scatter(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='RMSE - t. genérico')

    min_x_specific_training = np.argmin(df_astronomy_specific_training[key]) + 1
    min_y_specific_training = np.min(df_astronomy_specific_training[key])

    min_x_non_specific_training = np.argmin(df_astronomy_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_astronomy_non_specific_training[key])

    max_x_specific_training = np.argmax(df_astronomy_specific_training[key]) + 1
    max_y_specific_training = np.max(df_astronomy_specific_training[key])

    max_x_non_specific_training = np.argmax(df_astronomy_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_astronomy_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='Máx. t. genérico')

    m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
    m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 40))
        polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 40))

    ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='RMSE - t. específico - regressão em 1º grau')
    ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='RMSE - t. genérico - regressão em 1º grau')

    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='RMSE - t. específico - regressão em 40º grau')
    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='RMSE - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='RMSE - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='RMSE - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()



    plt.savefig('rmse_astronomy_compound.svg')
    # plt.savefig('rmse_astronomy_compound.pdf')

    plt.show()


def psnr_astronomy():
    # PSNR - Astronomy

    key = 'psnr'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#42d1f5'
    color_histogram_non_specific_training = '#51f542'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('PSNR - Base de dados de Astronomia', y=0.92)
    fig.suptitle('PSNR - Base de dados de Astronomia', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='PSNR - t. específico')
    ax1.scatter(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='PSNR - t. genérico')

    min_x_specific_training = np.argmin(df_astronomy_specific_training[key]) + 1
    min_y_specific_training = np.min(df_astronomy_specific_training[key])

    min_x_non_specific_training = np.argmin(df_astronomy_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_astronomy_non_specific_training[key])

    max_x_specific_training = np.argmax(df_astronomy_specific_training[key]) + 1
    max_y_specific_training = np.max(df_astronomy_specific_training[key])

    max_x_non_specific_training = np.argmax(df_astronomy_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_astronomy_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='Máx. t. genérico')

    m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
    m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 40))
        polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 40))

    ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='PSNR - t. específico - regressão em 1º grau')
    ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='PSNR - t. genérico - regressão em 1º grau')

    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='PSNR - t. específico - regressão em 40º grau')
    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='PSNR - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='PSNR - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='PSNR - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()



    plt.savefig('psnr_astronomy_compound.svg')
    # plt.savefig('psnr_astronomy_compound.pdf')

    plt.show()


def ergas_astronomy():
    # ERGAS - Astronomy

    key = 'ergas'

    size_tick_min_max = 100
    size_tick_scatter = 10

    color_scatter_specific_training = '#1f77b4'
    color_scatter_non_specific_training = '#ff7f0e'

    color_plot_polynomium_1_deg_specific_training = '#2ca02c'
    color_plot_polynomium_1_deg_non_specific_training = '#d62728'

    color_plot_polynomium_60_deg_specific_training = '#9467bd'
    color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

    color_histogram_specific_training = '#42d1f5'
    color_histogram_non_specific_training = '#51f542'


    fig = plt.figure(figsize=(15,12))

    # fig.suptitle('ERGAS - Base de dados de Astronomia', y=0.92)
    fig.suptitle('ERGAS - Base de dados de Astronomia', size=16)

    gs = fig.add_gridspec(3,2)

    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[2, :])


    # General plot

    ax1.set_title('Gráfico geral', pad=5)

    ax1.set_yscale("log")

    ax1.scatter(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_specific_training, label='ERGAS - t. específico')
    ax1.scatter(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], marker='.', s=size_tick_scatter, color=color_scatter_non_specific_training, label='ERGAS - t. genérico')

    min_x_specific_training = np.argmin(df_astronomy_specific_training[key]) + 1
    min_y_specific_training = np.min(df_astronomy_specific_training[key])

    min_x_non_specific_training = np.argmin(df_astronomy_non_specific_training[key]) + 1
    min_y_non_specific_training = np.min(df_astronomy_non_specific_training[key])

    max_x_specific_training = np.argmax(df_astronomy_specific_training[key]) + 1
    max_y_specific_training = np.max(df_astronomy_specific_training[key])

    max_x_non_specific_training = np.argmax(df_astronomy_non_specific_training[key]) + 1
    max_y_non_specific_training = np.max(df_astronomy_non_specific_training[key])

    ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='Min. t. específico')
    ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='Min. t. genérico')
    ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='Máx. t. específico')
    ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='Máx. t. genérico')

    m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
    m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 40))
        polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 40))

    ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='ERGAS - t. específico - regressão em 1º grau')
    ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='ERGAS - t. genérico - regressão em 1º grau')

    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='ERGAS - t. específico - regressão em 40º grau')
    ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='ERGAS - t. genérico - regressão em 40º grau')

    ax1.grid(which='major', color='#ababab', linestyle='-')
    ax1.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0),prop={'size': 10})
    ax1.minorticks_on()
    ax1.set_axisbelow(True)




    # Histogram specific training

    ax2.set_title('Histograma', pad=5)

    ax2.grid(which='major', color='#ababab', linestyle='-')
    ax2.grid(which='minor', color='#d6d6d6', linestyle='--')

    ax2.minorticks_on()

    _ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='ERGAS - treinamento específico', edgecolor='black', lw=0.5)
    _ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='ERGAS - treinamento genérico', edgecolor='black', lw=0.5)

    ax2.legend(loc="upper right")

    ax2.set_axisbelow(True)

    plt.tight_layout()



    plt.savefig('ergas_astronomy_compound.svg')
    # plt.savefig('ergas_astronomy_compound.pdf')

    plt.show()


if __name__ == '__main__':
    mse_mri()
    rmse_mri()
    psnr_mri()
    ergas_mri()

    mse_astronomy()
    rmse_astronomy()
    psnr_astronomy()
    ergas_astronomy()