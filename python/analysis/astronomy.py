import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings


size_tick_min_max = 100
size_tick_scatter = 10

color_scatter_specific_training = '#1f77b4'
color_scatter_non_specific_training = '#ff7f0e'

color_plot_polynomium_1_deg_specific_training = '#2ca02c'
color_plot_polynomium_1_deg_non_specific_training = '#d62728'

color_plot_polynomium_60_deg_specific_training = '#9467bd'
color_plot_polynomium_60_deg_non_specific_training = '#8c564b'

color_histogram_specific_training = '#ffd000'
color_histogram_non_specific_training = '#2ca02c'

color_minor_grid = '#c9c9c9'
color_major_grid = '#a1a1a1'

#-----------------------------------------------------------------------------------------------------------------------------

input_file_astronomy_specific_training = './results_json/saida_astronomy.json'
input_file_astronomy_non_specific_training = './results_json/saida_astronomy_general_purpose_training.json'

# algorithms = ['mse', 'rmse', 'psnr', 'uqi', 'ssim', 'ergas', 'scc', 'rase', 'sam', 'msssim', 'vifp']
algorithms = ['mse', 'rmse', 'psnr', 'uqi', 'ergas', 'scc', 'rase', 'sam', 'vifp']

input_file_astronomy_specific_training_content = open(input_file_astronomy_specific_training, 'r').read()
input_file_astronomy_non_specific_training_content = open(input_file_astronomy_non_specific_training, 'r').read()

input_file_astronomy_specific_training_dict = json.loads(input_file_astronomy_specific_training_content)
input_file_astronomy_non_specific_training_dict = json.loads(input_file_astronomy_non_specific_training_content)

astronomy_specific_training_dict = input_file_astronomy_specific_training_dict['astronomy']
astronomy_non_specific_training_dict = input_file_astronomy_non_specific_training_dict['astronomy-general-training']


def get_img_data(img, img_name):
    results = [img['results'][algorithm] for algorithm in algorithms if 'results' in img]
    return [img_name] + results    

def get_column_names():
    return ['img_name'] + algorithms


df_astronomy_specific_training_data = [get_img_data(astronomy_specific_training_dict[img_name], img_name) for img_name in astronomy_specific_training_dict]
df_astronomy_non_specific_training_data = [get_img_data(astronomy_non_specific_training_dict[img_name], img_name) for img_name in astronomy_non_specific_training_dict]

df_astronomy_specific_training = pd.DataFrame(df_astronomy_specific_training_data, columns = get_column_names())
df_astronomy_non_specific_training = pd.DataFrame(df_astronomy_non_specific_training_data, columns = get_column_names())

df_astronomy_specific_training.replace(np.nan, 0, regex=True)
df_astronomy_non_specific_training.replace(np.nan, 0, regex=True)


for algorithm in algorithms:
    df_astronomy_specific_training[algorithm] = df_astronomy_specific_training[algorithm].replace('nan', np.nan)
    df_astronomy_non_specific_training[algorithm] = df_astronomy_non_specific_training[algorithm].replace('nan', np.nan)

for algorithm in algorithms:
    df_astronomy_specific_training[algorithm] = pd.to_numeric(df_astronomy_specific_training[algorithm], downcast="float")
    df_astronomy_non_specific_training[algorithm] = pd.to_numeric(df_astronomy_non_specific_training[algorithm], downcast="float")


df_astronomy_specific_training['index'] = range(1, len(df_astronomy_specific_training) + 1)
df_astronomy_non_specific_training['index'] = range(1, len(df_astronomy_non_specific_training) + 1)


if len(df_astronomy_specific_training) > len(df_astronomy_non_specific_training):
    df_astronomy_specific_training = df_astronomy_specific_training.iloc[:len(df_astronomy_non_specific_training)]
elif len(df_astronomy_specific_training) < len(df_astronomy_non_specific_training):
    df_astronomy_non_specific_training = df_astronomy_non_specific_training.iloc[:len(df_astronomy_specific_training)]


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MSE - MRI

key = 'mse'

fig = plt.figure(figsize=(18,15))

fig.suptitle('MSE - Base de dados Astronômica', y=0.95, fontsize=18, fontweight=60)

gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2, :])


# General plot

ax1.set_title('Gráfico geral')

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

ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='MSE - Min. t. específico')
ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='MSE - Min. t. genérico')
ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='MSE - Máx. t. específico')
ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='MSE - Máx. t. genérico')

m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 60))
    polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 60))

ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='MSE - t. específico interpolado em 1º grau')
ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='MSE - t. genérico interpolado em 1º grau')

ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='MSE - t. específico interpolado em 60º grau')
ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='MSE - t. genérico interpolado em 60º grau')

ax1.grid(which='major', color=color_major_grid, linestyle='-')
ax1.grid(which='minor', color=color_minor_grid, linestyle='--')

ax1.legend(loc="lower right")
ax1.minorticks_on()
ax1.set_axisbelow(True)




# Histogram specific training

ax2.set_title('Histograma')

ax2.grid(which='major', color=color_major_grid, linestyle='-')
ax2.grid(which='minor', color=color_minor_grid, linestyle='--')

ax2.minorticks_on()

_ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='MSE - t. específico', edgecolor='black')
_ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='MSE - t. genérico', edgecolor='black')

ax2.legend(loc="upper right")

ax2.set_axisbelow(True)





plt.savefig('mse_astronomy_compound.svg')
plt.savefig('mse_astronomy_compound.pdf')

# plt.show()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# RMSE - MRI

key = 'rmse'

fig = plt.figure(figsize=(18,15))

fig.suptitle('RMSE - Base de dados Astronômica', y=0.95, fontsize=18, fontweight=60)

gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2, :])


# General plot

ax1.set_title('Gráfico geral')

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

ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='RMSE - Min. t. específico')
ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='RMSE - Min. t. genérico')
ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='RMSE - Máx. t. específico')
ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='RMSE - Máx. t. genérico')

m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 60))
    polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 60))

ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='RMSE - t. específico interpolado em 1º grau')
ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='RMSE - t. genérico interpolado em 1º grau')

ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='RMSE - t. específico interpolado em 60º grau')
ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='RMSE - t. genérico interpolado em 60º grau')

ax1.grid(which='major', color=color_major_grid, linestyle='-')
ax1.grid(which='minor', color=color_minor_grid, linestyle='--')

ax1.legend(loc="lower right")
ax1.minorticks_on()
ax1.set_axisbelow(True)




# Histogram specific training

ax2.set_title('Histograma')

ax2.grid(which='major', color=color_major_grid, linestyle='-')
ax2.grid(which='minor', color=color_minor_grid, linestyle='--')

ax2.minorticks_on()

_ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='RMSE - t. específico', edgecolor='black')
_ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='RMSE - t. genérico', edgecolor='black')

ax2.legend(loc="upper right")

ax2.set_axisbelow(True)





plt.savefig('rmse_astronomy_compound.svg')
plt.savefig('rmse_astronomy_compound.pdf')

# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ERGAS - MRI

key = 'ergas'

fig = plt.figure(figsize=(18,15))

fig.suptitle('ERGAS - Base de dados Astronômica', y=0.95, fontsize=18, fontweight=60)

gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2, :])


# General plot

ax1.set_title('Gráfico geral')

ax1.set_yscale("linear")

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

ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='ERGAS - Min. t. específico')
ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='ERGAS - Min. t. genérico')
ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='ERGAS - Máx. t. específico')
ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='ERGAS - Máx. t. genérico')

m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 60))
    polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 60))

ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='ERGAS - t. específico interpolado em 1º grau')
ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='ERGAS - t. genérico interpolado em 1º grau')

ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='ERGAS - t. específico interpolado em 60º grau')
ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='ERGAS - t. genérico interpolado em 60º grau')

ax1.grid(which='major', color=color_major_grid, linestyle='-')
ax1.grid(which='minor', color=color_minor_grid, linestyle='--')

ax1.legend(loc="lower right")
ax1.minorticks_on()
ax1.set_axisbelow(True)




# Histogram specific training

ax2.set_title('Histograma')

ax2.grid(which='major', color=color_major_grid, linestyle='-')
ax2.grid(which='minor', color=color_minor_grid, linestyle='--')

ax2.minorticks_on()

_ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='ERGAS - t. específico', edgecolor='black')
_ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='ERGAS - t. genérico', edgecolor='black')

ax2.legend(loc="upper right")

ax2.set_axisbelow(True)





plt.savefig('ergas_astronomy_compound.svg')
plt.savefig('ergas_astronomy_compound.pdf')

# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PSNR - MRI

key = 'psnr'

fig = plt.figure(figsize=(18,15))

fig.suptitle('PSNR - Base de dados Astronômica', y=0.95, fontsize=18, fontweight=60)

gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2, :])


# General plot

ax1.set_title('Gráfico geral')

ax1.set_yscale("linear")

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

ax1.scatter(min_x_specific_training, min_y_specific_training, marker='*', s=size_tick_min_max, c='r', label='PSNR - Min. t. específico')
ax1.scatter(min_x_non_specific_training, min_y_non_specific_training, marker='*', s=size_tick_min_max, c='b', label='PSNR - Min. t. genérico')
ax1.scatter(max_x_specific_training, max_y_specific_training, marker='*', s=size_tick_min_max, c='y', label='PSNR - Máx. t. específico')
ax1.scatter(max_x_non_specific_training, max_y_non_specific_training, marker='*', s=size_tick_min_max, c='g', label='PSNR - Máx. t. genérico')

m_astronomy_specific_training, b_astronomy_specific_training = np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 1)
m_astronomy_non_specific_training, b_astronomy_non_specific_training = np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 1)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    polynomium_astronomy_specific_training = np.poly1d(np.polyfit(df_astronomy_specific_training['index'], df_astronomy_specific_training[key], 60))
    polynomium_astronomy_non_specific_training = np.poly1d(np.polyfit(df_astronomy_non_specific_training['index'], df_astronomy_non_specific_training[key], 60))

ax1.plot(df_astronomy_specific_training['index'], m_astronomy_specific_training*df_astronomy_specific_training['index'] + b_astronomy_specific_training, color=color_plot_polynomium_1_deg_specific_training, label='PSNR - t. específico interpolado em 1º grau')
ax1.plot(df_astronomy_non_specific_training['index'], m_astronomy_non_specific_training*df_astronomy_non_specific_training['index'] + b_astronomy_non_specific_training, color=color_plot_polynomium_1_deg_non_specific_training, label='PSNR - t. genérico interpolado em 1º grau')

ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_specific_training, label='PSNR - t. específico interpolado em 60º grau')
ax1.plot(df_astronomy_specific_training['index'], polynomium_astronomy_non_specific_training(df_astronomy_specific_training['index']), '--', color=color_plot_polynomium_60_deg_non_specific_training, label='PSNR - t. genérico interpolado em 60º grau')

ax1.grid(which='major', color=color_major_grid, linestyle='-')
ax1.grid(which='minor', color=color_minor_grid, linestyle='--')

ax1.legend(loc="lower right")
ax1.minorticks_on()
ax1.set_axisbelow(True)




# Histogram specific training

ax2.set_title('Histograma')

ax2.grid(which='major', color=color_major_grid, linestyle='-')
ax2.grid(which='minor', color=color_minor_grid, linestyle='--')

ax2.minorticks_on()

_ = ax2.hist(df_astronomy_specific_training[key], bins='auto', color=color_histogram_specific_training, label='PSNR - t. específico', edgecolor='black')
_ = ax2.hist(df_astronomy_non_specific_training[key], bins='auto', color=color_histogram_non_specific_training, label='PSNR - t. genérico', edgecolor='black')

ax2.legend(loc="upper right")

ax2.set_axisbelow(True)





plt.savefig('psnr_astronomy_compound.svg')
plt.savefig('psnr_astronomy_compound.pdf')

# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------