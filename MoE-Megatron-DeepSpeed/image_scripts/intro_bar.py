from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


ticklabelpad = mpl.rcParams['xtick.major.pad']


activation_ratio = {
    # 'Exp1-Top1': [20, 19, 18, 17, 16, 15],
    'Exp2-Top1': [30.347, 28.889, 11.949, 12.119, 13.578, 16.833],
    'Exp4-Top1': [33.957, 12.872, 10.788, 10.874, 13.654, 15.26],
    'Exp8-Top1': [31.987, 8.469, 8.724, 9.649, 11.687, 14.702],
}

x_labels = ['2', '4', '6', '8', '10', '12']

different_colors = {
	'Exp2-Top1': (58 / 255.0, 27 / 255.0, 0 / 25.0),
	'Exp4-Top1': (199 / 255.0, 160 / 255.0, 133 / 255.0),
	'Exp8-Top1': (201 / 255.0, 71 / 255.0, 55 / 255.0),
	'xxx': (252 / 255.0, 240 / 255.0, 225 / 255.0),
	'FedBF': 'r',
	# 'FedFT': 'b',
}

different_marks = {
	'Exp2-Top1': 'D',
	'Exp4-Top1': 'x',
	'Exp8-Top1': '^',
	'FedBF': 'o',
	'FedFT': 's',
}

font_size = 130

my_nrows = 1
my_ncols = 1

fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, figsize=(60, 40))
fig.subplots_adjust(hspace=0.75)

this_axes = axes

# 添加柱状图上的数值标签
def autolabel(bars):
    pass
    # for bar in bars:
    #     height = bar.get_height()
    #     this_axes.annotate(f'{height}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')


for index, model_name in enumerate(activation_ratio.keys()):
    # 设置柱状图的宽度
    bar_width = 0.2

    # 设置模型 a 和模型 b 的激活值数据
    this_data = activation_ratio[model_name]

    # 绘制柱状图
    x = np.arange(1, len(this_data) + 1)

    num_layers = len(this_data)
    num_models = len(activation_ratio)
    if num_layers % 2 == 0:
        x_pos = x + (index - (num_models/2 - 1) - 0.5) * bar_width
    else:
        x_pos = x + (index - (num_models-1)/2) * bar_width

    this_bars = this_axes.bar(x_pos, this_data, bar_width, label=model_name, color=different_colors[model_name], zorder=10)

    autolabel(this_bars)


plt.setp(this_axes.spines.values(), linewidth=6)

this_axes.set_xticks(x)
this_axes.set_xticklabels(x_labels)

this_axes.spines['top'].set(linewidth=1, color="lightgray")
this_axes.spines['right'].set(linewidth=1, color="lightgray")

this_axes.tick_params(width=6, length=6)

this_axes.tick_params(labelsize=font_size, pad=15)

this_axes.set_ylabel("% of positive activation", fontsize=font_size + 10, labelpad=100)

this_axes.set_xlabel("Layer Index", fontsize=font_size + 10, labelpad=20)

this_axes.legend(fontsize=font_size)
# this_axes.legend(fontsize=60, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)

# this_axes.set_title(data_name, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})

# this_axes.grid(linestyle="-", color=(237 / 255.0, 237 / 255.0, 237 / 255.0), linewidth=1)
this_axes.grid(axis='y', linestyle="-", color="lightgray", linewidth=3.5)

# plt.sca(this_axes)
# plt.xticks([1, 2, 3], list(['1', '2', '3']), rotation='horizontal')

fig.savefig("0_intro_bar_fig.png")
fig.savefig("0_intro_bar_fig.pdf")