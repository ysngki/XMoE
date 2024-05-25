from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl

ticklabelpad = mpl.rcParams['xtick.major.pad']

# seed=42
activation_ratio = {
	'GPT-L12': [20, 19, 18, 17, 16, 15, 14],
}


different_colors = {
	'GPT-L12': (9 / 255.0, 147 / 255.0, 150 / 255.0),
	'FedPF': (238 / 255.0, 155 / 255.0, 0 / 255.0),
	'FedAP': (174 / 255.0, 32 / 255.0, 18 / 255.0),
	'FedBF': 'r',
	'FedFT': (0 / 255.0, 48 / 255.0, 225 / 255.0),
	# 'FedFT': 'b',
}

different_marks = {
	'GPT-L12': '^',
	'FedPF': 'D',
	'FedAP': 'x',
	'FedBF': 'o',
	'FedFT': 's',
}

font_size = 80

my_nrows = 1
my_ncols = 1

fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, figsize=(60, 40))
fig.subplots_adjust(hspace=0.75)

this_axes = axes

for index, model_name in enumerate(activation_ratio.keys()):
	width = 0.25

	this_axes.plot(range(1, 7, 2), activation_ratio[model_name], marker=different_marks[model_name],
				   markersize=50, label=model_name, linewidth=10, color=different_colors[model_name])

plt.setp(this_axes.spines.values(), linewidth=6)
this_axes.spines['top'].set(linewidth=1, color="lightgray")
this_axes.spines['right'].set(linewidth=1, color="lightgray")

this_axes.tick_params(width=6, length=6)

this_axes.tick_params(labelsize=font_size, pad=15)

this_axes.set_ylabel("% of positive activation", fontsize=font_size, labelpad=30)

this_axes.set_xlabel("Layer", fontsize=font_size, labelpad=0)

this_axes.legend(fontsize=font_size, framealpha=0.9)
this_axes.legend(fontsize=60, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)

# this_axes.set_title(data_name, fontdict={'fontsize': font_size, 'horizontalalignment': 'center'})

this_axes.grid(linestyle="-", color="lightgray", linewidth=1)

# plt.sca(this_axes)
# plt.xticks([1, 2, 3], list(['1', '2', '3']), rotation='horizontal')

fig.savefig("epoch_exp.png")