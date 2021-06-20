import sys, os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# hyperparameter
learning_rates = np.logspace(-1, -5, 5)
wds = np.logspace(-4, -8, 5)
batch_sizes = [16, 32, 64, 128, 256]
optimizers = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD']
schedulers = ([[None, None], ['StepLR', 'step_size=10, gamma=0.2'], ['ExponentialLR', 'gamma=0.9'], ['CosineAnnealingLR', "T_max=10, eta_min=param['lr']*0.01"], 
				['ReduceLROnPlateau', "'min'"]#, ['OneCycleLR', "max_lr=param['lr']*10, steps_per_epoch=args.epochs, epochs=args.epochs"]
				])

result_template = "../data/result/hw2/cifar-10_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}"
fig_template = "../data/result/hw2/figure/cifar-10_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}"

def plot_result(data, idx, title, row=3) :
	ax = plt.subplot(row, 5, idx)
	ax.plot(data)
	ax.grid()
	ax.set_ylim([0.1, 1.05])
	ax.set_title(title)

	return ax

# find optimal hyperparamete
optimal_parameter = pd.DataFrame([], columns = ['optimizer', 'scheduler', 'lr', 'wd', 'batch_size'])
default_param = {'lr':0.001, 'wd':1e-5, 'batch_size':64}
for optimizer in optimizers :
	for scheduler in schedulers :
		fig = plt.figure(figsize=(16,9))
		fig.suptitle("{}, {}".format(optimizer, scheduler[0]))
		m_lr, tmp = 0, 0
		for i, lr in enumerate(learning_rates) :
			result = pd.read_csv(result_template.format(lr, default_param['wd'], default_param['batch_size'], optimizer, scheduler[0])+'.csv')
			data = list(result.iloc[2][1:])
			plot_result(data, i+1, "lr={}".format(lr))
			if tmp < max(data) :
				tmp = max(data)
				m_lr = lr
		m_wd, tmp = 0, 0
		for i, wd in enumerate(wds) :
			result = pd.read_csv(result_template.format(default_param['lr'], wd, default_param['batch_size'], optimizer, scheduler[0])+'.csv')
			data = list(result.iloc[2][1:])
			plot_result(data, i+6, "wd={}".format(wd))
			if tmp < max(data) :
				tmp = max(data)
				m_wd = wd
		m_bs, tmp = 0, 0
		for i, batch_size in enumerate(batch_sizes) :
			result = pd.read_csv(result_template.format(default_param['lr'], default_param['wd'], batch_size, optimizer, scheduler[0])+'.csv')
			data = list(result.iloc[2][1:])
			plot_result(data, i+11, "batch size={}".format(batch_size))
			if tmp < max(data) :
				tmp = max(data)
				m_bs = batch_size
		optimal_parameter = optimal_parameter.append({'optimizer':optimizer, 'scheduler':scheduler[0], 'lr':m_lr, 'wd':m_wd, 'batch_size':m_bs}, ignore_index=True)
# 		plt.show()
		# plt.savefig(fig_template.format(m_lr, m_wd, m_bs, optimizer, scheduler[0]) + ".png", dpi=300)
		plt.close()
# optimal_parameter.to_csv("../data/result/hw2/optimal_parameter.csv")

# plot optimal parameter case
fig = plt.figure(figsize=(16,9))
fig.suptitle("result@optimal_parameter")

for i, param in optimal_parameter.iterrows() :
	# print(param['lr'])
	result = pd.read_csv(result_template.format(param['lr'], param['wd'], param['batch_size'], param['optimizer'], param['scheduler'])+'.csv')
	data = list(result.iloc[2][1:])
	# print(result)
	plot_result(data, i+1, "{}, {}".format(param['optimizer'], param['scheduler']), 6)
# plt.show()
plt.savefig("../data/result/hw2/figure/optimal_parameter.png", dpi=300)
plt.close()

# plot optimal result
if os.path.isfile("../data/result/hw2/optimal_result.csv") :
	optimal_result = pd.read_csv("../data/result/hw2/optimal_result.csv")
	data = np.zeros((len(optimizers), len(schedulers)))
	for i, result in optimal_result.iterrows() :
		data[i] = list(result)[1:]
	fig = plt.figure(figsize=(9,9))
	fig.suptitle("optimal_result")
	plt.imshow(data, cmap=plt.get_cmap('hot'))
	plt.colorbar()
	plt.xticks(np.arange(0, 5), labels=[str(x[0]) for x in schedulers])
	plt.yticks(np.arange(0, 6), labels=optimizers)
	# plt.show()
	plt.savefig("../data/result/hw2/figure/optimal_result.png", dpi=300)