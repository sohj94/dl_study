import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy

import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter 
from torchsummary import summary
import numpy as np
import pandas as pd

from dataset_builder import load_data_set
from trainer import Trainer

# import model
import torchvision.models as models
# from model.resnet import resnet50

parser = argparse.ArgumentParser()

parser.add_argument('--data', dest='data', default="cifar-10")
parser.add_argument('--result_dir', dest='result_dir', default="../data/result/hw2/")
parser.add_argument('--model_dir', dest='model_dir', default="../data/hw2/")
parser.add_argument('--epochs', dest='epochs', type=int, default=30)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', action='store_false', default=True)
parser.add_argument('--continue_train', dest='continue_train', action='store_true', default=False)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)
parser.add_argument('--skip_exist_train', dest='skip_exist_train', action='store_true', default=False)

args = parser.parse_args()

# torch 초기 설정
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print("set vars and device done")

# 경로 생성
if not os.path.isdir(args.model_dir) :
    os.makedirs(args.model_dir)

# writer = SummaryWriter('runs/alexnet')

# hyperparameter
learning_rates = np.logspace(-1, -5, 5)
wds = np.logspace(-4, -8, 5)
batch_sizes = [16, 32, 64, 128, 256]
optimizers = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD']
schedulers = ([[None, None], ['StepLR', 'step_size=10, gamma=0.2'], ['ExponentialLR', 'gamma=0.9'], ['CosineAnnealingLR', "T_max=10, eta_min=param['lr']*0.01"], 
				['ReduceLROnPlateau', "'min'"]#, ['OneCycleLR', "max_lr=param['lr']*10, steps_per_epoch=args.epochs, epochs=args.epochs"]
				])

# # Dataset, Dataloader 정의
train_dataset, test_dataset = load_data_set(args.data)
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
print(len(test_data.dataset))

# define model
model = models.resnet50(num_classes=10)
model.to(device)
criterion = nn.CrossEntropyLoss()

# check model
x = torch.randn(64, 3, 32, 32).to(device)
output = model(x)
summary(model, (3, 32, 32), device=device.type)

# hyperparameter setting
hyperparameter = []
default_param = {'lr':0.001, 'wd':1e-5, 'batch_size':64}
for optimizer in optimizers :
	for scheduler in schedulers :
		for lr in learning_rates :
			param = copy.deepcopy(default_param)
			param['lr'] = lr
			param['optimizer'] = optimizer
			param['scheduler'] = scheduler
			hyperparameter.append(param)
		for wd in wds :
			param = copy.deepcopy(default_param)
			param['wd'] = wd
			param['optimizer'] = optimizer
			param['scheduler'] = scheduler
			hyperparameter.append(param)
		for batch_size in batch_sizes :
			param = copy.deepcopy(default_param)
			param['batch_size'] = batch_size
			param['optimizer'] = optimizer
			param['scheduler'] = scheduler
			hyperparameter.append(param)

for param in hyperparameter :
	model = models.resnet50(num_classes=10).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = eval('optim.'+param['optimizer']+"(model.parameters(), lr=param['lr'], weight_decay=param['wd'])")
	scheduler = None if param['scheduler'][0] is None else eval('optim.lr_scheduler.'+param['scheduler'][0]+"(optimizer,"+param['scheduler'][1]+')')
	scheduler_flag = True if param['scheduler'][0] == 'ReduceLROnPlateau' else False

	print("learning rate: {}, weight decay: {}, batch size: {}, optimizer: {}, scheduler: {}"\
		.format(param['lr'], param['wd'], param['batch_size'], optimizer, scheduler))

	# set trainer
	trainer = Trainer(model, criterion, optimizer, [scheduler, scheduler_flag], args)

	# skip for existing result
	if (not args.skip_exist_train) and (os.path.isfile(args.model_dir + args.data + "_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}.pth".format( \
			param['lr'], param['wd'], param['batch_size'], param['optimizer'], param['scheduler'][0]))) :
		print("skip train")
		continue

	#train
	if args.train:
		if args.continue_train :
			# last_epoch = int(os.listdir(args.model_dir)[-1].split('epoch_')[1][:3])
			last_epoch = 30
			model.load_state_dict(torch.load(args.model_dir + args.data + "_width_{0:03}_depth_{1:03}.pth".format(width, depth)))
			# 그 다음 epoch부터 학습 시작
			trainer.fit(train_data, last_epoch+1)
		else :
			trainer.fit(train_data)
		torch.save(model.state_dict(), args.model_dir + args.data + "_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}.pth".format( \
			param['lr'], param['wd'], param['batch_size'], param['optimizer'], param['scheduler'][0]))
		model_history = pd.DataFrame([trainer.lr_history, trainer.loss_history, trainer.metric_history], index=["learning_rate", "loss", "metric"])
		print(model_history)
		model_history.to_csv(args.result_dir + args.data + "_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}.csv".format( \
			param['lr'], param['wd'], param['batch_size'], param['optimizer'], param['scheduler'][0]))
	else:
		# model.load_state_dict(torch.load(args.model_dir + args.data + "_width_{0:03}_depth_{1:03}.pth".format(width, depth)))
		model.load_state_dict(torch.load(args.model_dir + args.data + "_lr_{0:03}_wd_{1:03}_batch_{2:03}_{3}_{4}.pth".format( \
			param['lr'], param['wd'], param['batch_size'], param['optimizer'], param['scheduler'][0])))

	# torch.save(model.state_dict(), args.model_dir + args.data + "_width_{0:03}_depth_{1:03}.pth".format(width, depth))
	accuracy = trainer.test(test_data)
	# print("accuracy of model with width {} depth {}: {}".format(width, depth, accuracy))
	print("accuracy of model: {}".format(accuracy))

	# result = pd.DataFrame(accuracies)
	# result.to_csv(args.result_dir, index = False)