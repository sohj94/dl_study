import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

from dataset_builder import load_data_set
from trainer import Trainer

# import model
from model.hw1_model import hw1_model

parser = argparse.ArgumentParser()

parser.add_argument('--data', dest='data', default="cifar-10")
parser.add_argument('--result_dir', dest='result_dir', default="../data/result/hw1_result_cifar-10.csv")
parser.add_argument('--model_dir', dest='model_dir', default="../data/temp/")
parser.add_argument('--epochs', dest='epochs', type=int, default=20)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', action='store_false', default=True)
parser.add_argument('--continue_train', dest='continue_train', action='store_true', default=False)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)

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

# # Dataset, Dataloader 정의
train_dataset, test_dataset = load_data_set(args.data)
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

accuracies = []
for width in range(10,151,10) :
	tmp_accuracy = []
	for depth in range(3,16) :

		model = hw1_model(input_size = torch.numel(train_dataset[0][0]), width = width, depth = depth)
		model.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)

		# set trainer
		trainer = Trainer(model, criterion, optimizer, args)

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
		else:
			model.load_state_dict(torch.load(args.model_dir + args.data + "_width_{0:03}_depth_{1:03}.pth".format(width, depth)))

		torch.save(model.state_dict(), args.model_dir + args.data + "_width_{0:03}_depth_{1:03}.pth".format(width, depth))
		accuracy = trainer.test(test_data)
		print("accuracy of model with width {} depth {}: {}".format(width, depth, accuracy))

		tmp_accuracy.append(accuracy)
	accuracies.append(tmp_accuracy)
accuracies = np.array(accuracies)
print(accuracies)

result = pd.DataFrame(accuracies)
result.to_csv(args.result_dir, index = False)