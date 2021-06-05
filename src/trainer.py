import pandas as pd
import numpy as np
import torch
from torch import nn
import time
from tqdm import tqdm

class Trainer:
	def __init__(self, model, criterion, optimizer, scheduler = None, args = None):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.args = args
		self.scheduler = scheduler
		self.loss_history = []
		self.metric_history = []
		self.lr_history = []

	def fit(self, train_data, last_epoch=0):
		self.model.train()
		start_time = time.time()
		for epoch in range(self.args.epochs):
			current_lr = get_lr(self.optimizer)
			print('Epoch {}/{}, current lr={}'.format(epoch, self.args.epochs, current_lr))
			train_loss, train_metric = loss_epoch(self.model, self.criterion, train_data, False, self.optimizer)
			self.loss_history.append(train_loss)
			self.metric_history.append(train_metric)
			self.lr_history.append(current_lr)
			if self.scheduler is not None :
				self.scheduler.step(train_loss)
			print('train loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, 100*train_metric, (time.time()-start_time)/60))
			print('-'*10)

			# epoch_loss = 0.
			# for iter, (image, label) in tqdm(enumerate(train_data)) :
			# 	# tmp = time.time()
			# 	image, label = image.cuda(), label.cuda()
			# 	pred = self.model(image)
			# 	loss = self.criterion(input=pred, target=label)
			# 	self.optimizer.zero_grad()
			# 	loss.backward()
			# 	self.optimizer.step()
			# 	epoch_loss += loss.detach().item()
			# 	# print("Ran in {} seconds".format(time.time() - tmp))
			# 	# print('epoch : {0} step : [{1}/{2}] loss : {3}'.format(epoch+last_epoch, iter, len(train_data), loss.detach().item()))
			# epoch_loss /= len(train_data)
			# print('\nepoch : {0} epoch loss : {1}\n'.format(epoch+last_epoch, epoch_loss))

			# torch.save(self.model.state_dict(), self.args.model_dir + "epoch_{0:03}.pth".format(epoch+last_epoch))

	def test(self, test_data):
		self.model.eval()
		# result = pd.read_csv(self.args.test_csv_dir)
		_, accuracy = loss_epoch(self.model, self.criterion, test_data)

		# correct = 0
		# for iter, (image, label) in tqdm(enumerate(test_data)):
		# 	image = image.cuda()
		# 	label = label.numpy()
		# 	pred = self.model(image)
		# 	pred = nn.Softmax(dim=1)(pred)
		# 	pred = pred.detach().cpu().numpy()
		# 	answer_id = np.argmax(pred, axis=1)
		# 	confidence = pred[0, answer_id]
		# 	if answer_id == label :
		# 		correct += 1
		# accuracy = correct / len(test_data)
		print("accuracy: {}".format(accuracy))
			# submission.loc[iter, 'answer_id'] = answer_id
			# submission.loc[iter, 'conf'] = confidence
		# result.to_csv(self.args.test_csv_submission_dir, index=False)

		return accuracy

def get_lr(opt) :
	for param_group in opt.param_groups :
		return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, target) :
	pred = output.argmax(1, keepdim=True)
	corrects = pred.eq(target.view_as(pred)).sum().item()
	return corrects


# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None) :
	loss = loss_func(output, target)
	metric_b = metric_batch(output, target)

	if opt is not None :
		opt.zero_grad()
		loss.backward()
		opt.step()

	return loss.item(), metric_b

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None) :
	running_loss = 0.0
	running_metric = 0.0
	len_data = len(dataset_dl.dataset)

	for xb, yb in tqdm(dataset_dl) :
		xb = xb.cuda()
		yb = yb.cuda()
		output = model(xb)

		loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

		running_loss += loss_b

		if metric_b is not None:
			running_metric += metric_b

		if sanity_check is True:
			break

	loss = running_loss / len_data
	metric = running_metric / len_data

	return loss, metric