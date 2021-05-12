import pandas as pd
import numpy as np
import torch
from torch import nn
import time
from tqdm import tqdm

class Trainer:
	def __init__(self, model, criterion, optimizer, args):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.args = args

	def fit(self, train_data, last_epoch=0):
		self.model.train()
		for epoch in range(self.args.epochs):
			epoch_loss = 0.
			for iter, (image, label) in tqdm(enumerate(train_data)) :
				# tmp = time.time()
				image, label = image.cuda(), label.cuda()
				pred = self.model(image)
				loss = self.criterion(input=pred, target=label)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.detach().item()
				# print("Ran in {} seconds".format(time.time() - tmp))
				# print('epoch : {0} step : [{1}/{2}] loss : {3}'.format(epoch+last_epoch, iter, len(train_data), loss.detach().item()))
			epoch_loss /= len(train_data)
			print('\nepoch : {0} epoch loss : {1}\n'.format(epoch+last_epoch, epoch_loss))

			# torch.save(self.model.state_dict(), self.args.model_dir + "epoch_{0:03}.pth".format(epoch+last_epoch))

	def test(self, test_data):
		self.model.eval()
		# result = pd.read_csv(self.args.test_csv_dir)
		correct = 0
		for iter, (image, label) in tqdm(enumerate(test_data)):
			image = image.cuda()
			label = label.numpy()
			pred = self.model(image)
			pred = nn.Softmax(dim=1)(pred)
			pred = pred.detach().cpu().numpy()
			answer_id = np.argmax(pred, axis=1)
			confidence = pred[0, answer_id]
			if answer_id == label :
				correct += 1
		accuracy = correct / len(test_data)
		print("accuracy: {}".format(accuracy))
			# submission.loc[iter, 'answer_id'] = answer_id
			# submission.loc[iter, 'conf'] = confidence
		# result.to_csv(self.args.test_csv_submission_dir, index=False)

		return accuracy