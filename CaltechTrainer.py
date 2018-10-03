import torch
from Trainer import Trainer
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import time
import copy

class CaltechTrainer(Trainer):
	def __init__(self, params):
		self.model = params.get('model')
		self.loader = params.get('loader')
		self.optimizer = params.get('optimizer')
		self.loss_function = params.get('loss_function')
		self.device = params.get('device')
		self.model = self.model.to(self.device)
		self.scheduler = params.get('scheduler')
		self.epochs = params.get('epochs')

		if params.get('device') == torch.device('cuda:0'):
			print('model in gpu')
			self.model.cuda()

	def train(self):
		"""
		adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
		"""
		since = time.time()

		model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.00
		
		for epoch in range(self.epochs):
			print('Epoch : {}/{}'.format(epoch, self.epochs-1))
			print('-'*10)

			for phase in ["train","eval"]:
				if phase=="train":
					self.model.train()
					self.scheduler.step()
				else:
					model.eval()
		
				running_loss = 0.0
				running_corrects = 0

				for batch_idx, data in enumerate(self.loader[phase],0):
					images, labels = data
					images = images.to(self.device)
					labels = labels.to(self.device)
					
					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(images)
						_, preds = torch.max(outputs, 1)
						loss = self.loss_function(outputs, labels)

						if phase == "train":
							loss.backward()
							self.optimizer.step()
					running_loss += loss.item() * images.size(0)
					running_corrects += torch.sum(preds == labels.data)

					torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, 'model_checkpoint.mdl')

			epoch_loss = running_loss/len(self.loader[phase])
			epoch_corrects = running_corrects.double()/len(self.loader[phase])

			print("{} epoch_loss: {:.3f} epoch_acc: {:.3f}".format(phase, epoch_loss, epoch_corrects))

			if phase == "eval" and epoch_corrects > best_acc:
				best_acc = epoch_corrects
				best_model_wts = copy.deepcopy(self.model.state_dict())

			print()

		time_elapsed = time.time() - since
		print("training completed in: {:.f}m {:.f}s".format(time_elapsed//60, time%60))

		print("Best accuracy: {:.4f}".format(best_acc))

		model.load_state_dict(best_model_wts)
		torch.save(self.model, 'model_complete.mdl')
		return model
