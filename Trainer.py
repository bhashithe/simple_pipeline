from abc import ABC, abstractmethod

class Trainer(ABC):
	#def __init__(self, model, loader, optimizer, loss_function, scheduler, device, epochs):
	def __init__(self, *args, **kwargs):
		""" Instantiate a Trainer object
		Args:
			model: This is a torch.nn.Module implemented object
			loader: Dictionary with torch.util.data.DataLoader object as values, keys must be ['train', 'val']
			optimizer: torch.optim type optimizer
			loss_function: any torch criterion function typically, torch.nn.CrossEntropy()
			scheduler: torch.optim.lr_scheduler type object
			device: torch.device
		"""

		super().__init__()
		self.model = model
		self.loader = loader
		self.optimizer = optimizer
		self.loss_funtion = loss_function
		self.scheduler = scheduler
		self.device = device
		self.epochs = epochs

	@abstractmethod
	def train(self):
		""" Should train and validate the model in the given number of epochs
		Args:
			*DEPRECATED* epochs: the number of epochs the network should train
		
		Returns:
			torch.nn.Module type object which the trains instantiated with
			Good to note that this should return the best model after training/validation
		"""
		pass
