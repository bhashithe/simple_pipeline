import torch
import argparse 
import importlib

class Pipeline():

	def arguments(self):
		parser = argparse.ArgumentParser(description='PyTorch NN training pipeline')
		parser.add_argument('parameters', help='Python file to use in the operation mode', type=str)
		parser.add_argument('--trainer', help='Extended trainer class', type=str)
		parser.add_argument('--generate', help='Operation mode: generate a parameters file or a model file using this filename')
		
		args = parser.parse_args()
		return args.parameters,args.trainer, args.generate

	def generate_model(self, filename='Model.py'):
		cont = """\
import torch

class Model(torch.nn.Module)
	def __init__(self):
		super.__init__(Model, self)

	def forward(self):
		pass

"""
		with open(filename+'.py', 'w') as f:
			f.write(cont)
		
		return True

	def generate_parameters(self, filename='parameters.py'):
		"""
		This will generate a parameters file where it could be used to do the training. The file will have the following  structure.

		A training_parameter() function which does not take arguments, but returns a dictionary with all the required parameters to train

		Arguments:
			filename: path and the name of the file, extention should be .py
		"""
		
		content = """\
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

model = #Add nn.Module implemented classifier

trns = # torchvision.transfomrs you want to add if there is any

dataset_size = {'train': 7000, 'val': 2144} # add your own sizes
dataset = # load torch.util.data.Dataset type data here so that you have access to random_split and DataLoaders
batch_size = 20
data = {}
data['train'], data['val'] = random_split(dataset, (dataset_size['train'], dataset_size['val']))
loader = {phase: DataLoader(data[phase],batch_size=batch_size) for phase in ['train','val']}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def training_parameters():
	parameters = {
		'model': model,
		'optimizer': Adam(model.parameters()),
		'scheduler': StepLR(optimizer, step_size=7),
		'loader': loader,
		'loss_function': torch.nn.CrossEntropyLoss(),
		'device': device,
		'epochs': 25
	}

	return parameters

if __name__ == '__main__':
	training_parameters()
"""

		with open(filename+'.py', 'w') as f:
			f.write(content)

		print('file written')
		return True

def main():
	p = Pipeline()
	parameters, trainer, generate = p.arguments()

	if generate == 'parameters':
		p.generate_parameters(parameters)

	if generate == 'model':
		p.generate_model(parameters)

	if not generate:
		module = importlib.import_module(parameters)
		params = getattr(module, 'training_parameters')
		parameters = params()

		module = importlib.import_module(trainer)
		trainer = getattr(module, trainer)
		trainer = trainer(parameters)

		trainer.train()

if __name__ == '__main__':
	main()
