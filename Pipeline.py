import torch
import argparse 
import importlib

class Pipeline():

	def arguments(self):
		parser = argparse.ArgumentParser(description='PyTorch NN training pipeline')
		parser.add_argument('parameters', help='Python file to use in the operation mode', type=str)
		parser.add_argument('trainer', help='Extended trainer class', type=str)
		parser.add_argument('--generate', help='generate parameters file with defaults')
		
		args = parser.parse_args()
		return args.parameters,args.trainer, args.generate

	def generate_model(self, filename='Model.py'):
		pass

	def generate_parameters(self, filename='parameters.py'):
		"""
		This will generate a parameters file where it could be used to do the training. The file will have the following  structure.

		A training_parameter() function which does not take arguments, but returns a dictionary with all the required parameters to train

		Arguments:
			filename: path and the name of the file, extention should be .py
		"""
		pass

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
