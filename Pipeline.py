import torch

import argparse

class Pipeline():

	def arguments(self):
		parser = argparse.ArgumentParser(description='PyTorch NN training pipeline')
		parser.add_argument('parameters', help='parameters JSON file', type=str)
		parser.add_argument('--generate', help='generate parameters file with defaults')
		
		args = parser.parse_args()
		return args.parameters, args.generate

	def generate_parameters(self, filename='parameters.json'):
		pass

def main():
	p = Pipeline()
	parameters, generate = p.arguments()

	if generate:
		p.generate_parameters(generate)

if __name__ == '__main__':
	main()
