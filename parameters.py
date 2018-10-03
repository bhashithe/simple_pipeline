import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from CaltechClassifier import CaltechClassifier
from torchvision import transforms

model = CaltechClassifier(102)

trns = transforms.Compose([transforms.Resize([224, 224]), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])

image_data = ImageFolder('101_ObjectCategories', transform=trns)
dataset_size = {'train': 7000, 'val': 2144}
batch_size = 20
data = {}
data['train'], data['val'] = random_split(image_data, (dataset_size['train'], dataset_size['val']))
loader = {phase: DataLoader(data[phase],batch_size=batch_size) for phase in ['train','val']}

optimizer = Adam(model.parameters())

loss_function = torch.nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def training_parameters():
	parameters = {
		'model': model,
		'optimizer': Adam(model.parameters()),
		'scheduler': StepLR(optimizer, step_size=7),
		'loader': loader,
		'loss_function': loss_function,
		'device': device,
		'epochs': 25
	}

	return parameters

if __name__ == '__main__':
	training_parameters()
