import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

returnable = {
	'optimizer': Adam(model.parameters()),
	'scheduler': StepLR(optimizer),
	'loader': loader,
	'loss_function': loss_function,
	'device': device
}
