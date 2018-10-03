# Simple Pipeline

This is a encapsulation where it is possible to train different kind of neural networks on the same training logic which you have came up previously. You need 3 things

- `torch.nn.Module` implemented neural network
- `Trainer` object which you have to implement extending the provided abstract class
- `torch.util.data.DataLoader` object to load the data batch wise

You need to implement the above two classes and then create a script where you combine everything together with a dictionary. A sample script is given below.

``` python
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
```

Now this script can be invoked with our pipeline to encapsulate training.

``` bash
python Pipeline.py parameters trainer
```

Even though the parameter and trainer files are python files, you are supposed to omit the file name. Here `Caltech101` dataset has been used as an example, I have implemented a CNN classifier (`CaltechClassifier`) and the training logic (`CaltechTrainer`) which we are going to train the model on. Therefore my pipeline call would look like this,

``` bash
python Pipeline.py parameters CaltechTrainer
```

Notice here that I have not changed the parameter files name

# To do
- [x] Create Pipeline class
	- [x] Add functionality so that it can load a parameter python file to load all the required modules
	- [x] Can take command line argument of the file name to load the parameters
	- [ ] Can generate a skeleton parameter file
- [x] Create sample code
- [ ] Pause and Continue training
- [ ] Give a list of parameter files to select best model out of them
