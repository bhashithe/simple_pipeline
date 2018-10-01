# Simple Pipeline

This is a encapsulation where it is possible to train different kind of neural networks on the same training logic which you have came up previously. You need 3 things

- `torch.nn.Module` implemented neural network
- `torch.util.data.DataLoader` object to load the data batch wise
- `Trainer` object which you have to implement extending the provided abstract class

# To do
- [x] Create Pipeline class
	- [ ] Add functionality so that it can load a parameter JSON file to load all the required modules
	- [ ] Can take command line argument of the file name to load the parameters
	- [ ] Can generate a skeleton parameter file
- [ ] Create sample code
