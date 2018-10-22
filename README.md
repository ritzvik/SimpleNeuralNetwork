# SimpleNeuralNetwork
## Dense Neural Network Implementation using only Numpy

### hdigits.npz
  numpy.save_compressed version of MNIST handwriiten digits dataset. Usage :
  ```
npzfile = np.load('hdigits.npz')
train_images = npzfile['mnist_train_images']
train_labels = npzfile['mnist_train_labels']
test_images = npzfile['mnist_test_images']
test_labels = npzfile['mnist_test_labels']
  ```
  arrays would be of shape (n,1,28,28) where n denotes number of instances, 1 is number of channels in images, and 28 * 28 is dimensions for train and test images<br />
  labels are 1D numpy array
  
### nnClass.py
  Contains the class defination of dense neural network having sigmoid activation function.<br />
  Can be directly used or used to make components of bigger, more complex networks
  
### train.py
  Trains on MNIST train dataset and saves the network using pickle.
  
### test.py
  Tests MNIST test dataset on the model made by train.py and prints accuracy.
