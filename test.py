import numpy as np
import _pickle as pickle 
from nnClass import NN


def main():
  npzfile = np.load('hdigits.npz')
  te = npzfile['mnist_test_images']
  tel = npzfile['mnist_test_labels']
  flat_te = te.flatten().reshape((len(te),28*28))

  filehandler = open('model.pkl','rb')
  network = pickle.load(filehandler)
  filehandler.close()

  count=0
  for instance,label in zip(flat_te,tel):
    prediction = np.argmax(network.feedforward(instance))
    if (label==prediction):
      print('Y')
      count += 1
    else:
      print('N')

  print(100*count/len(flat_te),'%')


if __name__ == '__main__':
  main()
