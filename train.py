import numpy as np
from nnClass import NN
import _pickle as pickle

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

def main():
  npzfile = np.load('hdigits.npz')
  tr = npzfile['mnist_train_images']
  trl = npzfile['mnist_train_labels']
  batchsize = 1
  flat_tr = tr.flatten().reshape((len(tr),28*28))
  flat_tr, trl = unison_shuffled_copies(flat_tr, trl)

  count=0
  network=NN(inputLayerSize = 28*28, layerSizes = (256,128,64,10))
  for i,instance in enumerate(flat_tr):
    network.feedforward(instance)
    network.backpropagate(instance, trl[i])
    count += 1
    if count >= batchsize:
      network.updateWeightsBiases(alpha=0.05)
      count=0
    print(i,trl[i],network.getcost(trl[i]))


  filehandler = open('model.pkl','wb')
  pickle.dump(network, filehandler)
  filehandler.close()

if __name__ == '__main__':
  main()