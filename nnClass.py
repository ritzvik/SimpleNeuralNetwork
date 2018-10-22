import numpy as np

class NN:
  def __init__ (self, inputLayerSize, layerSizes = (64,64,10)):
    self.weights = [np.random.normal(0,0.001,inputLayerSize*layerSizes[0]).reshape((inputLayerSize, layerSizes[0]))]
    for i in range(1,len(layerSizes)):
      self.weights.append(np.random.normal(0,0.001,layerSizes[i-1]*layerSizes[i]).reshape((layerSizes[i-1], layerSizes[i])))
    #
    self.biases = [np.random.normal(0,0.0001,size) for size in layerSizes]
    self.layerSizes = layerSizes
    self.outputVals = [np.zeros(size) for size in layerSizes]
    self.initialize_derivatives()

  def sigmoidnp(self, npvector):
    return 1/(1+np.exp(-npvector))

  def feedforward(self, inputVector):
    self.outputVals[0] = self.sigmoidnp(np.dot(inputVector, self.weights[0])+self.biases[0])
    for i in range(1,len(self.weights)):
      self.outputVals[i] = self.sigmoidnp(np.dot(self.outputVals[i-1], self.weights[i])+self.biases[i])
    return self.outputVals[-1]

  def getcost(self, label):
    sigma = 0
    for i in range(len(self.outputVals[-1])):
      if i != label:
        sigma += (self.outputVals[-1][i])**2
      else:
        sigma += (self.outputVals[-1][i]-1)**2
    return sigma

  def initialize_derivatives(self):
    self.dw = [np.zeros(npvector.shape) for npvector in self.weights]     # derivative of weights
    self.db = [np.zeros(npvector.shape) for npvector in self.biases]      # derivative of biases
    self.do = [np.zeros(npvector.shape) for npvector in self.outputVals]  # derivative of outputs
    self.cdw = [np.zeros(npvector.shape) for npvector in self.weights]    # cummilative derivative of weights
    self.cdb = [np.zeros(npvector.shape) for npvector in self.biases]     # cummilative derivative of biases

  def backpropagate(self, inputVector, label):
    # update derivative vector of output layer
    self.do[-1] = -2*self.outputVals[-1]
    self.do[-1][label] = -2*(self.outputVals[-1][label]-1)
    # update derivatives of weights and biases
    for i in range(len(self.weights)-1,-1,-1):
      if i>0:
        self.dw[i] = np.outer(self.outputVals[i-1], self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      else:
        self.dw[i] = np.outer(inputVector, self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      self.db[i] = self.do[i]*self.outputVals[i]*(1-self.outputVals[i])
      if i>0:
        self.do[i-1] = np.matmul(self.weights[i],self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      # update cummilative weights
      self.cdw[i] += self.dw[i]
      self.cdb[i] += self.db[i]

  def backpropagate_part(self, inputVector, output_dy):
    self.do[-1] = output_dy
    for i in range(len(self.weights)-1,-1,-1):
      if i>0:
        self.dw[i] = np.outer(self.outputVals[i-1], self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      else:
        self.dw[i] = np.outer(inputVector, self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      self.db[i] = self.do[i]*self.outputVals[i]*(1-self.outputVals[i])
      if i>0:
        self.do[i-1] = np.matmul(self.weights[i], self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      else:
        self.dinp = np.matmul(self.weights[0], self.do[i]*self.outputVals[i]*(1-self.outputVals[i]))
      self.cdw[i] += self.dw[i]
      self.cdb[i] += self.db[i]
    return self.dinp

  def updateWeightsBiases(self, alpha=0.02):
    for i in range(len(self.weights)):
      self.weights[i] = self.weights[i] + alpha*self.cdw[i]
      self.cdw[i].fill(0)
      self.biases[i] = self.biases[i] + alpha*self.cdb[i]
      self.cdb[i].fill(0)
