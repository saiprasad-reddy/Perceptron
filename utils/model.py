import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3)*10e-4 #Sample eights
    print(f"Initial weights are being trained : \n{self.weights}")
    self.eta = eta #learning rate
    self.epochs = epochs #number of iterations

  def activationfunction(self, inputs, weights):
    z = np.dot(inputs, weights) #z gives dot product of X*W
    return np.where(z>0,1,0) #condition if true retirn 1 else 0
  
  def fit(self, X, y):
    self.X = X
    self.y = y
    X_with_bais = np.c_[self.X,-np.ones((len(self.X), 1))] #Concatinating bais with inputs
    print(f"Input after bais : \n{X_with_bais}")
    for epoch in range(self.epochs):
      print("**"*10)
      print(f"for epoch : {epoch+1}")
      print("**"*10)
      y_hat = self.activationfunction(X_with_bais, self.weights) #forward propagation
      print(f"Predicted value after forward pass : \n{y_hat}")
      self.error = self.y-y_hat
      print(f"Error is : \n{self.error}")
      self.weights = self.weights + self.eta*np.dot(X_with_bais.T,self.error) #backward propagation
      print(f"Updated weights after : \n{epoch}/{self.epochs} \n{self.weights}")
      print("##"*10)
  
  def predict(self, X):
    X_with_bais = np.c_[self.X,-np.ones((len(self.X), 1))]
    return self.activationfunction(X_with_bais, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total Loss : \n{total_loss}")
    return total_loss