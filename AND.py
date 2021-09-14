from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np


AND = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,0,0,1]
}
df = pd.DataFrame(AND)
df



X,y = prepare_data(df)
eta = 0.3 #learning rate should be between 0 and 1
epochs = 10
model_AND = Perceptron(eta=eta, epochs=epochs)
model_AND.fit(X, y)
loss = model_AND.total_loss