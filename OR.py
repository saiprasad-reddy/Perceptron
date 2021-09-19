"""
author : Saiprasad Reddy
email   : saiprasadreddy@gmail.com
"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotfinename):
        
        df = pd.DataFrame(data)
        df

        X,y = prepare_data(df)
        model = Perceptron(eta=eta, epochs=epochs)
        model.fit(X, y)
        loss = model.total_loss

        save_model(model, filename=filename)
        save_plot(df, plotfinename, model)
if __name__ == '__main__':
        
        OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[1,1,1,0]
        }
        eta = 0.3 #learning rate should be between 0 and 1
        epochs = 10
        main(data=OR, eta=eta, epochs=epochs, filename="OR.model", plotfinename="OR.png")
