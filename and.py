    """
    author : Siva Nagendra
    email : Sivanagendra.paruchuri@gmail.com
    
    """
from utils.all_utils import prepare_data, save_model, save_plot
from utils.model import Perceptron
import pandas as pd

def main(data , eta, epochs, filename, plot_file_name):
    df = pd.DataFrame(AND)

    X,y = prepare_data(df)

    ETA = eta # 0 and 1
    EPOCHS = epochs

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()


    save_model(model, filename)
    save_plot(df, plot_file_name, model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3
    Epochs = 10
    file_name = "and.model"
    plot_file_name = "and.png"
    main(data=AND, eta=ETA, epochs=Epochs, filename= file_name, plot_file_name= plot_file_name)