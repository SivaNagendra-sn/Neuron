"""
    author : Siva Nagendra
    email : Sivanagendra.paruchuri@gmail.com
"""
from utils.all_utils import prepare_data, save_model, save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os

log_string = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    # levelname - Info or exception
    # module - name of the module where the issue has raised eg.
    # message - message that we want to log
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "runninglogs.log"), level=logging.INFO, format=log_string, filemode="a")
    # filemode - 'a' - to append all the log info to previous logs (By default also it will give the same)

def main(data , eta, epochs, filename, plot_file_name):

    df = pd.DataFrame(data)

    X,y = prepare_data(df)
    logging.info(f"This is the actual dataframe {df}")
    ETA = eta # 0 and 1
    EPOCHS = epochs

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename)
    save_plot(df, plot_file_name, model)

if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.3
    Epochs = 100
    file_name = "or.model"
    plot_file_name = "or.png"
    try:
        logging.info(">>>>> Starting  Training >>>>>>>>>")
        main(data=OR, eta=ETA, epochs=Epochs, filename= file_name, plot_file_name= plot_file_name)
        logging.info(">>>>> Training Done Succesfully >>>>>>>>> \n")
    except Exception as e:
        logging.exception(e)
        raise e # To print exception in terminal output