from application_logging.logger import App_Logger
from ModelFinder import finder
import pandas as pd
class prediction:

    def __init__(self):
        self.file_read = open("ModelLogs/Model_Predict.txt", 'a+')
        self.log_writer = App_Logger()

    def predict_results(self,data):
        try:
            self.log_writer.log(self.file_read, 'Start of Prediction')
            predictions = []
            model_load=finder.ModelFinder()
            model = model_load.load_model()

            result=(model.predict(data))

            self.log_writer.log(self.file_read, 'Model loaded successfully')
            for res in result:
                if res == 0:
                    predictions.append('<=50K')
                else:
                    predictions.append('>50K')

            final = pd.DataFrame(list(predictions), columns=['Predictions'])
            #path = "Prediction_Output_File/Predictions.csv"
            #final.to_csv(path, header=True,mode='a+')  # appends result to prediction file
            #self.log_writer.log(self.file_read, 'End of Prediction')
            return final
        except Exception as ex:
            self.log_writer.log(self.file_read, 'Error occurred while running the prediction!! Error:: %s' % ex)


