import os
import pandas as pd
from flask import Flask, request, render_template,Response
from flask_cors import cross_origin,CORS
import ReadData
from DataSplitting import DataSplitting
from Training_Data_Traonsfrmation import Preprocessing
import numpy as np
from ModelFinder.finder import ModelFinder
from predictionGetData import Data_Getter_Pred
from PredictFromModel import prediction
from wsgiref import simple_server

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    try:
        if request.form is not None:
            path=request.form['filepath']
            data_obj=Data_Getter_Pred(path)
            data=data_obj.get_data()
            preprocess = Preprocessing.Preprocessor()
            X = preprocess.removeExtraSpace(data)
            columns_with_null_values, is_null_present = preprocess.columnsWithMissingVlaue(X)
            if (is_null_present):
                X = preprocess.imputeMissingValue(columns_with_null_values,X)
            preprocess.removeUnwantedFeatures(X, ['education'])
            X = preprocess.computeOutliars(X, 'fnlwgt')
            X = preprocess.outliarsCompute(X, 'hours-per-week')
            df_numeric = preprocess.scaleDownNumericFeatures(X)
            df_category = preprocess.encodeCategoryFeatures(X)
            df = pd.concat([df_numeric, df_category], axis=1)
            predict=prediction()
            path=predict.predict_results(df)
            #path=predict.predict_results(df)
            path=pd.concat([data, path], axis=1)
            path.to_html("Table.htm")
            html_file =path.to_html()
            return Response(html_file)

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
        #print("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
        #print(("Error Occurred! %s" %KeyError))
    except Exception as e:
        return Response("Error Occurred! %s" %e)
        #print(("Error Occurred! %s" %e))




@app.route('/train',methods=['POST'])
@cross_origin()
def train_predict():
    try:
        readdataobj = ReadData.ReadData()
        data=readdataobj.readData()
        '''datasplitobj=DataSplitting(data)
        X_train, X_test, y_train, y_test = datasplitobj.split_data()'''

        preprocess=Preprocessing.Preprocessor()
        data=preprocess.removeExtraSpace(data)
        X, y = preprocess.seperateDependentIndependentColumns(data)
        y=y.map({'<=50K':0,'>50K':1})
        columns_with_null_values , is_null_present=preprocess.columnsWithMissingVlaue(X)
        if(is_null_present):
            X=preprocess.imputeMissingValue(columns_with_null_values,X)
        preprocess.removeUnwantedFeatures(X , ['education'])
        X = preprocess.computeOutliars(X, 'fnlwgt')
        X = preprocess.outliarsCompute(X, 'hours-per-week')
        df_numeric=preprocess.scaleDownNumericFeatures(X)
        df_category=preprocess.encodeCategoryFeatures(X)
        
        df=pd.concat([df_numeric,df_category], axis=1)
        X,y = preprocess.handleImbalancedDataSet(df,y)
        split_data = DataSplitting()
        X_train, X_test, y_train, y_test = split_data.split_data(X,y)
        
        modelfinder=ModelFinder()
        modelfinder.get_best_model(X_train, X_test, y_train, y_test)


    except ValueError:
        print("Value error occurred %s" %ValueError)

    except KeyError:
        print("Key error occurred %s" %KeyError)

    except Exception as e:
        print("Error Occurred while training %s" %e)



if __name__=='__main__':
    app.run(debug=True,threaded=True)
