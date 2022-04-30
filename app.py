from flask import Flask ,render_template, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pyodbc
from sys import path
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

model = pickle.load(open('naive_bayes_model.pkl','rb'))

id3_model = pickle.load(open('id3_model.pkl', 'rb'))

# fit the OrdinalEncoder to encode data entries before passing to the model
# using same dataset used to train the model
def load_dataset(filename):
    data = read_csv(filename)
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X = X.astype(str)
    y = y.reshape((len(y), 1))
    return X, y

X, y = load_dataset('heart_disease_male(csv).csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = np.nan)
oe.fit(X_train)


# prepare the SQL server analysis services connection
path.append('\\Program Files\\Microsoft.NET\\ADOMD.NET\\150')
from pyadomd import Pyadomd
model_name = 'MultidimensionalProject1'
port_number = 'DESKTOP-EM0Q92G'
connection_string = f'Provider=MSOLAP;Data Source={port_number};Catalog={model_name};'


# building the home page
@app.route('/' ,methods=['POST'])
def home():
    
    feachures = [str(x) for x in request.form.values()]
    final_feachures = np.array([feachures])
    prediction = []
    df = pd.DataFrame()
    Algorithm = 'Algorithm'
    SSAS = 'SSAS'
    Coded = 'Coded'
    Bayes = 'Bayes'
    Id3 = 'Id3'
    SSAS_Bayes = ''
    Coded_Bayes = ''
    SSAS_Id3 = ''
    Coded_Id3 = ''
    if final_feachures.size != 0:
        entries2 = oe.transform(final_feachures)
        entries3 = np.nan_to_num(entries2)
        prediction = model.predict_proba(entries3)
        id3_prediction = id3_model.predict(entries3)
        prediction = prediction * 100
        id3_prediction = id3_prediction * 100

        dmx_query = f"""
        select flattened

        predicthistogram(disease)

        from bayesAlltextMM

        natural prediction join
        (select '{final_feachures[0][0]}' as Age,
        '{final_feachures[0][3]}' as Blood_Sugar,
        '{final_feachures[0][1]}' as Chest_Pain_Type,
        '{final_feachures[0][6]}' as Excercise_Angina,
        '{final_feachures[0][5]}' as Max_Heart_Rate,
        '{final_feachures[0][2]}' as Rest_Blood_Pressure,
        '{final_feachures[0][4]}' as Rest_electro) as t
        """

        dmx_query2 = f""" 
        SELECT
        [id3id3MM].[Disease]
        From
        [id3id3MM]
        NATURAL PREDICTION JOIN
        (SELECT '{final_feachures[0][0]}' AS [Age],
        '{final_feachures[0][3]}' AS [Blood Sugar],
        '{final_feachures[0][1]}' AS [Chest Pain Type],
        '{final_feachures[0][6]}' AS [Exercice Angina],
        '{final_feachures[0][5]}' AS [Max Heart Rate],
        '{final_feachures[0][2]}' AS [Rest Blood Pressure],
        '{final_feachures[0][4]}' AS [Rest Electro]) AS t
        """

        con = Pyadomd(connection_string)
        con.open()

        result = con.cursor().execute(dmx_query)
        df = pd.DataFrame(result.fetchone())
        df[[2]] = df[[2]]*100
        con.close()

        con.open()
        result_id3 = con.cursor().execute(dmx_query2)
        df_id3 = pd.DataFrame(result_id3.fetchone())
        
        con.close()

        SSAS_Bayes = f" {df.at[0, 0]} = {df.at[0, 2]} \n {df.at[1, 0]}  = {df.at[1, 2]} \n {df.at[2, 0]}  = {df.at[2, 2]} "
        SSAS_Id3 = f" {df.at[0, 0]}  "
        Coded_Bayes = f" Positive = {prediction[0][1]} \n Negative = {prediction[0][0]} "
        if id3_prediction == [0] :
            Coded_Id3 = 'Negative'
        else :
            Coded_Id3 = 'Positive'

    return render_template("index.html", Algorithm = Algorithm, Bayes = Bayes, Id3 = Id3, 
                            SSAS = SSAS, Coded = Coded , SSAS_Bayes = SSAS_Bayes, Coded_Bayes = Coded_Bayes, 
                            SSAS_Id3 = SSAS_Id3, Coded_Id3 = Coded_Id3)


if __name__ == '__main__':
    app.run(debug = True)
