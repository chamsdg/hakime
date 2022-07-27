## Importation des libraries
import joblib
#import uvicorn
from flask import Flask, jsonify
import pandas as pd
import pickle
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# path data
PATH = 'dataset/'


# data = pd.read_csv('dataset/test_dataframe.csv')
data = pd.read_parquet('test_df.parquet')
print('la taille de Dataframe est = ', data.shape)

# size of DataFrame

print('la taille de la nouvelle Dataframe est = ', data.shape)

# Chargement du modèle

model = pickle.load(open('ModelClassifier.pkl', 'rb'))



# app flask
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello Everyone ,  This is my Web API for credit scoring "


@app.route('/credit/<id_client>')
def credit(id_client):

    print('id client = ', id_client)
    
    #Récupération des données du client en question
    
    ID = int(id_client)
    X = data[data['SK_ID_CURR'] == ID]
    
    ## Elimination des features non important et la variable Cible 
    ##   'SK_ID_CURR'    : 
    ##   'INDEX'         :   Index de la dataframe  automatique
    ##   'TARGET'        :  variable cible 
    
    notimportant_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
    selected_features = [col for col in data.columns if col not in notimportant_features]
    
    # Declaration de vecteur X
    X = X[selected_features]
    
    print('La taille du vecteur X  = ', X.shape)
    
    proba = model.predict_proba(X)
    prediction = model.predict(X)
    
 
    #DEBUG
    
    print('L''identificateur du client : ', id_client)
  
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0]) 
        }
   
    print('Lancer une nouvelle Prédiction : \n', dict_final)
    

     # Sauvegarder le résultat sous forme de JSON file 
        
    return jsonify(dict_final)


#  lancement de l'application   (mode local  et non en mode production) 
if __name__ == "__main__":
    app.debug = True
    app.run()
