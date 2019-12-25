#step -1 # Importing flask module in the project is mandatory 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features=[3])

#Step -2 Flask constructor takes the name of  
# current module (__name__) as argument.app = Flask(__name__)

app = Flask(__name__)

#Step -3 Load Trained  Model
model = pickle.load(open('Multilinearregression', 'rb'))
onehotencoder  = pickle.load(open('Multilinearregression_O', 'rb'))
labelencoder  = pickle.load(open('Multilinearregression_L', 'rb'))

# Step -4 The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #Data = (float(x) for x in request.form.values())
    research=request.form['research']
    adm=request.form['adm']
    digital=request.form['digital']
    region=request.form['region']
    
    Data = {'Research':[research],'Office Administation':[adm],'Digital Marketing':[digital],'Region':[region]}
    print(Data)
   
  
   
    
    df1 = pd.DataFrame(Data,columns=['Research','Office Administation','Digital Marketing','Region'])
    print(df1)
    x = df1.values
    print(x)
    x[:,3] = labelencoder.transform(x[:,3])
    print(x)
    x = onehotencoder.transform(x).toarray()
    x = x[:,1:]
    print(x)
    output = model.predict(x)

    
   
    return render_template('index.html', prediction_text=' ROI profit is {}'.format(output))


# main driver function
 # run() method of Flask class runs the application  
    # on the local development server.
if __name__ == "__main__":
    app.run(debug=True)

