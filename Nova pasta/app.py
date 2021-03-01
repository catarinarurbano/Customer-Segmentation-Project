import os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#PROJECT_ROOT = Path(os.path.abspath('')).resolve().parents[0]

app = Flask(__name__)  # Initialize the flask App
#model = load(os.path.join(PROJECT_ROOT, 'analysis', 'knn_for_final_model.pkl'))
#model = load(os.path.join(PROJECT_ROOT, 'knn_for_final_model.pkl'))
model = load('knn_for_final_model.pkl') 
scaler= load('scaler.pkl') 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = list(request.form.values())[0].split(',')
    final_features = [np.array(int_features)]
    
    #scaling features:
    final_features= scaler.transform(final_features)
    
    prediction = model.predict(final_features)

    if (str(prediction)=='[0]'): #cluster 0
        output='The customer belongs to the group of Loyal Customers. \nThe marketing strategy should be: Offer Loyalty Cards'
    if (str(prediction)=='[1]'): #cluster 1
        output='The customer belongs to the group of Champions Customers. \nThe marketing strategy should be: Social Events and Wine Tasting Sessions'
    if (str(prediction)=='[2]'): #cluster 2
        output='The customer belongs to the group of Almost Lost Customers. \nThe marketing strategy should be: Social Media giveaways and online discounts'
    return render_template('index.html', prediction_text=output)
    #return render_template('index.html', prediction_text='The customer belongs to the cluster '+str(prediction))
if __name__ == "__main__":
    app.run(debug=True)