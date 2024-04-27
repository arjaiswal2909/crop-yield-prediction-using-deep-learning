from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler_x.pkl', 'rb') as f:
    scaler_x = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', prediction=None, input_data=None)
    elif request.method == 'POST':
        state = request.form['state']
        season = request.form['season']
        area = float(request.form['area'])
        
        input_df = pd.DataFrame(columns=[
            'State_Andaman and Nicobar Islands', 'State_Andhra Pradesh', 'State_Arunachal Pradesh',
            'State_Assam', 'State_Bihar', 'State_Chandigarh', 'State_Chhattisgarh', 'State_Dadra and Nagar Haveli',
            'State_Daman and Diu', 'State_Delhi', 'State_Goa', 'State_Gujarat', 'State_Haryana', 'State_Himachal Pradesh',
            'State_Jammu and Kashmir ', 'State_Jharkhand', 'State_Karnataka', 'State_Kerala', 'State_Madhya Pradesh',
            'State_Maharashtra', 'State_Manipur', 'State_Meghalaya', 'State_Mizoram', 'State_Nagaland', 'State_Odisha',
            'State_Puducherry', 'State_Punjab', 'State_Rajasthan', 'State_Sikkim', 'State_Tamil Nadu', 'State_Telangana ',
            'State_THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU', 'State_Tripura', 'State_Uttar Pradesh', 'State_Uttarakhand',
            'Season_Kharif','Season_Autumn', 'Season_Rabi','Season_Total', 'Season_Summer', 'Season_Winter','Season_WholeYear'
        ])
        
        input_df.loc[0] = 0 
        
        input_df.loc[0, f'State_{state}'] = 1
        input_df.loc[0, f'Season_{season}'] = 1

        input_data = np.concatenate((input_df.values, np.array([[area]])), axis=1)

        input_data_scaled = scaler_x.transform(input_data)

        prediction_scaled = model.predict(input_data_scaled)[0]

        prediction = scaler_y.inverse_transform(np.array([[prediction_scaled]]))
        prediction_value = str(prediction[0][0])

        return prediction_value

if __name__ == '__main__':
    app.run(debug=True)
