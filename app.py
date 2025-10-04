from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form values
        company = request.form['company']
        type_name = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Resolution → calculate PPI
        X_res = int(resolution.split('x')[0])
        y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

        # Prepare dataframe for model
        query_df = pd.DataFrame([{
            'Company': company,
            'TypeName': type_name,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen,
            'Ips': ips,
            'ppi': ppi,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'Gpu brand': gpu,
            'os': os
        }])

        # Prediction
        prediction = round(int(np.exp(pipe.predict(query_df)[0])))

        return render_template('index.html', result=f"₹{prediction:,}",
                               companies=df['Company'].unique(),
                               types=df['TypeName'].unique(),
                               cpus=df['Cpu brand'].unique(),
                               gpus=df['Gpu brand'].unique(),
                               oss=df['os'].unique()
                               )

    # Initial GET request
    return render_template('index.html',
                           companies=df['Company'].unique(),
                           types=df['TypeName'].unique(),
                           cpus=df['Cpu brand'].unique(),
                           gpus=df['Gpu brand'].unique(),
                           oss=df['os'].unique()
                           )

if __name__ == "__main__":
    app.run(debug=True)

