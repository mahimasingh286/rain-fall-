from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

def train_model(months, rainfall):
    X = np.array(months, dtype=float).reshape(-1, 1)
    y = np.array(rainfall, dtype=float)
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_rainfall(model, month):
    future_month = np.array([[month]], dtype=float)
    prediction = model.predict(future_month)
    return future_month, prediction

def plot_rainfall(months, rainfall, future_month, prediction):
    try:
        prediction = np.array(prediction)
        
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        axs[0, 0].scatter(months, rainfall, color='blue', label='Actual Rainfall (Input Data)', marker='o')
        axs[0, 0].scatter(future_month, prediction, color='green', label='Predicted Rainfall (Next Month)', s=100)
        axs[0, 0].plot(np.append(months, future_month), np.append(rainfall, prediction.flatten()), linestyle='dashed', color='red', label='Trend Line')
        axs[0, 0].axvline(x=1, color='gray', linestyle='--', label='Prediction Boundary')
        axs[0, 0].set_xlabel('Month')
        axs[0, 0].set_ylabel('Rainfall (mm)')
        axs[0, 0].set_title('Rainfall Prediction for Next Month')
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', linewidth=0.6)

    
        axs[0, 1].hist(rainfall, bins=10, color='orange', edgecolor='black')
        axs[0, 1].set_title('Distribution of Rainfall Data')
        axs[0, 1].set_xlabel('Rainfall (mm)')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].grid(True, linestyle='--', linewidth=0.6)

        
        axs[1, 0].plot(months, rainfall, marker='o', color='purple', linestyle='-')
        axs[1, 0].set_title('Rainfall Trend Over Time')
        axs[1, 0].set_xlabel('Month')
        axs[1, 0].set_ylabel('Rainfall (mm)')
        axs[1, 0].grid(True, linestyle='--', linewidth=0.6)

       
        axs[1, 1].boxplot(rainfall, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'))
        axs[1, 1].set_title('Box Plot of Rainfall Variability')
        axs[1, 1].set_ylabel('Rainfall (mm)')

        
        plot_path = 'static/rainfall_analysis.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        return plot_path
    except Exception as e:
        print(f"An error occurred while plotting: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        num_months = int(data['months'])
        rainfall = list(map(float, data['rainfall']))
        

        months = list(range(1, num_months + 1))
        model = train_model(months, rainfall)
        
    
        future_month, prediction = predict_rainfall(model, month=num_months + 1)
        
    
        plot_path = plot_rainfall(months, rainfall, future_month, prediction)
        
        return jsonify({
            'predictedRainfall': prediction[0],
            'imageUrl': plot_path
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'There was an error processing the request.'}), 500

if __name__== '__main__':
    app.run(host='0.0.0.0', port=5000 )