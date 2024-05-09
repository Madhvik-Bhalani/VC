from flask import Flask, request, render_template,jsonify
from Models.predict_new_data import predict_new_data
from Models.age_group_data_analysis.age_group_data import calculate_avg_data
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def Home():
    return render_template("index.html", title="Home | DiabetIQ Insight")


@app.route("/predict_diabetes", methods=["GET", "POST"])
def predict_diabetes():
    if request.method == "GET":
        return render_template("predict_diabetes.html", title="Assess Your Diabetes | DiabetIQ Insight")
    
    if request.method == "POST":
        # convert form value into array
        features = [float(x) for x in request.form.values()]

        # make 2D array for Standard Scaller
        f_features = np.array(features).reshape(1, -1)

        # make predication with multiple model
        predicted_data = predict_new_data(f_features)

        # calculate age group wise avg data
        age = int(request.form.get("age"))
        age_group_avg_data = calculate_avg_data(age)

        return render_template("predict_diabetes.html",title="Assess Your Diabetes | DiabetIQ Insight",datas=[predicted_data, features, age_group_avg_data])


@app.route("/explore_dataset", methods=["GET"])
def explore_dataset():
    data = pd.read_csv("Dataset/diabetes.csv")
    return render_template("dataset.html", title="Explore Dataset | DiabetIQ Insight", datas=data)


@app.route("/trained_models", methods=["GET"])
def trained_models():
    return render_template("trained_models.html", title="Trained Models | DiabetIQ Insight")

# APIs for Predicting Diabetes and Displaying Dataset

@app.route("/api/dataset", methods=['GET'])
def dataset_api():
    datas = pd.read_csv('dataset/diabetes.csv')
    data_dict = datas.to_dict(orient='Records')
    return jsonify(data_dict)


@app.route("/api/predict-diabetes", methods=['POST'])
def predict_api():
    features = [float(x) for x in request.form.values()]
    f_features = np.array(features).reshape(1, -1)
    predicted_data = predict_new_data(f_features)
    
    age = int(request.form.get("age"))
    age_group_avg_data = calculate_avg_data(age)
    
    result_response = [{'Model': item[0], 'Result': float(item[1])} for item in predicted_data]
    avg_data_response = [{'Factor': item[0], 'Age Group Avg Data': item[1]} for item in age_group_avg_data[:-1]]
    user_data = [{'Your Data':item} for item in features]
    
    merged_data = [{'Factor': avg_data['Factor'], 'Age Group Avg Data': avg_data['Age Group Avg Data'], 'Your Data': user_data[index+1]['Your Data']}
               for index, avg_data in enumerate(avg_data_response)]

    
    return jsonify(result_response + merged_data)

if __name__ == "__main__":
    app.run()
