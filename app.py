from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
pd.options.display.max_rows = 200   # show 200 rows


app = Flask(__name__)

df = pd.read_csv('Dataset/final.csv')   # use forward slashes
# OR (recommended for cross-platform)

df = pd.read_csv(os.path.join("Dataset", "final.csv"))


gb = joblib.load("Models/Gradient_Boosting.pkl")
lr = joblib.load("Models/Linear_Regression.pkl")
rf = joblib.load("Models/Random_Forest.pkl")
xg = joblib.load("Models/XGBoost.pkl")

lst = [gb, lr, rf, xg]
# print(lst)
# lst is showing as None [ Need to check why ]

@app.route("/")
def index():
    return render_template("index.html")   

@app.route("/submit", methods=["POST"])
def submit():
    option = request.form.get('Option')

    # labels = {
    #     "DA_Charts": "Data Analytics Charts",
    #     "US_Map": "Best Colleges Locations [ USA ]",
    #     "Prediction": "Predict Placement Status"
    # }

    # option_txt = labels.get(option)

    # return f"<h1>{option_txt} option is selected :)</h1>"


    if option == 'DA_Charts':
        return render_template("Data_Analytics.html")
    elif option == 'US_Map':
        return redirect(url_for("US"))
    elif option == 'Prediction':
        return render_template("Predictions.html")

@app.route('/DA', methods=["POST", "GET"])
def DA():

   
    check = request.form.get("Charts")   # now works!
    if check == 'GPST':
        plt.figure(figsize=(10, 6))
        sns.histplot(x='gender', data=df, hue='placement_status', palette={'Placed': 'blue', 'Not Placed': 'red'},
                      multiple='dodge')
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.show()
    
    elif check == 'PCIS':
        fig = px.pie(data_frame=df, names='stream', opacity=0.9, title='Stream-wise student distribution')
        fig.update_traces(
            textinfo='percent',
            textfont=dict(size=15, color='black', family='Arial')
        )
        fig.update_traces(
            textinfo='percent',
            textfont=dict(size=18, color='black', family='Arial Black'),  # Bold & thicker
        )

        fig.update_layout(
            title=dict(
                text="Stream-wise student distribution",
                font=dict(size=22, family="Arial Black", color="black")   # Bold Title
            )
        )
        fig.show()

    elif check == 'SPSD':
        
        plt.figure(figsize=(10, 10))
        ax = sns.histplot(
            x='stream',
            data=df,
            hue='placement_status',
            multiple='dodge',
            shrink=0.8
        )

        plt.xlabel("Stream")
        plt.ylabel("Count")


        for p in ax.patches:
            height = p.get_height()
            if height > 0:  
                ax.text(
                    p.get_x() + p.get_width() / 2,  
                    height,                         
                    int(height),                    
                    ha='center', va='bottom', fontsize=10
                )
        plt.legend()

        plt.show()

    elif check == 'PSDA':

        plt.figure(figsize=(12, 6))
        sns.histplot(x='location', data=df, hue='placement_status', multiple='dodge', shrink=0.8)
        plt.xticks(rotation=45)
        plt.xlabel("Location")
        plt.ylabel("Count")
        plt.show()

    else:
        return "<h3>No valid chart option selected</h3>"

    return render_template("Data_Analytics.html")   # <-- load second HTML page


@app.route('/US', methods=["GET", "POST"])
def US():
    df_filtered = df[df['placement_status'] == 'Placed']

    # City coordinates dictionary
    city_coords = {
        "Berkeley": (37.8715, -122.2730),
        "Ann Arbor": (42.2808, -83.7430),
        "Los Angeles": (34.0522, -118.2437),
        "Urbana-Champaign": (40.1106, -88.2272),
        "Washington": (38.9072, -77.0369),
        "College Park": (38.9897, -76.9378),
        "Boulder": (40.01499, -105.2705),
        "Rochester": (43.1566, -77.6088),
        "Santa Cruz": (36.9741, -122.0308),
        "Connecticut": (41.6032, -73.0877),
        "Delaware": (38.9108, -75.5277),
        "San Francisco": (37.7749, -122.4194),
        "Dallas": (32.7767, -96.7970),
        "Virginia": (37.4316, -78.6569),
        "Riverside": (33.9533, -117.3961),
        "Pennsylvania": (41.2033, -77.1945),
        "Chapel Hill": (35.9132, -79.0558)
    }

    
# Compute counts only for cities in city_coords
    city_counts = df_filtered['location'].value_counts().to_dict()

    data = {"City": [], "Lat": [], "Lon": [], "Count": []}
    for city, count in city_counts.items():
        if city in city_coords:
            lat, lon = city_coords[city]
            data["City"].append(city)
            data["Lat"].append(lat)
            data["Lon"].append(lon)
            data["Count"].append(count)


    df_map = pd.DataFrame(data)

    fig = px.scatter_geo(
        df_map,
        lat="Lat",
        lon="Lon",
        hover_name="City",           # <-- show city name on hover
        hover_data={"Lat": True, "Lon": True, "Count": True},  
        size_max=50,  # max marker size
        color_continuous_scale=px.colors.sequential.Viridis,
        scope="usa"
    )

    # Optional: style land, lakes, and borders
    fig.update_geos(
        showland=True,
        landcolor="lightgreen",   # land appears light brown
        lakecolor="lightblue",
        showcountries=True,
        countrycolor="black"
    )

    fig.update_layout(
        width=1000,
        height=500
    )
   
    fig.update_traces(
        marker=dict(
            size=12,         # size of the circle
            color='yellow',   # fill color (can be transparent using 'rgba(0,0,0,0)')
            line=dict(width=2, color='blue')  # border thickness and color
        )
    )


    # Convert figure to HTML div
    graph_html = fig.to_html(full_html=False)

    return render_template("USA.html", graph_html=graph_html)

@app.route("/Predict", methods=["POST"])
def predict():
    # Example: form fields 'feature1', 'feature2', ...
    Gender = int(request.form.get("gender"))
    # Degree = float(request.form.get("degree"))
    Stream = float(request.form.get("stream"))
    Age = float(request.form.get("age"))
    College = float(request.form.get("college_name"))
    Status = float(request.form.get("placement_status"))
    GPA = float(request.form.get("gpa"))
    YOE = float(request.form.get("years_of_experience"))

    # id,name,gender,age,degree,stream,college_name,placement_status,salary,gpa,years_of_experience
    
    # Create input DataFrame for model
    input_data = pd.DataFrame([[Gender, Age,  Stream, College, Status, GPA, YOE]], 
                              columns=["gender", "age", "stream", "college_name", "placement_status", "gpa", 
                                       "years_of_experience"])
    
    predictions = []
    for model in lst:
        pred = model.predict(input_data)[0]  # take the first element
        predictions.append(pred)


    model_names = ["Gradient Boosting", "Linear Regression", "Random Forest", "XGBoost"]

    pred_dict = {name: pred for name, pred in zip(model_names, predictions)}
    return render_template("FinalOP.html", predictions=pred_dict)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000 for local dev
    app.run(host="0.0.0.0", port=port, debug=True)




