from flask import Flask, jsonify
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import joblib

from sklearn.preprocessing import LabelEncoder

from haversine import haversine, Unit

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import base64
import json

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Used to find depth, well type and returns Map also
@app.route('/depthWellAdvanced/<float:user_lat>/<float:user_lon>')
def depthWellAdvanced(user_lat, user_lon):
    def create_well_recommendation_map(user_latitude, user_longitude, threshold_distance=3):
        # Load the dataset
        file_path = "dugwell.csv"  # Update this with the correct file path
        df = pd.read_csv(file_path)

        # Remove rows with missing values
        df = df.dropna(subset=['Y', 'X', 'Depth (m.bgl)', 'Well Type'])

        # Extract features (X) and target variables (y_depth, y_well_type)
        X = df[['Y', 'X']]
        y_depth = df['Depth (m.bgl)']
        y_well_type = df['Well Type']

        # Create and fit the KNN regression model for depth prediction
        depth_model = KNeighborsRegressor(n_neighbors=3)
        depth_model.fit(X, y_depth)

        # Create and fit the KNN classification model for well type prediction
        well_type_model = KNeighborsClassifier(n_neighbors=3)
        well_type_model.fit(X, y_well_type)

        # Function to recommend well type and find the nearest dugwell
        def recommend_well_and_nearest(user_lat, user_lon):
            user_location = [[user_lat, user_lon]]

            # Find the nearest dugwell
            nearest_dugwell_index = depth_model.kneighbors(user_location)[1][0][0]
            nearest_dugwell_coordinates = X.iloc[nearest_dugwell_index]
            nearest_dugwell_depth = y_depth.iloc[nearest_dugwell_index]
            nearest_dugwell_well_type = y_well_type.iloc[nearest_dugwell_index]

            # Check if the nearest dugwell is within the threshold distance
            distance_to_nearest_dugwell = geodesic(user_location[0], nearest_dugwell_coordinates).kilometers
            if distance_to_nearest_dugwell > threshold_distance:
                return f"No suitable well within {threshold_distance} km.", None, None

            return nearest_dugwell_depth, nearest_dugwell_well_type, nearest_dugwell_coordinates

        # Create a folium map centered at the user's location
        map_center = [user_latitude, user_longitude]
        map_object = folium.Map(location=map_center, zoom_start=12)

        # Get the recommendation and nearest well for the user's location
        recommendation_result = recommend_well_and_nearest(user_latitude, user_longitude)
        recommended_depth, recommended_well_type, recommended_coordinates = recommendation_result

        if isinstance(recommended_well_type, str) and recommended_well_type != "No suitable well within 3 km.":
            # Add a marker for the recommended well at the user's location
            folium.Marker(location=[user_latitude, user_longitude],
                        popup=f"Recommended Depth: {recommended_depth} meters, Recommended Well Type: {recommended_well_type}",
                        icon=folium.Icon(color='red')).add_to(map_object)
            
            # Add a marker for the nearest well
            folium.Marker(location=[recommended_coordinates['Y'], recommended_coordinates['X']],
                        popup=f"Nearest Well - Depth: {recommended_depth} meters, Well Type: {recommended_well_type}",
                        icon=folium.Icon(color='green')).add_to(map_object)
        else:
            # If not suitable, display a marker at the user's location with a message
            folium.Marker(location=[user_latitude, user_longitude],
                        popup="No suitable well within 3 km.",
                        icon=folium.Icon(color='gray')).add_to(map_object)

        return map_object, recommended_depth, recommended_well_type

    # Example usage
    user_latitude = user_lat
    user_longitude = user_lon

    map_object, recommended_depth, recommended_well_type = create_well_recommendation_map(user_latitude, user_longitude)

    # Simple print output of prediction
    print(f"Recommended Depth: {recommended_depth} meters")
    print(f"Recommended Well Type: {recommended_well_type}")

    # Save the map as an HTML file
    map_object.save("well_recommendation_map.html")

    # # reading that html file
    with open('well_recommendation_map.html', 'r') as file:
        # Read the contents of the file into a string
        html_content = file.read()


    result = {
        'depth': f"{recommended_depth}",
        'well_type': f"{recommended_well_type}",
        'html_content': f"{html_content}"
    }

    return jsonify(result)

# Used to find drillingTechnic and formation
@app.route('/drillingTechnic/<float:user_lat>/<float:user_lon>')
def drillingTechnic(user_lat, user_lon):
    
    # Load the Aquifer data from an Excel file
    file_path = 'Aquifer_data_Cuddalore.xlsx'
    aquifer_df = pd.read_excel(file_path)

    # Drop rows with missing values in the 'FORMATION' column
    aquifer_df = aquifer_df.dropna(subset=['FORMATION', 'Y_IN_DEC', 'X_IN_DEC'])

    # Encode categorical labels ('HardRock' and 'SoftRock') into numerical values
    label_encoder = LabelEncoder()
    aquifer_df['FORMATION'] = label_encoder.fit_transform(aquifer_df['FORMATION'])

    # Features and target variable
    X = aquifer_df[['Y_IN_DEC', 'X_IN_DEC']]
    y = aquifer_df['FORMATION']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    # Function to predict 'FORMATION' based on user input location
    def predict_formation(user_latitude, user_longitude):
        user_location = [[user_latitude, user_longitude]]

        # Find the nearest aquifer location
        nearest_aquifer_index = knn_classifier.kneighbors(user_location)[1][0][0]
        nearest_aquifer_formation = label_encoder.inverse_transform([y.iloc[nearest_aquifer_index]])[0]

        return nearest_aquifer_formation

    # Function to suggest drilling technique based on 'FORMATION'
    def suggest_drilling_technique(formation):
        if formation == 'SR':
            return "SoftRock formation suggests Rotary drilling technique."
        elif formation == 'HR':
            return "HardRock formation suggests Down the hole drilling technique."

    # Take user input for location
    user_latitude = user_lat
    user_longitude = user_lon

    # Predict 'FORMATION' based on user input
    predicted_formation = predict_formation(user_latitude, user_longitude)

    # Display the result and drilling technique suggestion
    print(f"The predicted 'Aquifer type for the given coordinates is: {predicted_formation}")

    if predicted_formation == 'SR':
        print(suggest_drilling_technique(predicted_formation))
        pf = suggest_drilling_technique(predicted_formation)
    elif predicted_formation == 'HR':
        print(suggest_drilling_technique(predicted_formation))
        pf = suggest_drilling_technique(predicted_formation)
    else:
        pf = "Not found"
        print("Not found")

    result = {
        'formation': f"{predicted_formation}",
        'drilling_technic': f"{pf}"
    }

    return jsonify(result)

# Used to find water quality (ONLY FOR CHLORIDE)
@app.route('/waterQualityChloride/<float:user_lat>/<float:user_lon>')
def waterQualityChloride(user_lat, user_lon):
    # Load the modified water quality dataset
    csv_file_path = 'Modified_Water_Quality_Shuffled.csv'
    df_water_quality = pd.read_csv(csv_file_path)

    # Take user input for the test location (latitude and longitude)
    user_latitude = user_lat
    user_longitude = user_lon
    test_location = (user_latitude, user_longitude)

    # Define the threshold distance in kilometers
    threshold_distance_km = 3.0

    # Calculate distances to all wells
    distances = []
    for index, row in df_water_quality.iterrows():
        well_location = (row['Latitude'], row['Longitude'])
        distance = haversine(test_location, well_location, unit=Unit.KILOMETERS)
        distances.append(distance)

    # Filter wells within the threshold distance
    nearby_wells_indices = np.where(np.array(distances) <= threshold_distance_km)[0]

    if len(nearby_wells_indices) > 0:
        # Calculate the mean chloride level for nearby wells
        mean_chloride = df_water_quality.iloc[nearby_wells_indices]['Chloride_mg_per_l'].mean()
        print(f"The predicted chloride level is: {mean_chloride:.2f} mg/l")
        chloride_level = f"The predicted chloride level is: {mean_chloride:.2f} mg/l"
    else:
        print(f"Not able to predict the chloride level")
        chloride_level = f"Not able to predict the chloride level"

    result = {
        'chloride_level': f"{chloride_level}"
    }

    return jsonify(result)

# Used to find water quality (EC, TDS, CHLORIDE)
@app.route('/waterQualityChlorideAll/<float:user_late>/<float:user_lon>')
def waterQualityChlorideAll(user_late, user_lon):
    # Load the dataset
    file_path = "UpdatedWaterQuality.csv"
    df_water_quality = pd.read_csv(file_path)

    # User input for latitude and longitude
    user_lat = user_late
    user_long = user_lon

    # User location
    user_location = (user_lat, user_long)

    # Define the columns of interest
    columns_of_interest = ['EC_1', 'F_1', 'EC_2', 'F_2', 'EC_3', 'F_3', 'EC_4', 'F_4']

    # Define the threshold distance in kilometers
    threshold_distance_km = 3.0

    # Initialize dictionaries to store nearby values and TDS for each column
    nearby_values = {}
    tds_values = {}
    means = {}

    # Calculate distances to all wells
    for column in columns_of_interest:
        nearby_column_values = []

        for index, row in df_water_quality.iterrows():
            well_location = (row['Y'], row['X'])
            distance = haversine(user_location, well_location, unit=Unit.KILOMETERS)

            if distance <= threshold_distance_km:
                nearby_column_values.append(row[column])

        # Calculate and print the mean value for the current column
        mean_value = np.mean(nearby_column_values)
        print(f"Mean {column} value of nearby wells: {mean_value:.2f}")
        means.update({
            f"{column}": f"value of nearby wells: {mean_value:.2f}"
        }
        )

        # Store the nearby values for the current column in the dictionary
        nearby_values[column] = nearby_column_values

        # Calculate TDS based on the mean EC value for 'EC_1', 'EC_2', 'EC_3', and 'EC_4'
        if column.startswith('EC'):
            aquifer_number = int(column.split('_')[1])
            tds_column = f'TDS_{aquifer_number}'
            tds_values[tds_column] = mean_value * 0.67

    # Print TDS values for all relevant aquifers
    tds = {}
    print("\nTDS values for Aquifers 1, 2, 3, and 4:")
    for aquifer_number in range(1, 5):
        tds_column = f'TDS_{aquifer_number}'
        print(f"TDS value for Aquifer {aquifer_number}: {tds_values.get(tds_column, 0):.2f}")
        tds.update({
            f"TDS Value of Aquifer {aquifer_number}": f"{tds_values.get(tds_column, 0):.2f}"
        })

    result = {
        'means' : means,
        'tds' : tds
    }

    return jsonify(result)

# Used to find depthOfWaterBearing
@app.route('/depthOfWaterBearing/<float:user_lat>/<float:user_lon>')
def depthOfWaterBearing(user_lat, user_lon):
    # Function to find three nearest aquifer locations and calculate average depth
    def find_three_nearest_aquifers_and_average_depth(file_path, formation_column, top_column, bottom_column, user_latitude, user_longitude):
        # Load the Aquifer data from an Excel file
        aquifer_df = pd.read_excel(file_path)

        # Drop rows with missing values in the specified columns
        aquifer_df = aquifer_df.dropna(subset=['FORMATION', 'Y_IN_DEC', 'X_IN_DEC', top_column, bottom_column])

        # Encode categorical labels into numerical values
        label_encoder = LabelEncoder()
        aquifer_df['FORMATION'] = label_encoder.fit_transform(aquifer_df['FORMATION'])

        # Features and target variable
        X = aquifer_df[['Y_IN_DEC', 'X_IN_DEC']]
        y = aquifer_df['FORMATION']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)

        # Function to find the three nearest aquifer locations
        def find_three_nearest_aquifers():
            user_location = [[user_latitude, user_longitude]]

            # Find the three nearest aquifer locations
            nearest_aquifer_indices = knn_classifier.kneighbors(user_location, n_neighbors=3)[1][0]
            nearest_aquifer_data = aquifer_df.iloc[nearest_aquifer_indices]

            return nearest_aquifer_data

        # Function to calculate the average depth for the three nearest aquifer locations
        def calculate_average_aquifer_depth(nearest_aquifer_data, column_name):
            average_depth = nearest_aquifer_data[column_name].mean()
            return average_depth

        # Find the three nearest aquifers
        nearest_aquifer_data = find_three_nearest_aquifers()

        # Calculate the average depth of specified columns for the three nearest aquifers
        average_top_depth = calculate_average_aquifer_depth(nearest_aquifer_data, top_column)
        average_bottom_depth = calculate_average_aquifer_depth(nearest_aquifer_data, bottom_column)

        # Display the results
        print(f"{formation_column} water bearing zone is: {average_top_depth} meters - {average_bottom_depth} meters")

        return f"{formation_column} water bearing zone is: {average_top_depth} meters - {average_bottom_depth} meters"


    # Use the same latitude and longitude for all zones
    user_latitude = user_lat
    user_longitude = user_lon

    # Run the program for each water bearing zone
    result = {
        'first' : find_three_nearest_aquifers_and_average_depth('Aquifer_data_Cuddalore.xlsx', 'First', 'Aq_I_top_Rl (m.amsl)', 'Aq_I_Bottom_RL (m.amsl)', user_latitude, user_longitude),
        'second' : find_three_nearest_aquifers_and_average_depth('Aquifer_data_Cuddalore.xlsx', 'Second', 'Aq_II_top_Rl (m.amsl)', 'Aq_II_Bottom_RL (m.amsl)', user_latitude, user_longitude),
        'third' : find_three_nearest_aquifers_and_average_depth('Aquifer_data_Cuddalore.xlsx', 'Third', 'Aq_III_top_Rl (m.amsl)', 'Aq_III_Bottom_RL (m.amsl)', user_latitude, user_longitude),
        'forth' : find_three_nearest_aquifers_and_average_depth('Aquifer_data_Cuddalore.xlsx', 'Fourth', 'Aq_IV_top_Rl  (m.amsl)', 'Aq_IV_top_Rl  (m.amsl)', user_latitude, user_longitude)
    }

    return jsonify(result)

# Used to find Water Discharge
@app.route('/waterDischarge/<float:user_lat>/<float:user_lon>')
def waterDischarge(user_lat, user_lon):
    def train_and_predict(file_path, target_columns, user_latitude, user_longitude):
        # Load the dataset
        df = pd.read_excel(file_path)

        predictions = {}

        for target_column in target_columns:
            # Select relevant columns
            selected_columns = ['Y_IN_DEC', 'X_IN_DEC', target_column]
            df_selected = df[selected_columns]

            # Drop rows with missing target values
            df_cleaned = df_selected.dropna(subset=[target_column])

            # Split the cleaned data into training and testing sets
            X = df_cleaned[['Y_IN_DEC', 'X_IN_DEC']]
            y = df_cleaned[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a pipeline for preprocessing and modeling
            model = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('regressor', LinearRegression())
            ])

            # Train the model
            model.fit(X_train, y_train)

            # Use the provided latitude and longitude values
            user_data = pd.DataFrame([[user_latitude, user_longitude]], columns=['Y_IN_DEC', 'X_IN_DEC'])

            prediction = model.predict(user_data)

            # If values are missing, the prediction will be based on the provided values
            print(f"Predicted {target_column}:", prediction[0])

            predictions[target_column] = prediction[0]

        return predictions

    def plot_wave_graph(predictions, export_path=None):
        aquifer_names = list(predictions.keys())
        predicted_values = list(predictions.values())

        # Define colors for each aquifer
        colors = ['#FF1493', '#00FF00', '#FFA500', '#0000FF']

        # Plotting the wave graph (line plot) with different colors and connected dots
        plt.figure(figsize=(8, 5))
        
        # Connect all points with a single line
        for i in range(len(aquifer_names)-1):
            plt.plot([aquifer_names[i], aquifer_names[i+1]], [predicted_values[i], predicted_values[i+1]], marker='o', linestyle='-', color=colors[i], markersize=8, linewidth=2)
        
        plt.xlabel('Aquifer')
        plt.ylabel('Predicted Values')
        plt.title('Predicted Values for Different Aquifers')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot as a JPG file if export_path is provided
        if export_path:
            plt.savefig(export_path, format='jpg')

        # Display the plot
        # plt.show()

    # File path
    file_path = 'Transmisivitty.xlsx'

    # Take latitude and longitude input only once
    user_latitude = user_lat
    user_longitude = user_lon

    # Target columns
    target_columns = ['aq1_yield (lps)', 'aq2_yield (lps)', 'AQ3_yield (lps)', 'AQ4_yield (lps)']

    # Train and predict for all targets
    all_predictions = train_and_predict(file_path, target_columns, user_latitude, user_longitude)

    # Provide the export path for the JPG file
    export_path = 'predicted_wave_graph.jpg'

    # Plotting the wave graph (line plot) with different colors and a single line connecting all points
    plot_wave_graph(all_predictions, export_path)

    # Encoding image    
    # Load the image as bytes
    with open('predicted_wave_graph.jpg', 'rb') as image_file:
        image_data = image_file.read()

    # Encode the image data to base64
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    # Create a dictionary with the image data
    image_dict = {'image': encoded_image}

    # Convert the dictionary to a JSON string
    result = {
        "image": json.dumps(image_dict),
        "predictions": f"{all_predictions}"
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
