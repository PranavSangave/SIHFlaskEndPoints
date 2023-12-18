from flask import Flask, jsonify
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/myModelTwo/<float:user_lat>/<float:user_lon>')
def myModelTwo(user_lat, user_lon):
    # Load the data
    data = pd.read_csv('data.csv')

    # Filter the data to include only dug wells
    dug_wells = data[data['SITE_TYPE'] == 'Dug Well']

    # Extract latitude and longitude columns
    coordinates = dug_wells[['LAT', 'LON']]

    # Use the loaded model for predictions
    X_user = [[user_lat, user_lon]]
    user_labels = model.labels_
    user_distances = [geodesic((user_lat, user_lon), coord).kilometers for coord in coordinates.itertuples(index=False, name=None)]

    # Define a threshold distance (e.g., 2 km)
    threshold_distance = 2

    # Check if the user-provided location is within 2 km of any well using the trained model
    within_range = any(distance <= threshold_distance for distance in user_distances)

    if within_range:
        result = {
            'Answer': 'You can dig a well at this location.'
        }
        return jsonify(result)
    else:
        result = {
            'Answer': 'You cannot dig a well at this location.'
        }
        return jsonify(result)


### NOT WORKING (ATTRIBUTE ERROR : FIX THIS FIRST)
@app.route('/newModel/<float:user_latitude>/<float:user_longitude>')
def newModel(user_latitude, user_longitude):
    # Load the combined model and imputers
    file_path_pkl = "siteType_depth_model.pkl"
    models_and_imputers = joblib.load(file_path_pkl)

    # Extract models and imputers
    depth_model = models_and_imputers['depth_model']
    well_type_model = models_and_imputers['well_type_model']
    depth_imputer = models_and_imputers['depth_imputer']
    well_type_imputer = models_and_imputers['well_type_imputer']

    # Function to recommend well type based on user input
    def recommend_well_type(user_lat, user_lon):
        user_location = [[user_lat, user_lon]]

        # Predict depth for user location
        user_depth_imputed = depth_imputer.transform(user_location)
        user_depth_prediction = depth_model.predict(user_depth_imputed)

        # Predict well type for user location
        user_well_type_imputed = well_type_imputer.transform(user_location)
        user_well_type_prediction = well_type_model.predict(user_well_type_imputed)

        return user_depth_prediction[0], user_well_type_prediction[0]

    # Example usage
    # user_latitude = float(input("Enter your latitude: "))
    # user_longitude = float(input("Enter your longitude:"))

    predicted_depth, suggested_well_type = recommend_well_type(user_latitude, user_longitude)

    result = {
        'predicted_depth': predicted_depth,
        'suggested_well_type': suggested_well_type
    }
    return jsonify(result)
     

@app.route('/myModel/<float:user_lat>/<float:user_lon>')
def myModel(user_lat, user_lon):
    # Load the data
    data = pd.read_csv('data.csv')

    # Filter the data to include only dug wells
    dug_wells = data[data['SITE_TYPE'] == 'Dug Well']

    # Extract latitude and longitude columns
    coordinates = dug_wells[['LAT', 'LON']]

    # Create a map centered at a suitable location (e.g., the average coordinates of the wells)
    avg_lat = coordinates['LAT'].mean()
    avg_lon = coordinates['LON'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=8)

    # Create a marker cluster for well locations
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each well location
    for _, row in coordinates.iterrows():
        folium.Marker([row['LAT'], row['LON']]).add_to(marker_cluster)

    # Save the map to an HTML file for visualization
    m.save('well_locations_map.html')

    # Prompt the user for latitude and longitude coordinates
    # user_lat = float(input("Enter your latitude: "))
    # user_lon = float(input("Enter your longitude: "))

    # Add a marker for the user-provided location
    folium.Marker([user_lat, user_lon], icon=folium.Icon(color='red')).add_to(m)

    # Save the updated map to visualize the user's location
    m.save('user_location_map.html')

    # # Calculate Haversine distance between coordinates
    def haversine_distance(coord1, coord2):
        return geodesic(coord1, coord2).kilometers

    # Check if the user-provided location is within 2 km of any well
    within_range = any(haversine_distance((user_lat, user_lon), (row['LAT'], row['LON'])) <= 2 for _, row in coordinates.iterrows())

    if within_range:
        result = {
            'Answer': 'You can dig a well at this location.'
        }
        return jsonify(result)
    else:
        result = {
            'Answer': 'You cannot dig a well at this location.'
        }
        return jsonify(result)

@app.route('/newModelSrushti/<float:user_lat>/<float:user_lon>')
def newModeSrushti(user_lat, user_lon):
    # User input for file path
    file_path = "water_quality_shuffled.csv"

    # Load the dataset
    df = pd.read_csv(file_path)

    # Remove rows with missing values
    df = df.dropna(subset=['Latitude', 'Longitude', 'EC_micro_mhos_per_cm', 'TDS_mg_per_l', 'Chloride_mg_per_l'])

    # Extract features (X) and target variables (y_ec, y_tds, y_chlorine)
    X = df[['Latitude', 'Longitude']]
    y_ec = df['EC_micro_mhos_per_cm']
    y_tds = df['TDS_mg_per_l']
    y_chlorine = df['Chloride_mg_per_l']

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_ec_train, y_ec_test, y_tds_train, y_tds_test, y_chlorine_train, y_chlorine_test = train_test_split(
        X, y_ec, y_tds, y_chlorine, test_size=0.3, random_state=42
    )

    # Handle missing values using SimpleImputer for EC, TDS, and Chlorine prediction
    ec_imputer = SimpleImputer(strategy='mean')
    tds_imputer = SimpleImputer(strategy='mean')
    chlorine_imputer = SimpleImputer(strategy='mean')

    X_ec_imputed_train = ec_imputer.fit_transform(X_train)
    X_ec_imputed_test = ec_imputer.transform(X_test)

    X_tds_imputed_train = tds_imputer.fit_transform(X_train)
    X_tds_imputed_test = tds_imputer.transform(X_test)

    X_chlorine_imputed_train = chlorine_imputer.fit_transform(X_train)
    X_chlorine_imputed_test = chlorine_imputer.transform(X_test)

    # Find 3 nearest neighbors for the given location
    test_location_lat = user_lat
    test_location_lon = user_lon
    test_location = [[test_location_lat, test_location_lon]]

    neighbors = NearestNeighbors(n_neighbors=3, metric='haversine')
    neighbors.fit(np.radians(X))
    distances, indices = neighbors.kneighbors(np.radians(test_location))

    # Extract the indices of the nearest neighbors
    nearest_indices = indices[0]

    # Extract EC, TDS, and Chlorine values of the nearest neighbors
    nearest_values_ec = y_ec.iloc[nearest_indices].values
    nearest_values_tds = y_tds.iloc[nearest_indices].values
    nearest_values_chlorine = y_chlorine.iloc[nearest_indices].values

    # Calculate the average values
    average_ec = np.mean(nearest_values_ec)
    average_tds = np.mean(nearest_values_tds)
    average_chlorine = np.mean(nearest_values_chlorine)

    print(f'\nAverage Predicted values of EC, TDS, and Chloride for the given location:')
    print(f'EC: {average_ec:.2f}')
    print(f'TDS: {average_tds:.2f}')
    print(f'Chloride: {average_chlorine:.2f}')

    # Define water quality classification based on parameters
    def classify_water_quality(ec, tds, chloride):
        if ec < 750 and tds < 500 and chloride < 250:
            return 'Good'
        elif 750 <= ec <= 1500 and 500 <= tds <= 1000 and 250 <= chloride <= 500:
            return 'Fair'
        else:
            return 'Not Acceptable'    
    
    # Classify water quality for the average values
    classification = classify_water_quality(average_ec, average_tds, average_chlorine)
    # print(f'Water Quality Classification: {classification}')

    result = {
        'EC': f'{average_ec:.2f}',
        'TDS': f'{average_tds:.2f}',
        'Chloride': f'{average_chlorine:.2f}',
        'classfication': f'{classification}'
    }

    # # Save the model to H5 format in the specified folder
    # model_save_path = "/content/drive/MyDrive/Smart_India_Hackathon_2023/water_quality_model.h5"
    # with h5py.File(model_save_path, 'w') as model_file:
    #     model_file.create_dataset('indices', data=indices)

    # print(f"Model saved to {model_save_path}")

    # if classification == 'Good':
    #     print("The water is suitable for drinking (potable).")
    # else:
    #     print("The water is not suitable for drinking (non-potable).")
        
    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True)
