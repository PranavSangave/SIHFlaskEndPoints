from flask import Flask, jsonify
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import joblib

from sklearn.cluster import KMeans

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

# Used to find TDS, EC & Chloride
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

    result = {
        'EC': f'{average_ec:.2f}',
        'TDS': f'{average_tds:.2f}',
        'Chloride': f'{average_chlorine:.2f}',
        'classfication': f'{classification}'
    }
        
    return jsonify(result)

# Used to find depth and well type
@app.route('/newCombined/<float:user_lat>/<float:user_lon>')
def newCombined(user_lat, user_lon):
    # Load the dataset
    file_path = "dugwell.csv"  # Update this with the correct file path
    df = pd.read_csv(file_path)

    # Remove rows with missing values
    df = df.dropna(subset=['Y', 'X', 'Depth (m.bgl)', 'Well Type'])

    # Extract features (X) and target variables (y_depth, y_well_type)
    X = df[['Y', 'X']]
    y_depth = df['Depth (m.bgl)']
    y_well_type = df['Well Type']

    # Split the data into training and testing sets
    X_train, X_test, y_depth_train, y_depth_test, y_well_type_train, y_well_type_test = train_test_split(
        X, y_depth, y_well_type, test_size=0.2, random_state=42
    )

    # Handle missing values using SimpleImputer for depth prediction
    depth_imputer = SimpleImputer(strategy='mean')
    X_depth_imputed = depth_imputer.fit_transform(X_train)

    # Create and fit the KNN regression model for depth prediction
    depth_model = KNeighborsRegressor(n_neighbors=3)
    depth_model.fit(X_depth_imputed, y_depth_train)

    # Predict depth for the test set
    X_test_depth_imputed = depth_imputer.transform(X_test)
    depth_predictions = depth_model.predict(X_test_depth_imputed)

    # Handle missing values using SimpleImputer for well type prediction
    well_type_imputer = SimpleImputer(strategy='most_frequent')
    X_well_type_imputed = well_type_imputer.fit_transform(X_train)

    # Create and fit the KNN classification model for well type prediction
    well_type_model = KNeighborsClassifier(n_neighbors=3)
    well_type_model.fit(X_well_type_imputed, y_well_type_train)

    # Predict well type for the test set
    X_test_well_type_imputed = well_type_imputer.transform(X_test)
    well_type_predictions = well_type_model.predict(X_test_well_type_imputed)

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
    user_latitude = user_lat
    user_longitude = user_lon

    predicted_depth, suggested_well_type = recommend_well_type(user_latitude, user_longitude)

    # Display the predictions
    # print(f"Predicted Depth: {predicted_depth} meters")
    # print(f"Suggested Well Type: {suggested_well_type}")

    result = {
        'predicted_depth': f'{predicted_depth}',
        'well_type': f'{suggested_well_type}'
    }

    return jsonify(result)

# Used to find flow
@app.route('/newFlowRate/<float:user_lat>/<float:user_lon>')
def newFlowRate(user_lat, user_lon):
    # Step 3: Load your Excel file into a Pandas DataFrame
    # Assuming your Excel file has a sheet named 'Sheet1' (adjust accordingly)
    df = pd.read_excel('Transmisivitty.xlsx', sheet_name='Sheet1')

    # Step 4: Data Preprocessing
    # Assuming 'aq1_yield (lps)' is the column you want to predict
    X = df[['Y_IN_DEC', 'X_IN_DEC']]  # Features (using 'Y_IN_DEC' and 'X_IN_DEC')
    y = df['aq1_yield (lps)']  # Target variable

    # Handle missing values in the target variable
    y.fillna(y.mean(), inplace=True)  # You can choose a different imputation strategy if needed

    # Handle missing values in the input features
    X.fillna(X.mean(), inplace=True)  # You can choose a different imputation strategy if needed

    # Normalize or standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Perform K-means clustering to identify clusters
    kmeans = KMeans(n_clusters=5, random_state=42)  # You can adjust the number of clusters (k) as needed
    X['Cluster'] = kmeans.fit_predict(X_scaled)

    # Step 6: Handle missing values in the input features using an imputer
    numerical_imputer = SimpleImputer(strategy='mean')
    X_scaled = numerical_imputer.fit_transform(X_scaled)

    # Step 7: Create and train the KNN model for each cluster
    knn_models = {}
    for cluster in range(kmeans.n_clusters):
        cluster_data = X[X['Cluster'] == cluster].drop('Cluster', axis=1)
        cluster_target = y[X['Cluster'] == cluster]

        knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
        knn_model.fit(cluster_data, cluster_target)

        knn_models[cluster] = knn_model

    # Step 8: Get user input - Latitude and Longitude
    user_latitude = user_lat
    user_longitude = user_lon

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Y_IN_DEC': [user_latitude],
        'X_IN_DEC': [user_longitude],
    })

    # Normalize or standardize user input
    user_input_scaled = scaler.transform(user_input)

    # Step 9: Use the trained KNN model for the nearest cluster to make predictions on the user data
    user_cluster = kmeans.predict(user_input_scaled)[0]
    user_prediction = knn_models[user_cluster].predict(user_input)

    # Step 10: Print or use the prediction as needed
    # Step 10: Print or use the prediction as needed
    print("Estimated Flow Rate:", user_prediction[0], "liters per second")
    result = {
        'flow_rate' : f'{user_prediction[0]}'
    }

    return jsonify(result)

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



if __name__ == "__main__":
    app.run(debug=True)
