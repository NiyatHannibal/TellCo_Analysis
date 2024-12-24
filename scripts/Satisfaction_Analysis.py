import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Aggregate Engagement Metrics
def aggregate_engagement_metrics(df):
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Sessions frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum'   # Total upload data
    }).reset_index()
    user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
    return user_metrics

# Aggregate Experience Metrics
def aggregate_experience_metrics(df):
    experience_metrics = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',  
        'Avg RTT UL (ms)': 'mean',   
        'Avg Bearer TP DL (kbps)': 'mean',   
        'Avg Bearer TP UL (kbps)': 'mean',   
        'TCP DL Retrans. Vol (Bytes)': 'mean',   
        'TCP UL Retrans. Vol (Bytes)': 'mean'    
    }).reset_index()
    return experience_metrics

# Normalize Metrics
def normalize_metrics(df, columns_to_normalize):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
    return normalized_df, scaler

# Apply KMeans Clustering
def apply_kmeans(normalized_metrics, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_metrics)
    return cluster_labels, kmeans

# Calculate Engagement Scores
def calculate_engagement_scores(normalized_df, cluster_labels, kmeans):
    # Identify the less engaged cluster
    cluster_means = normalized_df.groupby(cluster_labels).mean()
    less_engaged_cluster = cluster_means.mean(axis=1).idxmin()

    # Centroid of the less engaged cluster
    less_engaged_centroid = kmeans.cluster_centers_[less_engaged_cluster]

    # Calculate Euclidean distance to the less engaged cluster centroid
    distances = normalized_df.apply(lambda row: euclidean(row, less_engaged_centroid), axis=1)
    return distances

# Calculate Experience Scores
def calculate_experience_scores(normalized_df, cluster_labels, kmeans):
    # Identify the worst experience cluster
    cluster_means = normalized_df.groupby(cluster_labels).mean()
    worst_experience_cluster = cluster_means.mean(axis=1).idxmax()

    # Centroid of the worst experience cluster
    worst_experience_centroid = kmeans.cluster_centers_[worst_experience_cluster]

    # Calculate Euclidean distance to the worst experience cluster centroid
    distances = normalized_df.apply(lambda row: euclidean(row, worst_experience_centroid), axis=1)
    return distances

# Combine Engagement and Experience Metrics
def combine_metrics(engagement_metrics, experience_metrics):
    combined_metrics = pd.merge(engagement_metrics, experience_metrics, on='MSISDN/Number')
    return combined_metrics

# Full Pipeline Execution
def full_analysis_pipeline(data, k=3):
    # Aggregate metrics
    engagement_metrics = aggregate_engagement_metrics(data)
    experience_metrics = aggregate_experience_metrics(data)
    
    # Combine metrics
    combined_metrics = combine_metrics(engagement_metrics, experience_metrics)
    
    # Normalize metrics
    columns_to_normalize = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)', 
                            'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                            'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    normalized_df, scaler = normalize_metrics(combined_metrics, columns_to_normalize)
    
    # Apply clustering
    cluster_labels, kmeans = apply_kmeans(normalized_df, k=k)
    
    # Calculate engagement scores
    engagement_scores = calculate_engagement_scores(normalized_df, cluster_labels, kmeans)
    
    # Calculate experience scores
    experience_scores = calculate_experience_scores(normalized_df, cluster_labels, kmeans)
    
    # Add results to combined_metrics
    combined_metrics['Cluster'] = cluster_labels
    combined_metrics['Engagement Score'] = engagement_scores
    combined_metrics['Experience Score'] = experience_scores
    
    return combined_metrics

# Calculate Satisfaction Scores (average of engagement and experience scores)
def calculate_satisfaction_scores(engagement_scores, experience_scores):
    return (engagement_scores + experience_scores) / 2

# Report Top 10 Satisfied Customers
def top_10_satisfied_customers(combined_metrics):
    # Calculate satisfaction score
    combined_metrics['Satisfaction Score'] = calculate_satisfaction_scores(combined_metrics['Engagement Score'], combined_metrics['Experience Score'])
    
    # Sort users by satisfaction score in descending order and get top 10
    top_10 = combined_metrics[['MSISDN/Number', 'Satisfaction Score']].sort_values(by='Satisfaction Score', ascending=False).head(10)
    
    return top_10

# Prepare the data for regression model
def prepare_data_for_regression(combined_metrics):
    # Define the features (independent variables)
    features = combined_metrics[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)', 
                                 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                                 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]
    
    # Target variable: Satisfaction Score
    target = combined_metrics['Satisfaction Score']
    
    return features, target

# Train and evaluate the regression model
def regression_model(combined_metrics):
    # Step 1: Prepare the data
    features, target = prepare_data_for_regression(combined_metrics)
    
    # Step 2: Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Step 3: Initialize and train the model (Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Step 4: Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Step 5: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    
    # Return the model and test set results
    return model, y_test, y_pred

# usage of the regression model
def run_regression_on_combined_metrics(combined_metrics):
    model, y_test, y_pred = regression_model(combined_metrics)
    
    # Print a few predicted vs actual values for comparison
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print("\nPredicted vs Actual Satisfaction Scores (Top 10):")
    print(results.head(10))

# k-means(k-2)
# Step 1: Prepare the Data (combine Engagement and Experience Scores)
def prepare_data_for_kmeans(combined_metrics):
    # We assume that combined_metrics has columns 'Engagement Score' and 'Experience Score'
    scores_data = combined_metrics[['Engagement Score', 'Experience Score']]
    return scores_data

# Step 2: Normalize the Scores (Engagement and Experience)
def normalize_scores(scores_data):
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_data)
    normalized_df = pd.DataFrame(normalized_scores, columns=['Engagement Score', 'Experience Score'])
    return normalized_df, scaler

# Step 3: Apply K-Means (k=2)
def run_kmeans_on_scores(normalized_scores, k=2):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_scores)
    return cluster_labels, kmeans

# Step 4: Add Cluster Labels to Combined Metrics
def add_kmeans_clusters_to_combined_metrics(combined_metrics, cluster_labels):
    combined_metrics['KMeans Cluster'] = cluster_labels
    return combined_metrics

# Full Execution for K-Means Clustering on Engagement & Experience Scores
def run_kmeans_on_combined_metrics(combined_metrics):
    # Prepare data for K-Means
    scores_data = prepare_data_for_kmeans(combined_metrics)
    
    # Normalize the data
    normalized_scores, scaler = normalize_scores(scores_data)
    
    # Apply K-Means Clustering (k=2)
    cluster_labels, kmeans = run_kmeans_on_scores(normalized_scores, k=2)
    
    # Add the cluster labels to the original combined_metrics DataFrame
    combined_metrics_with_clusters = add_kmeans_clusters_to_combined_metrics(combined_metrics, cluster_labels)
    
    # Return the result with the cluster labels
    return combined_metrics_with_clusters

# Function to aggregate average satisfaction & experience scores per cluster
def aggregate_scores_per_cluster(combined_metrics_with_clusters):
    # Group by KMeans cluster and calculate the mean of satisfaction and experience scores
    aggregated_scores = combined_metrics_with_clusters.groupby('KMeans Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()
    
    return aggregated_scores
