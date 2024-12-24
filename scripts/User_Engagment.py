import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Aggregate Engagement Metrics Per Customer
def aggregate_engagement_metrics(df):
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Sessions frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum',  # Total upload data
    }).reset_index()

    user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
    return user_metrics

# Report Top 10 Customers
def report_top_users(user_metrics):
    print("Top 10 Users by Sessions Frequency:")
    print(user_metrics.nlargest(10, 'Bearer Id'))

    print("\nTop 10 Users by Session Duration:")
    print(user_metrics.nlargest(10, 'Dur. (ms)'))

    print("\nTop 10 Users by Total Traffic:")
    print(user_metrics.nlargest(10, 'Total Traffic (Bytes)'))

# Normalize Metrics and Apply K-Means
def normalize_metrics(user_metrics):
    # Define the columns to be normalized
    columns_to_normalize = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    normalized_metrics = scaler.fit_transform(user_metrics[columns_to_normalize])
    
    # Create a DataFrame with the normalized data
    normalized_df = pd.DataFrame(normalized_metrics, columns=columns_to_normalize)
    
    return normalized_df

def apply_kmeans(normalized_metrics, k=3):
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Apply KMeans and get the cluster labels
    cluster_labels = kmeans.fit_predict(normalized_metrics)
    
    # Return the cluster labels as a DataFrame (as a single column)
    return pd.Series(cluster_labels, name='Cluster')

# Cluster Analysis
def analyze_clusters(user_metrics, normalized_metrics):
    # Apply KMeans clustering (assuming you want to reapply it here or pass the existing cluster labels)
    cluster_labels = apply_kmeans(normalized_metrics)
    
    # Add the cluster labels as a new column in user_metrics
    user_metrics['Cluster'] = cluster_labels
    
    # Group by the 'Cluster' column and calculate min, max, mean, and sum for each column
    cluster_summary = user_metrics.groupby('Cluster').agg({
        'Bearer Id': ['min', 'max', 'mean', 'sum'],
        'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    
    print(cluster_summary)

    # Plot the distribution of users across clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(data=user_metrics, x='Cluster')
    plt.title("Distribution of Users Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.show()


# Application-Based Engagement
def top_application_users(df):
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

    app_engagement = {}
    for app in apps:
        app_engagement[app] = df.groupby('MSISDN/Number')[app].sum().nlargest(10)
        print(f"Top 10 Users for {app}:\n{app_engagement[app]}")

    return app_engagement

    
# Function to plot the top 3 most used applications based on total traffic
def plot_top_applications(df):
    # Define the list of applications
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Sum the traffic for each application
    total_app_traffic = df[apps].sum()
    
    # Sort the total traffic in descending order and get the top 3
    top_3_apps = total_app_traffic.nlargest(3)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create a barplot for the top 3 applications
    sns.barplot(x=top_3_apps.index, y=top_3_apps.values, palette="viridis")
    
    # Adding titles and labels
    plt.title("Top 3 Most Used Applications", fontsize=16)
    plt.xlabel("Application", fontsize=12)
    plt.ylabel("Total Traffic (Bytes)", fontsize=12)
    
    # Display the plot
    plt.tight_layout()
    plt.show()


# Elbow Method for Optimized K
def elbow_method(df, apps):
    # Normalize/standardize the engagement metrics (application download traffic in this case)
    scaler = StandardScaler()
    normalized_metrics = scaler.fit_transform(df[apps])

    distortions = []
    K = range(1, 11)  # You can extend this range for more granularity
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_metrics)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, marker='o', linestyle='-', color='b')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.xticks(K)
    plt.grid(True)
    plt.show()

    # Output the distortion values to analyze
    return distortions
