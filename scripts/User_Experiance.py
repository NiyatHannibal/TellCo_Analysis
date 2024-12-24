import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def aggregate_per_customer(df):
    """
    Aggregate metrics per customer, replacing missing values and handling outliers.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    
    # Perform the aggregation
    agg_df = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': lambda x: x.mode()[0],
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean'
    }).reset_index()

    # Rename columns to more user-friendly names
    agg_df.rename(columns={
        'TCP DL Retrans. Vol (Bytes)': 'Average TCP Retransmission DL',
        'TCP UL Retrans. Vol (Bytes)': 'Average TCP Retransmission UL',
        'Avg RTT DL (ms)': 'Average RTT DL',
        'Avg RTT UL (ms)': 'Average RTT UL',
        'Avg Bearer TP DL (kbps)': 'Average Throughput DL',
        'Avg Bearer TP UL (kbps)': 'Average Throughput UL'
    }, inplace=True)

    return agg_df


def compute_and_display_stats(data, columns_of_interest, top_n=10):
    """
    Computes and displays top, bottom, and most frequent values for specified columns.

    Parameters:
        data (pd.DataFrame): The dataset containing the columns of interest.
        columns_of_interest (dict): Mapping of original column names to aliases.
        top_n (int): Number of top/bottom/most frequent values to compute.

    Returns:
        dict: Results containing top, bottom, and most frequent values for each column.
    """
    # Rename columns for easier reference
    data.rename(columns=columns_of_interest, inplace=True)

    # Function to compute top, bottom, and most frequent values
    def compute_stats(series, top_n):
        top_values = series.nlargest(top_n)
        bottom_values = series.nsmallest(top_n)
        most_frequent = series.value_counts().head(top_n)
        return top_values, bottom_values, most_frequent

    # Iterate over the columns of interest and compute stats
    results = {}
    for col, col_alias in columns_of_interest.items():
        if col_alias in data.columns:
            print(f"Processing column: {col_alias}")
            top, bottom, frequent = compute_stats(data[col_alias].dropna(), top_n)
            results[col_alias] = {
                'Top Values': top,
                'Bottom Values': bottom,
                'Most Frequent': frequent
            }

    # Display results
    for metric, stats in results.items():
        print(f"\nMetric: {metric}")
        print("Top Values:")
        print(stats['Top Values'])
        print("\nBottom Values:")
        print(stats['Bottom Values'])
        print("\nMost Frequent:")
        print(stats['Most Frequent'])

    return results


def distribution_per_handset(df, metric_column, handset_column, title, ylabel, top_n=10):
    """
    Compute the distribution of a metric per handset type, limited to top N handsets for better visualization.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        metric_column (str): Column name for the metric.
        handset_column (str): Column name for the handset type.
        title (str): Title for the plot.
        ylabel (str): Label for the y-axis.
        top_n (int): Limit to the top N handsets by the mean of the metric.

    Returns:
        pd.DataFrame: Aggregated distribution DataFrame with limited handset types.
    """
    # Calculate the mean distribution per handset type
    distribution = df.groupby(handset_column)[metric_column].mean().reset_index()

    # Sort the distribution and select top N handset types
    distribution_sorted = distribution.sort_values(by=metric_column, ascending=False).head(top_n)

    # Create a horizontal bar plot for better readability
    plt.figure(figsize=(10, 6))  # Adjusted size for clarity
    sns.barplot(x=metric_column, y=handset_column, data=distribution_sorted, palette="viridis")

    # Add title and labels with larger fonts
    plt.title(title, fontsize=16)
    plt.xlabel(ylabel, fontsize=14)
    plt.ylabel("Handset Type", fontsize=14)

    # Add value labels on bars for easier interpretation
    for index, value in enumerate(distribution_sorted[metric_column]):
        plt.text(value + 0.5, index, f'{value:.2f}', va='center', fontsize=12)

    # Adjust layout to avoid overlap and improve clarity
    plt.tight_layout()
    plt.show()

    return distribution_sorted
    
def kmeans_clustering_experience(df, k=3):
    """
    Perform K-means clustering on user experience metrics and visualize the clusters.

    Parameters:
        df (pd.DataFrame): DataFrame containing experience metrics.
        k (int): Number of clusters for K-means.

    Returns:
        pd.DataFrame: DataFrame with cluster assignments.
        KMeans: Fitted KMeans model.
    """
    # Columns to use for clustering
    features = ['Throughput_DL', 'TCP_DL', 'RTT_DL']

    # Handle missing values
    df[features] = df[features].fillna(df[features].mean())

    # Standardize the features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df[features])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(standardized_data)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_data)
    df['PCA1'] = pca_data[:, 0]
    df['PCA2'] = pca_data[:, 1]

    # Plot clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
    plt.title(f'K-means Clustering of Users (K={k})', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.tight_layout()
    plt.show()

    return df, kmeans


def interpret_clusters(df):
    """
    Provide an interpretation of each cluster based on aggregated metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing clustered data.

    Returns:
        pd.DataFrame: Cluster interpretation.
    """
    cluster_summary = df.groupby('Cluster').agg({
        'Throughput_DL': ['mean', 'std'],
        'TCP_DL': ['mean', 'std'],
        'RTT_DL': ['mean', 'std']
    }).reset_index()

    print("Cluster Summary:\n", cluster_summary)
    return cluster_summary
