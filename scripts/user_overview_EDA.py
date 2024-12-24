import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def basic_data_overview(df):
    print(df.info())
    print(df.describe())

    
def handle_missing_values(df, unique_identifiers=None):
    """
    Function to handle missing values in a DataFrame based on column types.

    Parameters:
    - df: DataFrame containing the data.
    - unique_identifiers: List of column names that should not be imputed (e.g., unique identifiers like 'Bearer Id').
    """
    if unique_identifiers is None:
        unique_identifiers = []

    # Handle numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        if col not in unique_identifiers:
            # Replace missing values with mean for numerical columns
            df[col] = df[col].fillna(df[col].mean())
    
    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if col not in unique_identifiers:
            # Replace missing values with mode (most frequent value) for categorical columns
            df[col] = df[col].fillna(df[col].mode().iloc[0])

   # Handle unique identifier columns (drop rows with missing values)
    for col in unique_identifiers:
        print(f"Checking column '{col}' for NaN values...")
        # Ensure the column exists and check for NaNs
        if col in df.columns:
            print(f"NaN values in '{col}': {df[col].isna().sum()}")
            # Drop rows where the unique identifier column has missing values
            df.dropna(subset=[col], inplace=True)
        else:
            print(f"Column '{col}' not found in the DataFrame.")

    return df

def treat_outliers(df):
    # Loop through all columns in the DataFrame
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Treat the outliers by setting values outside the bounds to the corresponding bound
        df[col] = np.where(df[col] > upper_bound, upper_bound, 
                           np.where(df[col] < lower_bound, lower_bound, df[col]))
    
    return df

#function to find the top handsets
def top_handsets(df):
    # Get top 10 handsets based on their count
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    print("Top 10 Handsets:\n", top_10_handsets)

    # Plot top 10 handsets without the 'palette' argument
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index)
    plt.title("Top 10 Handsets")
    plt.xlabel("Count")
    plt.ylabel("Handset Type")
    plt.show()

    # Get top 3 manufacturers based on their count
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    print("Top 3 Handset Manufacturers:\n", top_3_manufacturers)

    # Plot top 3 manufacturers without the 'palette' argument
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_3_manufacturers.values, y=top_3_manufacturers.index)
    plt.title("Top 3 Handset Manufacturers")
    plt.xlabel("Count")
    plt.ylabel("Manufacturer")
    plt.show()

    # For each manufacturer, plot top 5 handsets for that manufacturer
    for manufacturer in top_3_manufacturers.index:
        top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        print(f"Top 5 Handsets for {manufacturer}:\n", top_5_handsets)

        # Plot top 5 handsets for the current manufacturer without the 'palette' argument
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_5_handsets.values, y=top_5_handsets.index)
        plt.title(f"Top 5 Handsets for {manufacturer}")
        plt.xlabel("Count")
        plt.ylabel("Handset Type")
        plt.show()

def user_aggregation_metrics(df):
    # Ensure the 'Start' and 'End' columns are in datetime format
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])
    
    # Create new columns for total download and upload data across all apps
    df['Total Download'] = df['Youtube DL (Bytes)'] + df['Netflix DL (Bytes)'] + df['Gaming DL (Bytes)'] + df['Other DL (Bytes)']
    df['Total Upload'] = df['Youtube UL (Bytes)'] + df['Netflix UL (Bytes)'] + df['Gaming UL (Bytes)'] + df['Other UL (Bytes)']

    # Aggregate the data by MSISDN/Number (User)
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Number of sessions
        'Dur. (ms)': 'sum',  # Total session duration
        'Total Download': 'sum',  # Total download data across apps
        'Total Upload': 'sum',  # Total upload data across apps
    }).reset_index()

    # Adding total data volume (Download + Upload)
    user_metrics['Total Data Volume (Bytes)'] = user_metrics['Total Download'] + user_metrics['Total Upload']
    
    
    return user_metrics

def decile_segmentation(user_metrics):
    user_metrics['Decile'] = pd.qcut(user_metrics['Dur. (ms)'], 10, labels=False)
    decile_data = user_metrics.groupby('Decile').agg({'Total Data Volume (Bytes)': 'sum'}).reset_index()
    print(decile_data)

def univariate_analysis(df):
    numeric_columns = df.select_dtypes(include=['float64']).columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

def bivariate_analysis(df):
    # Create a new column for Total Data Volume
    df['Total Data Volume (Bytes)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    # List of application-specific data columns to analyze
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Plot each app's download data against Total Data Volume
    for app in apps:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=app, y='Total Data Volume (Bytes)')
        plt.title(f"{app} vs Total Data Volume")
        plt.xlabel(app)
        plt.ylabel('Total Data Volume (Bytes)')
        plt.show()


def application_correlation_analysis(df):
    app_data = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]
    correlation_matrix = app_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix for Application Data")
    plt.show()

def pca_analysis(df):
    app_data = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]

    # Standardize data
    pca_data = StandardScaler().fit_transform(app_data)

    # PCA transformation
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data)

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    print("PCA Interpretation:")
    print("1. The first principal component explains {:.2f}% of the variance and relates to overall data usage patterns.".format(pca.explained_variance_ratio_[0] * 100))
    print("2. The second principal component explains {:.2f}% of the variance and focuses on specific app usage patterns.".format(pca.explained_variance_ratio_[1] * 100))
    print("3. Together, these two components capture {:.2f}% of the variance, providing a reduced dimensional view of user behavior.".format(sum(pca.explained_variance_ratio_) * 100))
