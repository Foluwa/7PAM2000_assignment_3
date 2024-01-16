#!/usr/bin/python3
"""
Created on Thur Dec 28 03:20:25 2023

@author: Moronfoluwa Akintola
"""

""""

This tutorial explains how to make a poster for an academic conference. 
While very instructive a few details will be different for your poster.

Your poster will be A1, portrait or landscape.
Bullet points are not mandatory (--> example posters)
Columns style is not the only good solution. One can make creative use of textboxes (--> example posters)
No logos, no photo, no acknowledgements and you do not have co-authors. Title and name are sufficient.
No need for references, if you use Worldbank data. Links to other data sources, if used.
You will show your graphs, of course. You can add figures from external sources.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.cluster import KMeans
from scipy.stats import linregress
from scipy.optimize import curve_fit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures

# Function definitions
def series_dict(df):
    """
        Extract Series Code and Series Name
    """
    data_dict = df[['Series Code', 'Series Name']].dropna().drop_duplicates().set_index('Series Code')['Series Name'].to_dict()
    return data_dict

def linear_fit(x, a, b):
    """
        Define a simple linear fit function
    """
    return a * x + b

def err_ranges(x, popt, pcov):
    """
        Define the error ranges function
    """
    perr = np.sqrt(np.diag(pcov))
    y = linear_fit(x, *popt)
    y_upper = linear_fit(x, *(popt + perr))
    y_lower = linear_fit(x, *(popt - perr))
    return y, y_lower, y_upper

def logistic_function(x, L, k, x0):
    """ Define the logistic function """
    return L / (1 + np.exp(-k * (x - x0)))

def convert_csv(filename):
    """ 
        Read the CSV file, convert to a dataframe and return the same dataframe 
    """
    df = pd.read_csv(filename)
    return df

def series_dict(df):
    """
        Extract Series Code and Series Name
    """
    df_dict = df[['Series Code', 'Series Name']].dropna().drop_duplicates().set_index('Series Code')['Series Name'].to_dict()
    return df_dict

def plot_scatter(data, series_x, series_y, chart_title):
    """ 
        Function takes title and series code argument
        and draws scatter chart from the data
    """
    series_one = data[data['Series Code'] == series_x]
    series_two = data[data['Series Code'] == series_y]
    # series_two = data[data['Series Code'] == 'SP.POP.TOTL']

    # Since the data is in wide format, we need to melt it to long format
    series_one_melted = series_one.melt(id_vars=['Country Name', 'Country Code'], 
                                                       value_vars=series_one.columns[4:], 
                                                       var_name='Year', value_name='access')
    
    series_two_melted = series_two.melt(id_vars=['Country Name', 'Country Code'], 
                                                   value_vars=series_two.columns[4:], 
                                                   var_name='Year', value_name='Total Population')

    # Merge the two datasets on 'Country Code' and 'Year'
    merged_data = pd.merge(series_one_melted, series_two_melted, on=['Country Code', 'Year'])

    # Convert 'Year' to datetime and extract the year for plotting
    merged_data['Year'] = pd.to_datetime(merged_data['Year'].str.extract('(\d{4})')[0]).dt.year

    # Convert 'access' to numeric, coercing errors to NaN
    merged_data['access'] = pd.to_numeric(merged_data['access'], errors='coerce')

    # Drop rows with NaN values in 'access'
    merged_data = merged_data.dropna(subset=['access'])

    # Plotting with specified number of ticks on the Y axis
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data['Total Population'], merged_data['access'])
    # plt.title('Access to Electricity vs Total Population')
    plt.title(chart_title)
    # plt.xlabel('Total Population')
    plt.xlabel(data_dict[series_x])
    plt.ylabel(data_dict[series_y])
    # Using a log scale for better visualization
    plt.xscale('log') 

    # Set the number of ticks on the Y axis to 5
    plt.yticks(np.linspace(merged_data['access'].min(), 
                           merged_data['access'].max(), 5))

    return plt.show()


def k_clustering(df, series_name_y, series_name_x, n_clusters):

    """
        Extract data for 'EG.ELC.ACCS.ZS' and 'SP.POP.TOTL'
    """ 
    # Load the dataset
#     file_path = 'data.csv'
#     data = pd.read_csv(file_path, encoding='ascii')

    # Extract data for 'SL.TLF.TOTL.IN' and 'SP.POP.TOTL'
    # labor_force = data[data['Series Code'] == 'EN.ATM.METH.KT.CE']
    # total_population = data[data['Series Code'] == 'SP.POP.TOTL']
    labor_force = df[df['Series Code'] == series_name_y]
    total_population = df[df['Series Code'] == series_name_x]

    # Melt the dataframes
    labor_force_melted = labor_force.melt(id_vars=['Country Name', 'Country Code'], 
                                          value_vars=labor_force.columns[4:], 
                                          var_name='Year', value_name='Labor Force')
    total_population_melted = total_population.melt(id_vars=['Country Name', 'Country Code'], 
                                                  value_vars=total_population.columns[4:], 
                                                  var_name='Year', value_name='Total Population')

    # Merge the two datasets on 'Country Code' and 'Year'
    merged_data = pd.merge(labor_force_melted, total_population_melted, 
                           on=['Country Code', 'Year'])

    # Convert 'Year' to datetime and extract the year for plotting
    merged_data['Year'] = pd.to_datetime(merged_data['Year'].str.extract('(\d{4})')[0]).dt.year

    # Convert to numeric and drop NaNs
    merged_data['Labor Force'] = pd.to_numeric(merged_data['Labor Force'], errors='coerce')
    merged_data['Total Population'] = pd.to_numeric(merged_data['Total Population'], errors='coerce')
    merged_data.dropna(subset=['Labor Force', 'Total Population'], inplace=True)

    # Prepare the data for clustering
    X = merged_data[['Labor Force', 'Total Population']]

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Assign the clusters to the dataframe
    merged_data['Cluster'] = kmeans.labels_

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_data, x='Total Population', y='Labor Force', hue='Cluster', palette='viridis')
    plt.title('K-Means Clustering of Labor Force and Total Population')
    plt.xlabel('Total Population')
    plt.ylabel('Labor Force')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Cluster')
    return plt.show()


def find_clusters_optimal_number(data, first_series, second_series):
    # The optimal number of clusters based on the silhouette score for the series code 'EN.ATM.CO2E.KT' and 'SP.POP.TOTL' is:

    # Optimal number of clusters: 2
    # The silhouette score is a measure of how similar an object is to its own cluster (cohesion) compared to 
    # other clusters (separation). The optimal number of clusters is the one that maximizes the average silhouette score, indicating the best definition of clusters. 
    # In this case, using 2 clusters would be the best choice according to this metric

    co2_emissions = data[data['Series Code'] == first_series]
    total_population = data[data['Series Code'] == second_series]

    # Melt the dataframes
    co2_emissions_melted = co2_emissions.melt(id_vars=['Country Name', 'Country Code'], 
                                            value_vars=co2_emissions.columns[4:], 
                                            var_name='Year', value_name='CO2 Emissions')
    total_population_melted = total_population.melt(id_vars=['Country Name', 'Country Code'], 
                                                value_vars=total_population.columns[4:], 
                                                var_name='Year', value_name='Total Population')

    # Merge the two datasets on 'Country Code' and 'Year'
    merged_data = pd.merge(co2_emissions_melted, total_population_melted, 
                        on=['Country Code', 'Year'])

    # Convert 'Year' to datetime and extract the year for plotting
    merged_data['Year'] = pd.to_datetime(merged_data['Year'].str.extract('(\d{4})')[0]).dt.year

    # Convert to numeric and drop NaNs
    merged_data['CO2 Emissions'] = pd.to_numeric(merged_data['CO2 Emissions'], errors='coerce')
    merged_data['Total Population'] = pd.to_numeric(merged_data['Total Population'], errors='coerce')
    merged_data.dropna(subset=['CO2 Emissions', 'Total Population'], inplace=True)

    # Prepare the data for clustering
    X = merged_data[['CO2 Emissions', 'Total Population']]

    # Determine the optimal number of clusters using silhouette score
    silhouette_scores = []

    # Test different numbers of clusters
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the number of clusters with the highest silhouette score
    optimal_n_clusters = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]

    print(f'Optimal number of clusters based on silhouette score for {first_series} and {second_series}:', optimal_n_clusters)
    return f'Optimal number of clusters based on silhouette score for {first_series} and {second_series}:, optimal_n_clusters'



def draw_clustered_data(df, series_codes):

    # Filter the dataset for the selected series codes
    filtered_data = df[df['Series Code'].isin(series_codes)]

    # Melt the dataframe
    melted_data = filtered_data.melt(id_vars=['Country Name', 'Country Code', 'Series Code'], 
                                     value_vars=filtered_data.columns[4:], 
                                     var_name='Year', value_name='Value')

    # Pivot the table to have series codes as columns and years as rows
    pivot_data = melted_data.pivot_table(index=['Country Name', 'Year'], columns='Series Code', values='Value', aggfunc='first')

    # Convert to numeric and drop NaNs
    pivot_data = pivot_data.apply(pd.to_numeric, errors='coerce')
    pivot_data.dropna(axis=1, how='all', inplace=True)
    pivot_data.dropna(axis=0, how='any', inplace=True)

    # Generate the scatter matrix
    scatter_matrix(pivot_data, alpha=0.2, figsize=(30, 30), diagonal='kde')
    plt.suptitle('Scatter Matrix for Selected Series Codes')
    return plt.show()


def elbow_method(data):
    # Preprocessing the data
    # Replace '..' with NaN for proper missing values handling
    preprocessed_data = data.replace('..', pd.NA)

    # Convert all year columns to numeric, errors='coerce' will replace errors with NaN
    for col in preprocessed_data.columns[4:]:
        preprocessed_data[col] = pd.to_numeric(preprocessed_data[col], errors='coerce')

    # Check for missing values
    missing_values = preprocessed_data.isna().sum()

    # Print the number of missing values for each column
    # print(missing_values)

    # Drop rows with any missing values
    preprocessed_data = preprocessed_data.dropna()

    # Reset index after dropping rows
    preprocessed_data.reset_index(drop=True, inplace=True)

    # Show the head of the preprocessed dataframe
    # preprocessed_data.head()


    # Extract the year columns for clustering
    X = preprocessed_data.iloc[:, 4:]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the Elbow method to find the optimal number of clusters
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.grid(True)
    return plt.show()

                 
def normal_range_line(df, series_a, series_b):
    # Load the original dataset
    #df = pd.read_csv('data.csv', encoding='ascii')

    # Replace '..' with NaN to handle non-numeric values
    df.replace('..', np.nan, inplace=True)

    # Filter the data for the series code 'NY.GNP.MKTP.CD' and 'EN.ATM.CO2E.KT'
    # filtered_data = df[(df['Series Code'] == 'NY.GNP.MKTP.CD') | (df['Series Code'] == 'EN.ATM.CO2E.KT')]
    # filtered_data = df[(df['Series Code'] == 'NY.GNP.MKTP.CD') | (df['Series Code'] == 'EN.ATM.CO2E.KT')]
    filtered_data = df[(df['Series Code'] == series_a) | (df['Series Code'] == series_b)]

    # Extract the year columns
    year_columns = [col for col in filtered_data.columns if 'YR' in col]

    # Melt the dataframe to have years and values in separate columns
    melted_data = filtered_data.melt(id_vars=['Country Name', 'Series Code'], value_vars=year_columns, var_name='Year', value_name='Value')

    # Drop rows with missing values
    melted_data.dropna(inplace=True)

    # Convert year to numerical format and values to float
    melted_data['Year'] = melted_data['Year'].str.extract('(\d{4})').astype(int)
    melted_data['Value'] = melted_data['Value'].astype(float)

    # Prepare data for kmeans clustering
    X = melted_data[['Year', 'Value']].values

    # Determine the number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Find the optimal number of clusters based on the elbow method
    optimal_clusters = np.argmax(np.diff(wcss)) + 1

    # Apply kmeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)

    # Add the cluster labels to the dataframe
    melted_data['Cluster'] = kmeans.labels_

    # Filter the data for India and the series code 'NY.GNP.MKTP.CD'
    series_data = melted_data[(melted_data['Country Name'] == 'India') &
                                   (melted_data['Series Code'] == series_a)]
                                #    (melted_data['Series Code'] == 'NY.GNP.MKTP.CD')]

    # Sort the data by year
    series_data.sort_values('Year', inplace=True)

    # Plot the cluster data for India across all years
    plt.figure(figsize=(14, 7))
    plt.scatter(series_data['Year'], series_data['Value'], c=series_data['Cluster'], cmap='viridis', marker='o')
    plt.title('Cluster Data for India (NY.GNP.MKTP.CD) - All Years')
    plt.xlabel('Year')
    plt.ylabel('GNP (Current US$)')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    return plt.show()


def polynomial_confi(df, country_code ,series_code):
    
    # Load the data
    # df = pd.read_csv('data.csv', encoding='ascii')

    # Prepare the data for India with the specified Series Code
    # country_data = df[(df['Country Code'] == 'IND') & (df['Series Code'] == 'NY.GNP.MKTP.CD')]
    country_data = df[(df['Country Code'] == country_code) & (df['Series Code'] == series_code)]

    # Drop the non-numeric columns to prepare for plotting
    country_data_numeric = country_data.drop(columns=['Series Name', 'Series Code', 'Country Name', 'Country Code'])

    # Convert the year columns to numeric, errors='coerce' will replace non-numeric values with 
    country_data_numeric = country_data_numeric.apply(pd.to_numeric, errors='coerce')

    # Drop rows with any  values
    country_data_numeric = country_data_numeric.dropna(axis=1)

    # Extract years and values for plotting
    years = np.array([int(year[:4]) for year in country_data_numeric.columns]).reshape(-1, 1)
    values = country_data_numeric.values.reshape(-1, 1)


    # Calculate the linear regression parameters
    slope, intercept, r_value, p_value, std_err = linregress(years.ravel(), values.ravel())

    # Create a function for the line of best fit
    fit = lambda x: slope * x + intercept

    # Calculate the confidence interval
    # The factor 1.96 is for 95% confidence interval
    confidence_interval = 1.96 * std_err

    # Calculate the upper and lower bounds of the confidence interval
    lower_bound = fit(years) - confidence_interval
    upper_bound = fit(years) + confidence_interval

    # Plotting the scatter plot again
    plt.figure(figsize=(10, 6))
    plt.scatter(years, values, color='blue', label='Actual Data')
    
    # Fit a polynomial regression model
    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(years)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, values)
    y_poly_pred = poly_model.predict(X_poly)
    
    # Calculate R2 score for the polynomial model
    r2 = r2_score(values, y_poly_pred)


    # Plot the polynomial regression line
    plt.plot(years, y_poly_pred, color='red', label=f'Polynomial Degree {degree} Fit (R2={r2:.2f})')

    # Plot the confidence interval
    plt.fill_between(years.ravel(), lower_bound.ravel(), upper_bound.ravel(), color='green', alpha=0.5, label='95% Confidence Interval')

    # Format the x-axis to show years without decimal points
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Gross National Product (Current US$)')
    plt.title('Gross National Product of India Over Time with Confidence Interval')
    plt.legend()

    # Show plot with best fitting function and confidence rate
    return plt.show()

def plot_prediction(data, series_code, country_name):
    # Filter for United Kingdom and the specific series code
    # 'NY.GNP.MKTP.CD'
    # data = data[(data['Country Name'] == 'United Kingdom') & (data['Series Code'] == series_code)]
    data = data[(data['Country Name'] == country_name) & (data['Series Code'] == series_code)]

    # Clean the year column names by removing the extra text
    country_data_numeric = data.drop(columns=['Country Name', 'Country Code', 'Series Name', 'Series Code'])
    country_data_numeric.columns = [col.split()[0] for col in country_data_numeric.columns]

    # Convert the year columns to integers and values to numeric
    country_data_numeric = country_data_numeric.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    years_country = np.array([int(year) for year in country_data_numeric.columns]).reshape(-1, 1)
    values_country = country_data_numeric.values.reshape(-1, 1)

    # Fit polynomial regression model
    degree = 3
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly_country = poly_reg.fit_transform(years_country)
    poly_model_country = LinearRegression()
    poly_model_country.fit(X_poly_country, values_country)
    y_poly_pred_country = poly_model_country.predict(X_poly_country)

    # Predict for the next 20 years
    future_years_country = np.arange(years_country[-1] + 1, years_country[-1] + 21).reshape(-1, 1)
    future_values_country = poly_model_country.predict(poly_reg.transform(future_years_country))

    # Calculate confidence interval
    mse_country = mean_squared_error(values_country, y_poly_pred_country)
    std_dev_country = np.sqrt(mse_country)
    confidence_range_country = 1.96 * std_dev_country

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(years_country, values_country, color='blue', label='Actual Data')
    plt.plot(years_country, y_poly_pred_country, color='red', label=f'Polynomial Degree {degree} Fit')
    plt.fill_between(years_country.ravel(), (y_poly_pred_country - confidence_range_country).ravel(), (y_poly_pred_country + confidence_range_country).ravel(), color='pink', alpha=0.5, label='95% Confidence Interval')
    plt.plot(future_years_country, future_values_country, color='green', linestyle='--', label='Predicted Future Values')
    plt.fill_between(future_years_country.ravel(), (future_values_country - confidence_range_country).ravel(), (future_values_country + confidence_range_country).ravel(), color='yellow', alpha=0.5, label='Future 95% Confidence Interval')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel('Year')
    plt.ylabel('Gross National Product (Current US$)')
    plt.title(f'Gross National Product of {country_name} Over Time with Predictions')
    plt.legend()
    return plt.show()


def plot_country_confidence(country_name, series_code, title):
    # Filter the data for India and the series code 'NY.GNP.MKTP.CD'
    selected_country = data[(data['Country Name'] == country_name) & (data['Series Code'] == series_code)]

    # Drop the non-year columns to focus on the yearly data
    sel_columns = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
    selected_country_years = selected_country.drop(columns=sel_columns)


    # Since there was an error with 'Unnamed: 0', we will exclude it from the drop
    selected_country_years = selected_country.drop(columns=sel_columns)

    # Transpose the data to have years as rows
    selected_country_transposed = selected_country_years.transpose()
    selected_country_transposed.columns = ['GNP']

    # Reset index to have years as a column
    selected_country_transposed.reset_index(inplace=True)
    selected_country_transposed.rename(columns={'index': 'Year'}, inplace=True)

    # Extract year from the column name and convert to integer
    selected_country_transposed['Year'] = selected_country_transposed['Year'].str.extract('(\d+)').astype(int)

    # Convert GNP column to numeric, coerce errors to NaN
    selected_country_transposed['GNP'] = pd.to_numeric(selected_country_transposed['GNP'], errors='coerce')

    # Drop rows with NaN values
    selected_country_cleaned = selected_country_transposed.dropna()

    # Fit the linear model to the data
    popt, pcov = curve_fit(linear_fit, selected_country_cleaned['Year'], selected_country_cleaned['GNP'])

    # Generate the confidence range
    years = np.array(selected_country_cleaned['Year'])
    GNP, GNP_lower, GNP_upper = err_ranges(years, popt, pcov)

    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(selected_country_cleaned['Year'], selected_country_cleaned['GNP'], 'o', label='Data')
    plt.plot(years, GNP, '-', label='Fit')
    plt.fill_between(years, GNP_lower, GNP_upper, color='gray', alpha=0.2, label='Confidence range')

    # Label the plot
    plt.xlabel('Year')
    plt.ylabel('GNP (current US$)')
    plt.title('GNP of India with Confidence Range')
    plt.legend()

    # Show the plot
    return plt.show()

def plot_gni(df):


    # Replace '..' with NaN to handle non-numeric values
    df.replace('..', np.nan, inplace=True)

    # Filter the data for the series code 'NY.GNP.MKTP.CD' and 'EN.ATM.CO2E.KT'
    filtered_data = df[(df['Series Code'] == 'NY.GNP.MKTP.CD') | (df['Series Code'] == 'EN.ATM.CO2E.KT')]

    # Extract the year columns
    year_columns = [col for col in filtered_data.columns if 'YR' in col]

    # Melt the dataframe to have years and values in separate columns
    melted_data = filtered_data.melt(id_vars=['Country Name', 'Series Code'], value_vars=year_columns, var_name='Year', value_name='Value')

    # Drop rows with missing values
    melted_data.dropna(inplace=True)

    # Convert year to numerical format and values to float
    melted_data['Year'] = melted_data['Year'].str.extract('(\d{4})').astype(int)
    melted_data['Value'] = melted_data['Value'].astype(float)

    # Prepare data for kmeans clustering
    X = melted_data[['Year', 'Value']].values

    # Determine the number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Find the optimal number of clusters based on the elbow method
    optimal_clusters = np.argmax(np.diff(wcss)) + 1

    # Apply kmeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)

    # Add the cluster labels to the dataframe
    melted_data['Cluster'] = kmeans.labels_

    # Filter the data for India and the series code 'NY.GNP.MKTP.CD'
    india_gnp_data = melted_data[(melted_data['Country Name'] == 'India') & (melted_data['Series Code'] == 'NY.GNP.MKTP.CD')]

    # Sort the data by year
    india_gnp_data.sort_values('Year', inplace=True)

    # Plot the cluster data for India across all years
    plt.figure(figsize=(14, 7))
    plt.scatter(india_gnp_data['Year'], india_gnp_data['Value'], c=india_gnp_data['Cluster'], cmap='viridis', marker='o')
    plt.title('Data for India GNI (current US$) over time')
    plt.xlabel('Year')
    plt.ylabel('GNP (Current US$)')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    return plt.show()

if __name__ == "__main__":
    print('if __name__ == "__main__":')
    
    # Load data into pandas dataframe from CSV file
    data = convert_csv('./data.csv')
    
    print('Dataframe Head:')
    data.head()

    # Inspect the dataset to identify numerical columns and check for NaNs
    print('Dataframe info:')
    print(data.info())

    print('Statistical summary:')
    print(data.describe())

    data_dict = series_dict(data)
    data_dict['EG.ELC.ACCS.ZS']

    plot_scatter(data, 'SP.POP.TOTL', 'SL.TLF.TOTL.IN', 'Access to Electricity vs Total Population')

    # k_clustering(data, 'EN.ATM.CO2E.KT', 'SP.POP.TOTL', 3)
    k_clustering(data, 'SL.TLF.TOTL.IN', 'SP.POP.TOTL', 3)

    series_codes = ['EG.ELC.ACCS.ZS', 'SP.POP.TOTL', 'SL.TLF.TOTL.IN', 'EN.ATM.METH.KT.CE', 'EN.ATM.CO2E.KT', 'AG.LND.ARBL.ZS', 
                    'AG.LND.FRST.ZS', 'NY.GDP.TOTL.RT.ZS', 'SP.URB.TOTL.IN.ZS', 'BG.GSR.NFSV.GD.ZS', 'NY.GDP.MKTP.KN', 'NY.GNP.MKTP.CD']

    draw_clustered_data(data,series_codes)

    elbow_method(data)

    normal_range_line(data, 'NY.GNP.MKTP.CD', 'EN.ATM.CO2E.KT')

    polynomial_confi(data, 'IND', 'NY.GNP.MKTP.CD')

    plot_prediction(data, 'NY.GNP.MKTP.CD', 'United Kingdom')

    # DEMOOOOOOOOOO
    country_name = 'United Kingdom'
    series_code = 'SL.TLF.TOTL.IN'
    title = ''
    plot_country_confidence(country_name, series_code, title)


    plot_gni(data)
