import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from fastapi import FastAPI, Path, Request, File, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from database import products_db
from random import sample
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import shutil


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    expose_headers=["*"]
)

# Global variables
global customer_data_cleaned, outliers_data, df
customer_data_cleaned, outliers_data, df = None, None, None


def clean_and_train():
    # Load data
    df = pd.read_csv('data_with_categories.csv', encoding="ISO-8859-1")

    # Preprocessing
    df = df.dropna(subset=['CustomerID', 'Description'])
    df.drop_duplicates(inplace=True)
    df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')
    df = df[~df['StockCode'].isin(
        [code for code in df['StockCode'].unique() if sum(c.isdigit() for c in str(code)) in (0, 1)])]
    df = df[~df['Description'].isin(["Next Day Carriage", "High Resolution Image"])]
    df['Description'] = df['Description'].str.upper()
    df = df[df['UnitPrice'] > 0]
    df.reset_index(drop=True, inplace=True)

    # Additional features
    # Days since last purchase

    # Convert InvoiceDate to datetime type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Convert InvoiceDate to datetime and extract only the date
    df['InvoiceDay'] = df['InvoiceDate'].dt.date

    # Find the most recent purchase date for each customer
    customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()

    # Find the most recent date in the entire dataset
    most_recent_date = df['InvoiceDay'].max()

    # Convert InvoiceDay to datetime type before subtraction
    customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
    most_recent_date = pd.to_datetime(most_recent_date)

    # Calculate the number of days since the last purchase for each customer
    customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

    # Remove the InvoiceDay column
    customer_data.drop(columns=['InvoiceDay'], inplace=True)

    # Total transactions and total products purchased by customer

    # Calculate the total number of transactions made by each customer
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

    # Calculate the total number of products purchased by each customer
    # total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    # total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
    # customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

    # Total spend and average transaction value

    # Calculate the total spend by each customer
    # df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
    # total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()

    # Calculate the average transaction value for each customer
    # average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
    # average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / \
    #                                                          average_transaction_value['Total_Transactions']

    # Merge the new features into the customer_data dataframe
    # customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
    # customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']],
    #                          on='CustomerID')

    # Product Diversity

    # Calculate the number of unique products purchased by each customer
    unique_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)

    # Merge the new feature into the customer_data dataframe
    customer_data = pd.merge(customer_data, unique_products_purchased, on='CustomerID')

    # Average Days Between Purchases and Favorite Shopping Day and Favorite Shopping Hour

    # Extract day of week and hour from InvoiceDate
    # df['Day_Of_Week'] = df['InvoiceDate'].dt.dayofweek
    # df['Hour'] = df['InvoiceDate'].dt.hour

    # Calculate the average number of days between consecutive purchases
    # days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(
    #     lambda x: (x.diff().dropna()).apply(lambda y: y.days))
    # average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
    # average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

    # Find the favorite shopping day of the week
    # favorite_shopping_day = df.groupby(['CustomerID', 'Day_Of_Week']).size().reset_index(name='Count')
    # favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][
    #     ['CustomerID', 'Day_Of_Week']]

    # Find the favorite shopping hour of the day
    # favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
    # favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][
    #     ['CustomerID', 'Hour']]

    # Merge the new features into the customer_data dataframe
    # customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
    # customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
    # customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')

    # Customer from UK or outside

    # Group by CustomerID and Country to get the number of transactions per country for each customer
    customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')

    # Get the country with the maximum number of transactions for each customer (in case a customer has transactions from multiple countries)
    customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates(
        'CustomerID')

    # Create a binary column indicating whether the customer is from the UK or not
    customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

    # Merge this data with our customer_data dataframe
    customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'Is_UK']], on='CustomerID', how='left')

    # Cancelation fraquency and cancelation rate

    # Calculate the total number of transactions made by each customer
    # total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

    # Calculate the number of cancelled transactions for each customer
    # cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
    # cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    # cancellation_frequency.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

    # Merge the Cancellation Frequency data into the customer_data dataframe
    # customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left')

    # Replace NaN values with 0 (for customers who have not cancelled any transaction)
    # customer_data['Cancellation_Frequency'].fillna(0, inplace=True)

    # Calculate the Cancellation Rate
    # customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

    # Monthly spending mean and Monthly spending std and spending trend

    # Extract month and year from InvoiceDate
    # df['Year'] = df['InvoiceDate'].dt.year
    # df['Month'] = df['InvoiceDate'].dt.month

    # Calculate monthly spending for each customer
    # monthly_spending = df.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend'].sum().reset_index()

    # Calculate Seasonal Buying Patterns: We are using monthly frequency as a proxy for seasonal buying patterns
    # seasonal_buying_patterns = monthly_spending.groupby('CustomerID')['Total_Spend'].agg(['mean', 'std']).reset_index()
    # seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'},
    #                                 inplace=True)

    # Replace NaN values in Monthly_Spending_Std with 0, implying no variability for customers with single transaction month
    # seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

    # Calculate Trends in Spending
    # We are using the slope of the linear trend line fitted to the customer's spending over time as an indicator of spending trends
    # def calculate_trend(spend_data):
    #     # If there are more than one data points, we calculate the trend using linear regression
    #     if len(spend_data) > 1:
    #         x = np.arange(len(spend_data))
    #         slope, _, _, _, _ = linregress(x, spend_data)
    #         return slope
    #     # If there is only one data point, no trend can be calculated, hence we return 0
    #     else:
    #         return 0

    # Apply the calculate_trend function to find the spending trend for each customer
    # spending_trends = monthly_spending.groupby('CustomerID')['Total_Spend'].apply(calculate_trend).reset_index()
    # spending_trends.rename(columns={'Total_Spend': 'Spending_Trend'}, inplace=True)

    # Merge the new features into the customer_data dataframe
    # customer_data = pd.merge(customer_data, seasonal_buying_patterns, on='CustomerID')
    # customer_data = pd.merge(customer_data, spending_trends, on='CustomerID')

    # Display the first few rows of the customer_data dataframe
    # customer_data.head()

    # Changing the data type of 'CustomerID' to string as it is a unique identifier and not used in mathematical operations
    # customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)

    # Convert data types of columns to optimal types
    # customer_data = customer_data.convert_dtypes()

    # Buys in categories

    # Zliczanie zakupów w każdej kategorii dla każdego klienta
    category_purchases = df.groupby(['CustomerID', 'Category']).size().unstack(fill_value=0)

    # Zmiana nazw kolumn na 'buysInCategoryX'
    category_purchases.columns = ['buysInCat' + str(i) for i in range(1, len(category_purchases.columns) + 1)]

    # Resetowanie indeksu
    category_purchases.reset_index(inplace=True)

    # Dołączanie zliczonych zakupów do DataFrame klientów
    customer_data = pd.merge(customer_data, category_purchases, on='CustomerID', how='left').fillna(0)

    customer_data = customer_data.drop(columns=['Days_Since_Last_Purchase'])

    # Outlier detection
    model = IsolationForest(contamination=0.05, random_state=0)
    customer_data['Outlier_Scores'] = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())
    customer_data['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_data['Outlier_Scores']]
    outliers_data = customer_data[customer_data['Is_Outlier'] == 1]
    customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0].drop(
        columns=['Outlier_Scores', 'Is_Outlier'])
    customer_data_cleaned.reset_index(drop=True, inplace=True)

    # Scaling features

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # List of columns that don't need to be scaled
    columns_to_exclude = ['CustomerID', 'Is_UK', 'Day_Of_Week']

    # List of columns that need to be scaled
    columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

    # Copy the cleaned dataset
    customer_data_scaled = customer_data_cleaned.copy()

    # Applying the scaler to the necessary columns in the dataset
    customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

    # Display the first few rows of the scaled data
    customer_data_scaled.head()

    # Dimensional reeduction

    # Setting CustomerID as the index column
    customer_data_scaled.set_index('CustomerID', inplace=True)

    # Apply PCA
    pca = PCA().fit(customer_data_scaled)

    # Calculate the Cumulative Sum of the Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Set the optimal k value (based on our analysis, we can choose 6)
    optimal_k = 6

    # Creating and using PCA

    # Creating a PCA object with 6 components
    pca = PCA(n_components=6)

    # Fitting and transforming the original data to the new PCA dataframe
    customer_data_pca = pca.fit_transform(customer_data_scaled)

    # Creating a new dataframe from the PCA dataframe, with columns labeled PC1, PC2, etc.
    customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC' + str(i + 1) for i in range(pca.n_components_)])

    # Adding the CustomerID index back to the new PCA dataframe
    customer_data_pca.index = customer_data_scaled.index

    # Apply KMeans clustering using the optimal k
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=100, random_state=0)
    kmeans.fit(customer_data_pca)

    # Get the frequency of each cluster
    cluster_frequencies = Counter(kmeans.labels_)

    # Append the new cluster labels back to the original dataset
    customer_data_cleaned['cluster'] = kmeans.labels_

    # Append the new cluster labels to the PCA version of the dataset
    customer_data_pca['cluster'] = kmeans.labels_

    # Display the first few rows of the original dataframe
    customer_data_cleaned.head(20)

    return customer_data_cleaned, outliers_data, df  # Return processed data for further use


def recommend_products(selected_customer_id, customer_data_cleaned, df, outliers_data):
    # Step 1: Extract the CustomerIDs of the outliers and remove their transactions from the main dataframe
    outlier_customer_ids = outliers_data['CustomerID'].astype('float').unique()
    df_filtered = df[~df['CustomerID'].isin(outlier_customer_ids)]

    # Step 2: Ensure consistent data type for CustomerID across both dataframes before merging
    customer_data_cleaned['CustomerID'] = customer_data_cleaned['CustomerID'].astype('float')

    # Step 3: Merge the transaction data with the customer data to get the cluster information for each transaction
    merged_data = df_filtered.merge(customer_data_cleaned[['CustomerID', 'cluster']], on='CustomerID', how='inner')

    # Step 4: Identify the top 10 best-selling products in each cluster based on the total quantity sold
    best_selling_products = merged_data.groupby(['cluster', 'StockCode', 'Description','UnitPrice'])['Quantity'].sum().reset_index()
    best_selling_products = best_selling_products.sort_values(by=['cluster', 'Quantity'], ascending=[True, False])
    top_products_per_cluster = best_selling_products.groupby('cluster').head(20)

    # Step 5: Create a record of products purchased by each customer in each cluster
    customer_purchases = merged_data.groupby(['CustomerID', 'cluster', 'StockCode'])['Quantity'].sum().reset_index()

    # Step 6: Recommendation
    recommendations = []
    for cluster in top_products_per_cluster['cluster'].unique():
        top_products = top_products_per_cluster[top_products_per_cluster['cluster'] == cluster]
        if selected_customer_id in customer_data_cleaned[customer_data_cleaned['cluster'] == cluster][
            'CustomerID'].values:
            customer_purchased_products = customer_purchases[
                (customer_purchases['CustomerID'] == selected_customer_id) &
                (customer_purchases['cluster'] == cluster)]['StockCode'].tolist()

            top_products_not_purchased = top_products[~top_products['StockCode'].isin(customer_purchased_products)]
            if len(top_products_not_purchased) >= 4:
                top_3_products_not_purchased = top_products_not_purchased.sample(4)
            else:
                # If less then four take all
                top_3_products_not_purchased = top_products_not_purchased


            for _, row in top_3_products_not_purchased.iterrows():
                recommendations.append({
                    'customer_id': selected_customer_id,
                    'rec_stock_code': row['StockCode'],
                    'rec_descr': row['Description'],
                    'rec_price': row['UnitPrice']
                })

    return recommendations


# customer_data_cleaned, outliers_data, df = clean_and_train()
# customer_id = 15746  # Assuming this is defined somewhere in your code
# recommendations_for_customer = recommend_products(customer_id, customer_data_cleaned, df, outliers_data)
# print(recommendations_for_customer)

class CustomerIDRequest(BaseModel):
    customer_id: int


@app.get("/random-products/")
async def get_random_products():
    # Zwraca 50 losowych rekordów z products_db, ale nie więcej niż jest dostępnych
    num_records = min(50, len(products_db))
    random_products = sample(products_db, num_records)
    return random_products


@app.post("/clean-and-train/")
async def run_clean_and_train():
    try:
        global customer_data_cleaned, outliers_data, df
        customer_data_cleaned, outliers_data, df = clean_and_train()
        return {"message": "Data cleaning and training process completed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/recommend-products/id/{customer_id}")
async def run_recommend_products(customer_id: int = Path(..., title="The ID of the customer to get recommendations for")):
    try:
        global customer_data_cleaned, outliers_data, df
        recommendations = recommend_products(customer_id, customer_data_cleaned, df, outliers_data)
        print(recommendations)
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return HTMLResponse(content=open("static/index.html").read())

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Zapisz plik w głównym katalogu projektu pod stałą nazwą
        with open("data_with_categories.csv", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "File uploaded and saved successfully."}
    except Exception as e:
        return {"error": str(e)}

# Uvicorn start
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
