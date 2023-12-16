# IBM Data Engineering Professional Certificate Final Project
## Build an ML Pipeline for Airfoil Noise Prediction

### Project Overview
As a data engineer at an aeronautics consulting company, your role is crucial in facilitating the work of data scientists who focus on machine learning algorithms. This project utilizes a modified version of the NASA Airfoil Self Noise dataset. The objective is to perform ETL (Extract, Transform, Load) tasks, build a machine learning pipeline for sound level prediction, evaluate the model, and finally, persist the model for future use.

### Objectives
1. **ETL Activity (Part 1):**
   - Load a CSV dataset
   - Remove duplicate rows
   - Drop rows with null values
   - Perform transformations
   - Store the cleaned data in Parquet format

2. **Machine Learning Pipeline (Part 2):**
   - Load the cleaned data
   - Convert string columns to numeric types
   - Define the VectorAssembler and StandardScaler stages
   - Create a LinearRegression stage
   - Build the pipeline, split the data, and fit the pipeline

3. **Evaluate the Model (Part 3):**
   - Predict using the model
   - Evaluate the model using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2)

4. **Persist the Model (Part 4):**
   - Save the model to the path "Final_Project"
   - Load the model from the path "Final_Project"
   - Make predictions using the loaded model
   - Display the predictions

### Datasets
- Modified NASA Airfoil Self Noise dataset: [NASA airfoil self-noise dataset](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) (licensed under CC BY 4.0)

### Setup
- Libraries: PySpark, findspark
- Install required libraries using:
  ```python
  !pip install pyspark==3.1.2 -q
  !pip install findspark -q
Instructions
Part 1 - Perform ETL Activity:

Import required libraries
Create a Spark session
Load the CSV file into a DataFrame
Print top 5 rows and total number of rows
Drop duplicates and null values
Rename the "SoundLevel" column and save the cleaned DataFrame in Parquet format
Part 2 - Create a Machine Learning Pipeline:

Load the cleaned data
Convert string columns to numeric types
Define the VectorAssembler and StandardScaler stages
Create a LinearRegression stage
Build the pipeline, split the data, and fit the pipeline
Part 3 - Evaluate the Model:

Predict using the model
Print MSE, MAE, and R2
Part 4 - Persist the Model:

Save the model to "Final_Project"
Load the model, make predictions, and display results
Evaluation
Run the provided code cells in each part
Answer the final evaluation quiz based on the results
Authors
Ramesh Sannareddy
Contributors
[List of contributors, if any]
Change Log
[Record changes and updates made to the project]
"""
Write the content to a README file
with open("README.md", "w") as readme_file:
readme_file.write(readme_content)

Part 1 - Perform ETL Activity
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

Create a Spark session
spark = SparkSession.builder.appName("AirfoilSelfNoisePrediction").getOrCreate()

Load the CSV file into a DataFrame
df = spark.read.csv("NASA_airfoil_noise_raw.csv", header=True)

Print top 5 rows
df.show(5)

Print total number of rows
rowcount1 = df.count()
print(rowcount1)

Drop duplicates
df = df.dropDuplicates()

Print total number of rows after dropping duplicates
rowcount2 = df.count()
print("Total number of rows after dropping duplicates:", rowcount2)

Drop rows with null values
df = df.dropna()

Print total number of rows after dropping rows with null values
rowcount3 = df.count()
print("Total number of rows after dropping rows with null values:", rowcount3)

Rename the column "SoundLevel" to "SoundLevelDecibels"
df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")

Save the dataframe in parquet format
df.write.parquet("NASA_airfoil_noise_cleaned.parquet")

### Author
Abhishek Kumar Singh

