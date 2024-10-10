# Gene Expression Prediction Using Machine Learning

## Project Objective
This project uses machine learning techniques to predict gene expression levels from the GSE45827 dataset. It also aims to explore gene expression patterns using unsupervised learning (Kmeans clustering).

## Data Preprocessing
1. Loaded dataset using `pandas` and removed any rows containing NaN values
2. Normalized dataset using `StandardScaler` to ensure all features are on same scale
3. Split dataset into training and testing subsets to evaluate model performance

## Model Selection
1. Applied **Linear Regression** to predict the expression levels of a target gene based on other genes in dataset
2. Utilized **Kmeans Clustering** to group gene expression profiles into distinct clusters
3. [Expand, compare future models such as Random Forest, NN, ?]

## Results
1. Visualized the cleaned dataset using a **heatmap**
2. Evaluated model performance using Mean Squared Error (MSE): [INSERT ONCE DONE]
3. Clustered gene expression patterns and visualized using K-means clustering
