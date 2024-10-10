import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("GSE45827_series_matrix.txt", sep="\t", comment='!', header=None) # index_col=0
print(df.head())
df_clean = df.drop(columns=[0]).apply(pd.to_numeric, errors='coerce')

df_clean = df_clean.dropna()
print(df_clean.head())
print(df_clean.describe())
# df_clean = df.dropna()
# print(df_clean.head())
# print(df_clean.describe())

sns.heatmap(df_clean, cmap="viridis")
plt.show()

# Separate features (X) and target (y) for supervised learning
# For example, let's predict the expression of the first gene based on others
X = df_clean.drop(df_clean.columns[0], axis=1)  # All genes except the first one
y = df_clean[df_clean.columns[0]]  # Target gene expression

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# UNSUPERVISED LEARNING

# Perform PCA to reduce dimensionality (optional but useful for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_pca)

# Visualize the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering of Gene Expression")
plt.show()
