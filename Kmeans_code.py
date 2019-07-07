import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import silhouette_samples

dataframes = []
for i in range(1,57):
    file_name = "/home/<user>/Desktop/python-machine-learning-/{}.csv".format(i)
    df = pd.read_csv(file_name, header=None, index_col=False)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    df.dropna(axis=0)
    df.dropna(axis=1)
    dataframes.append(df)

normalized_dataframes = []
for i in range(0,56):
    df = dataframes[i]
    normalized_df = (df-df.min())/(df.max()-df.min())
    normalized_df.fillna(0, inplace=True)
    normalized_dataframes.append(normalized_df)

sses = []
silhouette_scores = []
for i in range(0, 56):
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 0)
    df = normalized_dataframes[i]
    predictions = kmeans.fit_predict(df)
    silhouette_score = sc(df, predictions, metric='euclidean')
    sses.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score)

sse_df = pd.DataFrame(sses)
silhouette_scores_df = pd.DataFrame(silhouette_scores)

sse_df.to_excel(excel_writer="/home/<user>/Desktop/python-machine-learning-/1_to_20/sse.xlsx", header=None, index=True)
silhouette_scores_df.to_excel(excel_writer="/home/zahed/Desktop/python-machine-learning-/1_to_20/silhouette_scores.xlsx", header=None, index=True)

# Code for generating silhouette coefficients of individual samples in all datasets.
for i in range(0, 56):
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 0)
    labels = kmeans.fit_predict(normalized_dataframes[i])
    coeff = silhouette_samples(normalized_dataframes[i], labels)
    coeff_df = pd.DataFrame(coeff)
    file_name = "/home/<user>/Desktop/python-machine-learning-/1_to_20/Coefficients/silhouette_coeffs_{}.xlsx".format(i+1)
    coeff_df.to_excel(excel_writer=file_name, header=None, index=True)
