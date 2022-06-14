import jscatter
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from cmaps import glasbey_dark
from umap import UMAP
import matplotlib.pyplot as plt

data = pd.read_csv('data/Nature_Tumor_output.csv')
markers = ['CD4', 'CD8', 'CD3', 'CD45RA', 'CD27', 'CD19', 'CD103', 'CD28', 'CD69', 'PD1', 'HLADR', 'GranzymeB', 'CD25', 'ICOS', 'TCRgd', 'CD38', 'CD127', 'Tim3']

raw_expression = data[markers].values
data['complete_faust_label'] = 'CD4' + data['CD4_faust_annotation'] + 'CD8' + data['CD8_faust_annotation'] + 'CD3' + data['CD3_faust_annotation'] + 'CD45RA' + data['CD45RA_faust_annotation'] + 'CD27' + data['CD27_faust_annotation'] + 'CD19' + data['CD19_faust_annotation'] + 'CD103' + data['CD103_faust_annotation'] + 'CD28' + data['CD28_faust_annotation'] + 'CD69' + data['CD69_faust_annotation'] + 'PD1' + data['PD1_faust_annotation'] + 'HLADR' + data['HLADR_faust_annotation'] + 'GranzymeB' + data['GranzymeB_faust_annotation'] + 'CD25' + data['CD25_faust_annotation'] + 'ICOS' + data['ICOS_faust_annotation'] + 'TCRgd' + data['TCRgd_faust_annotation'] + 'CD38' + data['CD38_faust_annotation'] + 'CD127' + data['CD127_faust_annotation'] + 'Tim3' + data['Tim3_faust_annotation']

data[['faustLabels', 'complete_faust_label']].head(5)


expression_levels = list(data.CD3_faust_annotation.unique())
faust_labels = data.complete_faust_label.unique()
num_faust_labels = len(faust_labels)
t = 0
std_scaler = StandardScaler()
expression_level_translation = {
    '-': 0,
    '+': 1000,
}
embedding_expression = raw_expression.copy()
# 1. Normalize expressions for each phenotype to zero mean and unit variance
for i, faust_label in enumerate(faust_labels):
    if i % 1000 == 0:
        t = time()
        print(f'Transform {i}-{i + 999} of {len(faust_labels)} clusters...', end='\n')
    # 1a. We get the indices of all data points belonging to the same complete FAUST label\n",
    idxs = data.query(f'complete_faust_label == \"{faust_label}\"').index
    # 1b. Then we windsorize the expression values of all the data points for the FAUST label
    for m, marker in enumerate(markers):
        if (embedding_expression[idxs, m].shape[0] <= 1):
            next
        embedding_expression[idxs, m] = winsorize(embedding_expression[idxs, m], limits=[0.01, 0.01])
    # 1c. Then we normalize the expression values of all the data points        
    for m, marker in enumerate(markers):
        mean_val = np.mean(embedding_expression[idxs, m])
        sd_val = np.std(embedding_expression[idxs, m])
        if ((sd_val == 0) or (np.isnan(sd_val))):
            embedding_expression[idxs, m] = (embedding_expression[idxs, m] - mean_val)
        else:
            embedding_expression[idxs, m] = ((embedding_expression[idxs, m] - mean_val)/sd_val)
    # 2. Scale marker expressions based on the expression levels
    for m, marker in enumerate(markers):
        # 2a. Retrieve the expression level of a marker\n",
        expression_level = data.iloc[idxs[0]][f'{marker}_faust_annotation']
        # 2b. Further scale the normalized expression values respective new feature range
        embedding_expression[idxs, m] += expression_level_translation[expression_level]
        # 2c. Add perturbation to avoid exact ties
        perturb = np.random.normal(0, 0.00001, len(idxs))
        embedding_expression[idxs, m] += perturb
    if i % 1000 == 999:
        print(f'done! ({round(time() - t)}s)')

embedding = UMAP(n_neighbors=15, min_dist=0.2, metric='euclidean').fit_transform(embedding_expression)
df_embedding = pd.concat(
    [data.faustLabels, pd.DataFrame(embedding, columns=['umapX', 'umapY'])],
    axis=1
)
df_embedding.faustLabels = df_embedding.faustLabels.astype('category')
df_embedding.head()
plt.scatter(data=df_embedding, x='umapX', y='umapY')
plt.show()

