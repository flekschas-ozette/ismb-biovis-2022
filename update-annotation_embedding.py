import jscatter
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from cmaps import glasbey_dark
from umap import UMAP
import matplotlib.pyplot as plt

data = pd.read_csv('data/MMRF_1267_output.csv')
#markers = ['CD3', 'HLADR', 'NKG2D', 'CD8', 'CD45', 'CD4', 'CD5', 'CD14', 'CD19', 'CD127', 'CD11b', 'CD25', 'CD45RO', 'CD33', 'CD38', 'CD27', 'CD11c', 'CD45RA', 'CD66b', 'CCR7', 'GranzymeB', 'CD39', 'CD28', 'PD1', 'CD57', 'CD123', 'CD16', 'TIGIT', 'TIM3__CD366_', 'CD56', 'KLRG1', 'Ki67', 'PDL1', 'Tbet', 'ICOS', 'NKG2A', 'CD138', 'CD1c', 'DNAM1']
markers = ['CD3', 'HLADR', 'NKG2D', 'CD8', 'CD45', 'CD4', 'CD5', 'CD14', 'CD19', 'CD127', 'CD11b', 'CD25', 'CD45RO', 'CD33', 'CD38', 'CD27', 'CD11c', 'CD45RA', 'CD66b', 'CCR7', 'GranzymeB', 'CD39', 'CD28']

raw_expression = data[markers].values
#data['complete_faust_label'] = 'CD3' + data['CD3_faust_annotation'] + 'HLADR' + data['HLADR_faust_annotation'] + 'NKG2D' + data['NKG2D_faust_annotation'] + 'CD8' + data['CD8_faust_annotation'] + 'CD45' + data['CD45_faust_annotation'] + 'CD4' + data['CD4_faust_annotation'] + 'CD5' + data['CD5_faust_annotation'] + 'CD14' + data['CD14_faust_annotation'] + 'CD19' + data['CD19_faust_annotation'] + 'CD127' + data['CD127_faust_annotation'] + 'CD11b' + data['CD11b_faust_annotation'] + 'CD25' + data['CD25_faust_annotation'] + 'CD45RO' + data['CD45RO_faust_annotation'] + 'CD33' + data['CD33_faust_annotation'] + 'CD38' + data['CD38_faust_annotation'] + 'CD27' + data['CD27_faust_annotation'] + 'CD11c' + data['CD11c_faust_annotation'] + 'CD45RA' + data['CD45RA_faust_annotation'] + 'CD66b' + data['CD66b_faust_annotation'] + 'CCR7' + data['CCR7_faust_annotation'] + 'GranzymeB' + data['GranzymeB_faust_annotation'] + 'CD39' + data['CD39_faust_annotation'] + 'CD28' + data['CD28_faust_annotation'] + 'PD1' + data['PD1_faust_annotation'] + 'CD57' + data['CD57_faust_annotation'] + 'CD123' + data['CD123_faust_annotation'] + 'CD16' + data['CD16_faust_annotation'] + 'TIGIT' + data['TIGIT_faust_annotation'] + 'TIM3__CD366_' + data['TIM3__CD366__faust_annotation'] + 'CD56' + data['CD56_faust_annotation'] + 'KLRG1' + data['KLRG1_faust_annotation'] + 'Ki67' + data['Ki67_faust_annotation'] + 'PDL1' + data['PDL1_faust_annotation'] + 'Tbet' + data['Tbet_faust_annotation'] + 'ICOS' + data['ICOS_faust_annotation'] + 'NKG2A' + data['NKG2A_faust_annotation'] + 'CD138' + data['CD138_faust_annotation'] + 'CD1c' + data['CD1c_faust_annotation'] + 'DNAM1' + data['DNAM1_faust_annotation']
data['complete_faust_label'] = 'CD3' + data['CD3_faust_annotation'] + 'HLADR' + data['HLADR_faust_annotation'] + 'NKG2D' + data['NKG2D_faust_annotation'] + 'CD8' + data['CD8_faust_annotation'] + 'CD45' + data['CD45_faust_annotation'] + 'CD4' + data['CD4_faust_annotation'] + 'CD5' + data['CD5_faust_annotation'] + 'CD14' + data['CD14_faust_annotation'] + 'CD19' + data['CD19_faust_annotation'] + 'CD127' + data['CD127_faust_annotation'] + 'CD11b' + data['CD11b_faust_annotation'] + 'CD25' + data['CD25_faust_annotation'] + 'CD45RO' + data['CD45RO_faust_annotation'] + 'CD33' + data['CD33_faust_annotation'] + 'CD38' + data['CD38_faust_annotation'] + 'CD27' + data['CD27_faust_annotation'] + 'CD11c' + data['CD11c_faust_annotation'] + 'CD45RA' + data['CD45RA_faust_annotation'] + 'CD66b' + data['CD66b_faust_annotation'] + 'CCR7' + data['CCR7_faust_annotation'] + 'GranzymeB' + data['GranzymeB_faust_annotation'] + 'CD39' + data['CD39_faust_annotation'] + 'CD28' + data['CD28_faust_annotation']


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

