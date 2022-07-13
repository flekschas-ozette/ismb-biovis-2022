import jscatter
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from umap import UMAP

def prepare(df, num_markers=-1, spread_factor=1000):
    suffix = '_faust_annotation'
    all_markers = [c[:-len(suffix)] for c in list(df.columns) if c.endswith(suffix)]
    
    markers = all_markers[:num_markers] if num_markers > 0 else all_markers
    
    df['complete_faust_label'] = ''
    for marker in markers:
        df['complete_faust_label'] += marker + df[f'{marker}_faust_annotation']
    
    expression_levels = {
        l: i * spread_factor
        for i, l
        in enumerate(df[f'{markers[0]}_faust_annotation'].unique())
    }
    
    df.sort_values(by=['faustLabels'], ignore_index=True, inplace=True)
    
    return (markers, expression_levels, df[markers].values)

def transform(
    df,
    markers,
    expression_levels,
    log=False
):
    faust_labels = df.complete_faust_label.unique()

    marker_annotation_cols = [f'{m}_faust_annotation' for m in markers]

    embedding_expression = df[markers].values.copy()

    t = 0

    # For each cluster (i.e., cell phenotype defined by the FAUST label)
    for i, faust_label in enumerate(faust_labels):
        if log and i % 1000 == 0:
            t = time()
            print(f'Transform {i}-{i + 999} of {len(faust_labels)} clusters... ', end='')

        # First, we get the indices of all data points belonging to the cluster (i.e., cell phenotype)
        idxs = df.query(f'complete_faust_label == "{faust_label}"').index

        # 1. We winsorize the expression values to [0.01, 99.9]
        embedding_expression[idxs] = winsorize(
            embedding_expression[idxs],
            limits=[0.01, 0.01],
            axis=0,
        )

        # 2. Then we standardize the expression values
        # to have zero mean and unit standard deviation
        mean = embedding_expression[idxs].mean(axis=0)
        sd = np.nan_to_num(embedding_expression[idxs].std(axis=0))
        sd[sd == 0] = 1

        embedding_expression[idxs] -= mean
        embedding_expression[idxs] /= sd

        # 3. Next, we translate the expressions values based on their expression levels
        embedding_expression[idxs] += df.iloc[idxs[0]][marker_annotation_cols].map(
            expression_levels
        ).values

        if log and (i % 1000 == 999 or i == len(faust_labels) - 1):
            print(f'done! ({round(time() - t)}s)')
            
    return embedding_expression

def to_df(df, xy, save_as=None):    
    df_embedding = pd.concat(
        [
            pd.DataFrame(xy, columns=['x', 'y']),
            pd.DataFrame(df.complete_faust_label.values, columns=['cellType']),
            df,
        ],
        axis=1
    )
    df_embedding.cellType = df_embedding.cellType.where(
        df.faustLabels != '0_0_0_0_0',
        '0_0_0_0_0'
    ).astype('category')

    if save_as is not None:
        df_embedding.to_parquet(f'data/{save_as}.pq', compression='gzip')

    return df_embedding

def embed(df, data, embeddor, save_as=None):    
    return to_df(df, embeddor.fit_transform(data), save_as=save_as)

def embed_sample(df, data, embeddor, save_as=None):
    df.complete_faust_label.unique()
    return to_df(df, embeddor.fit_transform(data), save_as=save_as)

def sort_y(df_target, df_source, sort_prop):
    return pd.concat([df_a, df_b], axis=1)

def transform_embed(
    dfs,
    embeddor=UMAP,
    save_as=None,
    num_markers=-1,
    spread_factor=1000,
    embeddor_pca_init=False,
    embeddor_random_state=None,
    embed_raw=False,
    log=False,
):
    if not isinstance(dfs, list):
        dfs = [dfs]
        
    dfs = [df.copy() for df in dfs]
        
    expressions = []
    
    for df in dfs:
        markers, expression_levels, expression = prepare(df, num_markers, spread_factor)

        if not embed_raw:
            expression = transform(df, markers, expression_levels, log=log)
            
        expressions.append(expression)
        
    expressions = np.vstack(expressions)
    
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    
    embeddor_kwargs = {}
    
    if embeddor_random_state is not None:
        embeddor_kwargs['random_state'] = embeddor_random_state
    
    if embeddor_pca_init:
        embeddor_kwargs['init'] = PCA(n_components=2).fit_transform(
            df[[f'{m}_Windsorized' for m in markers]].values
        )
    
    return embed(df, expressions, embeddor(**embeddor_kwargs), save_as=save_as)
    