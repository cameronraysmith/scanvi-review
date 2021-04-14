---
jupyter:
  celltoolbar: Slideshow
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md,py:percent
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.2
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
---

<center><font size="+4">Seed labeling of single cell RNA sequencing data with variational autoencoders using scvi-tools</font></center>


# Introduction

<!-- #region {"colab_type": "text", "id": "p06lh96YIW7X"} -->
In this tutorial, we go through the steps of training scANVI for seed annotation. This is useful for when we have ground truth labels for a few cells and want to annotate unlabelled cells. For more information, please refer to the original [scANVI publication](https://www.biorxiv.org/content/biorxiv/early/2019/01/29/532895.full.pdf). 

Plan for this tutorial:

1. Loading the data
2. Creating the seed labels: groundtruth for a small fraction of cells
2. Training the scANVI model: transferring annotation to the whole dataset
3. Visualizing the latent space and predicted labels
<!-- #endregion -->

<!-- #region {"colab_type": "code", "id": "L-ThTcdj8ljr", "colab": {"base_uri": "https://localhost:8080/", "height": 382}, "outputId": "e8fec97d-596f-4d03-ce3c-a9f27d079d34"} -->
```python
import sys

#if True, will install via pypi, else will install from source
stable = True
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and stable:
    !pip install --quiet scvi-tools[tutorials]
elif IN_COLAB and not stable:
    !pip install --quiet --upgrade jsonschema
    !pip install --quiet git+https://github.com/yoseflab/scvi-tools@master#egg=scvi-tools[tutorials]
```
<!-- #endregion -->

```python colab_type="code" id="BouKibj8gMHT" colab={} tags=[]
import scanpy as sc
import numpy as np
from scipy import sparse

import torch
import scvi

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline
```

```python tags=[]
%run -i 'plotting.py'
```

<!-- #region {"colab_type": "text", "id": "a4UEPAe0KXs7"} -->
# Data Loading

For the purposes of this notebook, we will be labeling 4 cell types in a dataset of purified peripheral blood mononuclear cells from 10x Genomics:
+ CD4 Regulatory T cells
+ CD4 Naive T cells
+ CD4 Memory T cells
+ CD8 Naive T cells




<!-- #endregion -->

```python colab_type="code" id="EfAF_WN0_HES" colab={"base_uri": "https://localhost:8080/", "height": 173} outputId="d57a8d15-fbab-48b4-a0a3-bd3404948a76" tags=[]
adata = scvi.data.purified_pbmc_dataset(subset_datasets=["regulatory_t", "naive_t", "memory_t", "naive_cytotoxic"])
```

<!-- #region {"colab_type": "text", "id": "L-GTv1AfOzf6"} -->
From now on, we assume that cell type information for each cell is unavailable to us, and we seek to retrieve it.

# Automatic annotation using seed labels

In this section we hand curate and select cells which will serve as our ground truth labels.

We start by putting together a list of curated marker genes, from which we aim at identifying our ground truth cell types. These are extracted from the scANVI publication.



<!-- #endregion -->

```python colab_type="code" id="G58qhkFo1lhd" colab={} tags=[]
gene_subset = ["CD4", "FOXP3", "TNFRSF18", "IL2RA", "CTLA4", "CD44", "TCF7", "CD8B", "CCR7", "CD69", "PTPRC", "S100A4"]
```

<!-- #region {"colab_type": "text", "id": "3J5K4hL2AgZJ"} -->
We assign a score to every cell as a function of its cell type signature. In order to compute these scores, we need to normalize the data.
<!-- #endregion -->

```python colab_type="code" id="5h8r8lA0Afe9" colab={} tags=[]
normalized = adata.copy()
sc.pp.normalize_total(normalized, target_sum = 1e4) 
sc.pp.log1p(normalized)

normalized = normalized[:,gene_subset].copy()
sc.pp.scale(normalized)
```

<!-- #region {"colab_type": "text", "id": "XmSpKjyrBIZ_"} -->
Two helper functions 
1. `get_score` assigns a number to cells on the basis of expression levels from a list of `positive` and `negative` markers, and 
2. `get_cell_mask` identifies a subset of cells for which there is highest confidence in annotation as indicated by these scores.
<!-- #endregion -->

```python colab_type="code" id="l2h0D-NE1qKv" colab={} tags=[]
def get_score(normalized_adata, gene_set):
    """Returns the score per cell given a dictionary of + and - genes

    Parameters
    ----------
    normalized_adata
      anndata dataset that has been log normalized and scaled to mean 0, std 1
    gene_set
      a dictionary with two keys: 'positive' and 'negative'
      each key should contain a list of genes
      for each gene in gene_set['positive'], its expression will be added to the score
      for each gene in gene_set['negative'], its expression will be subtracted from its score
    
    Returns
    -------
    array of length of n_cells containing the score per cell
    """
    score = np.zeros(normalized_adata.n_obs)
    for gene in gene_set['positive']:
        expression = np.array(normalized_adata[:, gene].X)
        score += expression.flatten()
    for gene in gene_set['negative']:
        expression = np.array(normalized_adata[:, gene].X)
        score -= expression.flatten()
    return score
```

```python colab_type="code" id="l2h0D-NE1qKv" colab={} tags=[]
def get_cell_mask(normalized_adata, gene_set):
    """Calculates the score per cell for a list of genes, then returns a mask for
    the cells with the highest 50 scores. 

    Parameters
    ----------
    normalized_adata
      anndata dataset that has been log normalized and scaled to mean 0, std 1
    gene_set
      a dictionary with two keys: 'positive' and 'negative'
      each key should contain a list of genes
      for each gene in gene_set['positive'], its expression will be added to the score
      for each gene in gene_set['negative'], its expression will be subtracted from its score
    
    Returns
    -------
    Mask for the cells with the top 50 scores over the entire dataset
    """
    score = get_score(normalized_adata, gene_set)
    cell_idx = score.argsort()[-50:]
    mask = np.zeros(normalized_adata.n_obs)
    mask[cell_idx] = 1
    return mask.astype(bool)
```

<!-- #region {"colab_type": "text", "id": "5r7Z4bMvBLlv"} -->
We run those function to identify highly confident cells, that we aim at using as seed labels
<!-- #endregion -->

```python colab_type="code" id="8_24bN2A1rwi" colab={} tags=[]
#hand curated list of genes for identifying ground truth

cd4_reg_geneset = {"positive":["TNFRSF18", "CTLA4", "FOXP3", "IL2RA"],
                   "negative":["S100A4" ,"PTPRC" ,"CD8B"]}

cd8_naive_geneset = {"positive":["CD8B", "CCR7"],
                   "negative":["CD4"]}

cd4_naive_geneset = {"positive":["CCR7","CD4"],
                   "negative":["S100A4", "PTPRC", "FOXP3", "IL2RA", "CD69" ]}

cd4_mem_geneset = {"positive":["S100A4"],
                   "negative":["IL2RA" ,"FOXP3","TNFRSF18", "CCR7"]}

```

```python colab_type="code" id="BG21NDeZDBvO" colab={} tags=[]
cd4_reg_mask = get_cell_mask(normalized, cd4_reg_geneset,) 
cd8_naive_mask = get_cell_mask(normalized, cd8_naive_geneset,) 
cd4_naive_mask = get_cell_mask(normalized, cd4_naive_geneset,)
cd4_mem_mask = get_cell_mask(normalized, cd4_mem_geneset,)
```

```python colab_type="code" id="GMTYYLpaTVRK" colab={} tags=[]
seed_labels = np.array(cd4_mem_mask.shape[0] * ["Unknown"])
seed_labels[cd8_naive_mask] = "CD8 Naive T cell"
seed_labels[cd4_naive_mask] = "CD4 Naive T cell"
seed_labels[cd4_mem_mask] = "CD4 Memory T cell"
seed_labels[cd4_reg_mask] = "CD4 Regulatory T cell"

adata.obs["seed_labels"] = seed_labels
```

<!-- #region {"colab_type": "text", "id": "oCVDzbf7Vc3h"} -->
We can observe what seed label information we have now
<!-- #endregion -->

```python colab_type="code" id="0dya8rCRVcV6" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="4e07f980-b35a-40ab-efff-ec449cf85a4b" tags=[]
adata.obs.seed_labels.value_counts()
```

<!-- #region {"colab_type": "text", "id": "fYpjtvIGVqbJ"} -->
As expected, we use 50 cells for each cell type!
<!-- #endregion -->

<!-- #region {"colab_type": "text", "id": "WNqvEYGyBWDQ"} -->
# Transfer of annotation
<!-- #endregion -->

<!-- #region {"colab_type": "text", "id": "5EjPRWoZU8_e"} -->
As in the harmonization notebook, we need to register the AnnData object for use in scANVI. Namely, we can ignore the batch parameter because those cells don't have much batch effect to begin with. However, we will give the seed labels for scANVI to use.
<!-- #endregion -->

```python colab_type="code" id="EJ39viHUVK9q" colab={"base_uri": "https://localhost:8080/", "height": 156} outputId="c3aea987-fd17-4a08-847f-5a803e2660d4" tags=[]
scvi.data.setup_anndata(adata, batch_key=None, labels_key="seed_labels")
```

<!-- #region {"colab_type": "text", "id": "2o5MT9NTV7Nh"} -->
Now we can train scANVI and transfer the labels!
<!-- #endregion -->

```python colab_type="code" id="8Nj8QCs6V48K" colab={} tags=[]
lvae = scvi.model.SCANVI(adata, "Unknown", n_latent=30, n_layers=2)
```

```python colab_type="code" id="x497ZM0qWEtX" colab={"base_uri": "https://localhost:8080/", "height": 329, "referenced_widgets": ["5f2c8aeaed4f4cc0a2242e7e9e8e9c9b", "a2afe700b7d34f5989606ac4442ff9ea", "4b735e4c28c44f92b31507ad39927a92", "5b44212701e548e3abc0423536555e52"]} outputId="1cfdbf7a-3a33-46af-b984-71d291dd8177" tags=[]
lvae.train()
```

<!-- #region {"colab_type": "text", "id": "6JlPoPJsWKnJ"} -->
Now we can predict the missing cell types and represent them in the latent space.

<!-- #endregion -->

```python colab_type="code" id="heivhsePWMi8" colab={}
adata.obs["C_scANVI"] = lvae.predict(adata)
adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)
```

<!-- #region {"colab_type": "text", "id": "qGSIv792WL9x"} -->
Again, we may visualize the latent space as well as the inferred labels
<!-- #endregion -->

```python colab_type="code" id="-u1jn1VEWRyp" colab={}
sc.pp.neighbors(adata, use_rep="X_scANVI")
sc.tl.umap(adata)
```

```python colab_type="code" id="X_0IvSBTWacB" colab={"base_uri": "https://localhost:8080/", "height": 318} outputId="39aad6a2-ca5f-44fb-d2f5-3c80905879ae"
sc.pl.umap(adata, color=['labels', 'C_scANVI'])
```

<!-- #region {"colab_type": "text", "id": "bQlC6BcbWUah"} -->
From this, we can see that it is relatively easy for scANVI to separate the CD4 T cells from the CD8 T cells (in latent space and for classification). The regulatory CD4 T cells are sometimes misclassified as CD4 Naive, but this is rare. Better results may be obtained by careful hyperparameter selection for the classifier. This is in the [scvi documentation](https://scvi.readthedocs.io/en/stable/). 
<!-- #endregion -->
