# Covid-19-Literature-Clustering-v2022
In this project I am solving a hard problem that face the medical scientists which is searching in large amount of literatures which is related to Covid-19 , so in this Project the matter gone easier for them.

This Project Aims to cluster large amount of literatures into number of similar Groups related to each other, and applying the Topic-Model approach to name a Topic for each cluster to make it easier understanding what every cluster hides and talking about, also by Implementing an Interactive Plot using Bokeh , so that it any one can use by clicking on any cluster and gets the information needed.

### Kaggle Link: https://www.kaggle.com/code/remonboshra/covid-19-literature-clustering-2022

### Interactive Plot: https://github.com/Remon128/Covid-19-Literature-Clustering-v2022/blob/main/t-sne_covid-19_interactive.html

## Table of Contents
### 1- Loading the data
### 2- Pre-processing
### 3- Vectorization
### 4- PCA & Clustering
### 5- Dimensionality Reduction with t-SNE
### 6- Topic Modeling on Each Cluster
### 7- Classify
### 8- Plot
### 9- How to Use the Plot?


### Importing libraries
```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
```

### Data Preprocessing

```
df.dropna(inplace=True)
```

### English language detection

#### installing langdetect Module

```
!pip install langdetect
```

### Importing Language detection Libraries

```
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory
```

```
lang = detect(" ".join(text[]))
```

### Download and Install Spacy sparser a NLP Framework


```
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
```
```
pip install -U spacy
```

#### Import Spacy and NLP Libraries
```
#NLP 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg
```

#### Spacy Parser
```
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000
```

### Vectorization

```
from sklearn.feature_extraction.text import TfidfVectorizer
```

### PCA & Clustering

```
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
```

### K-means Algorithm

```
kmeans = KMeans(n_clusters=k, random_state=42)
```
![t-sne_covid19](https://user-images.githubusercontent.com/24530726/166269353-2ff83418-5196-438e-871b-f66bed3b902f.png)


### Dimensionality Reduction for 2D Plotting

```
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=50)
```

### Plotting Clusters Assigned to each Article

![improved_cluster_tsne](https://user-images.githubusercontent.com/24530726/166272217-f472ad64-0b99-445e-8084-2f052bb7c5bd.png)



### Topic Modeling on Each Cluster

```
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
```

### Buidling LDA Model

```
LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10,
                                    learning_method='online',verbose=False, random_state=42)
```


### Installing Bokeh for Interactive Visualization

```
! wget https://raw.githubusercontent.com/MaksimEkin/COVID19-Literature-Clustering/master/lib/plot_text.py
! wget https://raw.githubusercontent.com/MaksimEkin/COVID19-Literature-Clustering/master/lib/call_backs.py
! mv plot_text.py lib/.
! mv call_backs.py lib/.
! ls lib/
```

```![improved_cluster_tsne](https://user-images.githubusercontent.com/24530726/166288918-ee607bb5-945d-4c7f-bc33-56a27c38a559.png)

!pip3 install bokeh==2.4.2
```


#### Checking Bokeh Version
```
import bokeh
print(bokeh.__version__)
```




