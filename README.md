# pysimilarity

Pysimilarity is a library that contains several methods for determining similarities between groups. Currently, it supports calculating similarities using distances, and a classifier.

## Install

```
pip install pysimilarity
```

## Usage

```
from sklearn.datasets import load_iris
from pysimilarity import DistanceSimilarity
import pandas as pd

iris_dataset = load_iris()
iris_df = pd.DataFrame(iris_dataset['data'],columns=iris_dataset['feature_names']
                      )
iris_df['species'] = pd.Series(iris_dataset['target']).replace({0:'setosa', 1:'versicolor', 2:'virginica'})


dist_similarity = DistanceSimilarity()
setosa_data = iris_df.loc[iris_df.species == 'setosa']
setosa_similarity = dist_similarity.fit_transform(setosa_data.drop('species',axis=1)
                                                  ,iris_df.drop('species',axis=1))
print(setosa_similarity) # Array of shape n_rows of iris_df.
```



 





