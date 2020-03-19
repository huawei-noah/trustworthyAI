# Datasets for Causal Structure Learning


## Synthetic datasets
Codes for generating synthetic datasets used in the papers. Please see the [example notebook](examples_to_generate_synthetic_datasets.ipynb) 
for further details.

## Real dataset

We release a very challenging [dataset](https://github.com/zhushy/causal-datasets/tree/master/Real_Dataset) from real telecommunication networks, 
to find causal structures based on time series data. 

### Data format
- **real_dataset_processed.csv**: each row counts the numbers of occurrences of the Alarms (A_i,i=0,1,...57) in 10 minutes. The rows are arranged in the time order,
i.e., first 10 mins., second 10 mins., etc.
- **true_graph.csv**: the underlying causal relationships, according to expert experience.  (i,j)=1 implies an edge i

Notice: we will also release the original dataset, with detailed explainantion. So you may try process the data in your own way to discover the 
underlying causal graph. 

### Our methods
We are trying the event and times series methods to tackle this dataset. We are going to release our results soon, together
with several benchmark methods on this dataset. We welcome everyone to try this dataset and report the result!

### Results

| Methods| Precision | Recall | SHD
| PC (with Fisher-z test)   | | |
| ...|
