# Datasets for Causal Structure Learning


## Synthetic datasets
We provide codes for generating synthetic datasets used in the papers. Please see the [example notebook](examples_to_generate_synthetic_datasets.ipynb) for further details.

## Real dataset
We release a very challenging [dataset](https://github.com/zhushy/causal-datasets/tree/master/Real_Dataset) from real telecommunication networks, to find causal structures based on time series data. 

### Data format
- **real_dataset_processed.csv**: each row counts the numbers of occurrences of the alarms (A_i,i=0,1,...,56) in 10 minutes. The rows are arranged in the time order, i.e., first 10 mins., second 10 mins., etc.
- **true_graph.csv**: the underlying causal relationships, according to expert experience.  `(i,j)=1` implies an edge `i->j`.

Notice: we will also release the original dataset so that you may try processing the data in your own way to find the 
underlying causal graph. Currently we are working on a detailed description of the physical meanings and how to process the data.

### Our methods
We are trying some event and time series methods to tackle this problem. We are going to release our results soon, together
with the results from several benchmark methods on this dataset. We welcome everyone to try this dataset and report the result!

### Results

| Methods| Precision | Recall | SHD |
|---|---|---|---|
| PC (with Fisher-z test)   | | |
| ...|
