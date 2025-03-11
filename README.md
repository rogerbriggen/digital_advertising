# Digital Advertising

## Link-Sammlung
- [Analyzing Marketing Performance: Paid Search Campaign](https://medium.com/@farizalfitraaa/analyzing-marketing-performance-paid-search-campaign-6a9ed5f71c7f) ([Dataset](https://www.kaggle.com/datasets/marceaxl82/shopping-mall-paid-search-campaign-dataset))
- [PPC Campaign Performance Data](https://www.kaggle.com/datasets/aashwinkumar/ppc-campaign-performance-data)

## Environment

The conda environment is named torchrl_ads.

### The environment can be created with conda

````shell
conda env create -f environment.yml
conda env create -f environment_cuda.yml # for the cuda version

````

### The environment can be updated with conda

````shell
conda env update --file environment.yml --prune
````

### To save the updated environment

````shell
conda env export > environment.yml
````

### Install a specific version

````shell
pip install --force-reinstall -v "numpy==1.26.4"
````