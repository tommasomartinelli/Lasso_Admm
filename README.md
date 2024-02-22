# Lasso Regression and ADMM Implementation

## Overview
This repository contains an implementation of Lasso Regression along with an exploration of the Alternating Direction Method of Multipliers (ADMM) algorithm. The study investigates different optimization approaches, including a distributed version of ADMM.
The goal is to provide a comprehensive understanding of the Lasso Regression model and assess its performance under various scenarios. The project includes implementations, experiments, and results on synthetic and real-world datasets.

## Project composition
This project is organized into the following sections:

- **Lasso:**
  - The `lasso` folder contains Jupyter Notebooks implementing Lasso Regression.
  - Explore the code and follow instructions in the notebooks for running Lasso Regression.

- **Data:**
  - The `data` folder encompasses various aspects related to datasets used in the project.
    - **Concrete Compressive Strength Dataset:**
      - The original dataset can be accessed [here](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
      - A Jupyter Notebook (`Preprocessing.ipynb`) is available for exploratory data analysis (EDA) and preprocessing steps applied to the Concrete Compressive Strength Dataset.
      - Processed data resulting from the EDA and preprocessing steps are stored here.
    - **Synthetic Datasets:**
      - Two additional synthetic datasets are dynamically generated within the main Jupyter Notebook during the experimentation process.
  - Refer to the documentation for comprehensive details about the sources, characteristics, and preprocessing techniques applied to the datasets.

- **Doc:**
  - The `doc` folder holds comprehensive documentation in PDF format.
  - Find theoretical concepts, algorithm details, and project-specific information.

- **Img:**
  - The `img` folder stores images used in documentation or visualizations.

## Experiments and Results

In this section, we detail the experiments conducted on three distinct versions of Lasso Regression. These implementations include Gradient Descent (GD), Alternating Direction Method of Multipliers (ADMM), and Distributed ADMM splitted by examples. The models were tested on synthetic datasets generated within the notebook and a real-world dataset.

### Lasso Regression Versions

#### Gradient Descent (GD)
The GD model is a conventional approach for Lasso Regression. It was tested with varying step sizes and L1 penalties, exploring their impact on convergence, execution times, the number of iterations and their influence on performance metrics.

#### Alternating Direction Method of Multipliers (ADMM)
ADMM is an optimization technique that solves the Lasso problem efficiently. It was implemented with different parameter configurations to observe their influence on convergence, execution times, the number of iterations and their influence on performance metrics.

#### Distributed ADMM
The distributed version of ADMM involves parallelizing the optimization process among multiple agents. This model was tested with varying numbers of agents to assess its performance in a distributed setting.

### Experimental Setup

#### Synthetic Datasets
Two synthetic datasets were generated within the main Jupyter Notebook to simulate different scenarios for testing the Lasso Regression models.

#### Concrete Compressive Strength Dataset
A real-world dataset, the Concrete Compressive Strength Dataset, was used to evaluate the models in a practical context. The dataset was preprocessed using a dedicated Jupyter Notebook (`Preprocessing.ipynb`).

### Results

The models were rigorously tested on datasets with diverse characteristics. The results indicate that, in terms of metrics, all three versions of Lasso Regression showed similar performance. However, there were differences in execution times and the number of iterations required. The choice of parameters played a crucial role and had varying impacts depending on the dataset.

*Note: The presented results are preliminary and serve as initial insights. A more comprehensive evaluation would require an extensive set of model tests.*

## Usage

To utilize the Lasso Regression implementations provided in this repository, follow the steps below:

1. Explore the Jupyter Notebooks in the `lasso` folder to understand the Lasso Regression implementations.

2. Instantiate and utilize the `LassoReg` class with your custom parameters as needed. The class allows you to experiment with different optimization techniques and parameters.

3. Use the provided datasets for testing. The `data` folder contains the original Concrete Compressive Strength Dataset, along with synthetic datasets generated within the main Jupyter Notebook.

4. Refer to the documentation in the `doc` folder for comprehensive details about the model, optimization techniques, and dataset characteristics.

## Custom Testing

Feel free to test the Lasso Regression implementations on your custom datasets. Follow the steps below to integrate your data:

1. Prepare your dataset in a format compatible with the Lasso Regression model. Ensure it has the necessary features and labels.

2. Modify the existing Jupyter Notebooks or create a new one to include your dataset.

3. Instantiate the `LassoReg` class with your custom parameters and use it to fit and predict on your data.

4. Share your experience or findings by contributing to this project.

By testing the implementation on diverse datasets, you can contribute to the understanding and robustness of the Lasso Regression model.



