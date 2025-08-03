# Bayesian AB test package

## About

Bayesian posterior calculation, visualization and analysis for AB test. Includes only data distribution for which a conjugated prior is available.
Allows to build composite posterior from multiple models, which is useful for complex AB test scenarios.
The final posterior as well as the cumulative posterior (along a  number of "experiments" or dates) can be visualized in 1D and 2D.
Some basic Bayesian statistics are calculated, such as credible intervals, ROPE, Bayes factor, etc, and can be used for thresholding and decision making.

Currently available data type:
- Poisson data with gamma prior and posterior
- Bernoulli data with beta prior and posterior
- Normal data with both mean and standard deviation unknown with normal gamma prior and posterior
- Multinomial data with Dirichlet prior and posterior

## Foreword

This code was initially developed for teaching purposes and is not optimized for production use. I wanted to learn some advanced Python features such as mixins class and play with pyplot. However, the code is functional and can be used for Bayesian AB testing, it provides a good starting point for understanding Bayesian statistics in the context of AB testing and even allows for complex scenarios with composite models. It is a good middle ground between online calculators and full-fledged libraries like PyMC. 

## Installation

## Usage

To use the package, you can import the necessary classes and functions from `PyBayesAB`. Below is an example of how to create a model for Poisson data:

```python

from PyBayesAB import poisson
import pandas as pd

# Generate some Poisson data as pandas dataframe
data = pd.DataFrame({
    'group': ['A', 'B', 'A', 'B', 'B'],
    'values': [5, 7, 6, 8, 14],
    'experiment': [1, 1, 2, 2, 1]
})  

# create a model for Poisson data   
poisson_model = poisson.BaysPoisson()
# add the data
poisson_model.add_test_result(data)
# print some Bayesian metrics
print(poisson_model.summary_result(rope_interval=[-5,5], level=95))
```
 This will output the following Bayesian metrics summary:
```        
Bayesian metrics summary: 

Probablity that A is better than B = 2.33% 

There is 95% that the difference in Poisson mean is between -7.01 and 0.78 

The MAP (maximum a posterior estimate) if -3.50 

Probability that the difference is within the ROPE (region of practical equivalence) is 80.9% 

ROPE-based decision: Inconclusive: needs more data (overlaps with ROPE)  

Bayes factor (A vs B vs null): 

                For the null hypothesis: Parameter between -5.00 and 5.00
                For the alternative hypothesis: Parameter larger than 0 or smaller than 0
                The Bayes factor is 5.10, thus providing moderate evidence for the alternative
```

You can also build composite (independant) models. 
For example, if you have a Bernoulli model and a Normal model, you can create a composite model Bernoulli * Normal, which could model  a scenario where you have users that can by a product (Bernoulli) and the amount of money they spend (Normal). 

```python

from PyBayesAB import bernoulli, normal

# generate some Bernoulli data as pandas dataframe
bernoulli_data = pd.DataFrame({
    'group': ['A', 'B', 'A', 'B', 'B'],
    'values': [1, 0, 1, 0, 1],
    'experiment': [1, 1, 2, 2, 1]
})
# generate some Normal data as pandas dataframe
normal_data = pd.DataFrame({
    'group': ['A', 'B', 'A', 'B', 'B'],
    'values': [5.0, 7.0, 6.0, 8.0, 14.0],
    'experiment': [1, 1, 2, 2, 1]
})

# build seperate models
bernoulli_model = bernoulli.BaysBernoulli()
normal_model = normal.BaysNormal()
# add the data
bernoulli_model.add_test_result(bernoulli_data)
normal_model.add_test_result(normal_data)

# create a composite model
composite_model = bernoulli_model * normal_model

# plot the cumulative posterior for the composite model
fig = composite_model.plot_cum_posterior()

# get Bayesian statistics for the composite model
stats = composite_model.summary_result()

```

Important remarks: when using composite models, the posterior is calculated as a product of the posteriors of the individual models. This means that the models should be independent and not correlated. In the example above the models assumes that the purchase probability is independent of the amount of money spent.

## next steps
- add truncated normal distribution (example: for revenue models)
- add pareto distribution
- add decision making based on Bayes factor