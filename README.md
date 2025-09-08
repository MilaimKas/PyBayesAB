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

This code was initially developed for teaching purposes and is not optimized for production use. I wanted to learn some advanced Python features such as mixins class, class operations and play with pyplot. However, the code is functional and can be used for Bayesian AB testing, it provides a good starting point for understanding Bayesian statistics in the context of AB testing and even allows for complex scenarios with composite models. It is a good middle ground between online calculators and full-fledged libraries like PyMC. 

### Remarks on Bayesian Vs Frequentist statistics
Using simple models with un-informative priors, some Bayesian metrics such as the credible interval or the posterior mean can be seen as equivalent to their Frequentist counterpart: confidence interval and sample mean, respectively. 
Since AB testing relies on defining thresholds for decision making, both approaches can be used. However, Bayesian statistics provides a more intuitive interpretation of the results and allows to  incorporate prior knowledge into the analysis (altough some may argue that this is not always a good idea). 

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

n_exp = 20 # number of experiments (or dates)

# create a Bernoulli model
bernoulli_model = bernoulli.BaysBernoulli()

#add some random data to the Bernoulli model
p_A = 0.21
p_B = 0.2
for n in range(n_exp):
    n_trial = np.random.randint(10,50)
    bernoulli_model.add_rand_experiment(n_trial, p_A, group="A")
    bernoulli_model.add_rand_experiment(n_trial, p_B, group="B")

# create a Normal model
normal_model = normal.BaysNorm()

# add some random data to the Normal model
mu_A = 20
std_A = 10
tau_A = 1/std_A**2 
mu_B = 22
std_B = 12
tau_B = 1/std_B**2
for i in range(n_exp):
    n_data = np.random.randint(10,50)
    normal_model.add_rand_experiment(n_data, mu_A, std_A, group="A")
    normal_model.add_rand_experiment(n_data, mu_B, std_B, group="B")

# create a composite model
composite_model = bernoulli_model * normal_model

# get Bayesian statistics for the composite model
stats = composite_model.summary_result()

```
Some plots can be generated to visualize the posterior and cumulative posterior:

```python
fig = composite_model.plot_final_posterior()
```
![](final_post_comp.png)

```python
fig = composite_model.plot_cum_posterior()
```

```python
fig1, fig2 = composite_model.plot_bayesian_metrics(rope_interval=[-2, 2])
```
![](bay_met_fig1_comp.png)
![](bay_met_fig1_comp-1.png)

Important remarks: when using composite models, the posterior is calculated as a product of the posteriors of the individual models. This means that the models should be independent and not correlated. In the example above the models assumes that the purchase probability is independent of the amount of money spent.

## Code structure
The code is structured in a way that allows for easy extension and addition of new models. Each model is implemented as a class that inherits from the `BaysModel` class and the `PlotManager` class. The `BaysModel` class provides the basic functionality for adding data, calculating posterior, and visualizing the results. The `PlotManager` class provides the functionality for plotting the posterior and cumulative posterior. The models are implemented in separate files in the `src/PyBayesAB/distribution` directory where the `template.py` module can be used as basis.

## next steps
- add truncated normal distribution (example: for revenue models)
- add pareto distribution
- add decision making based on Bayes factor
- add possibility to use custom priors from normal distribution