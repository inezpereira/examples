# Import statements
import itertools
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils import shuffle


def fitmodel(x, params):
  '''
  Instantiates and fits Bayesian GMM
  Used in the parallel for loop
  '''
  # Gaussian mixture model
  clf = BayesianGaussianMixture(**params)
  # Fit
  clf = clf.fit(x, y=None)
  return clf

def plot_results(X, means, covariances, title):
    
  plt.plot(X, np.random.uniform(low=0, high=1, size=len(X)),'o', alpha=0.1, color='cornflowerblue', label='data points')
  for i, (mean, covar) in enumerate(zip(
    means, covariances)):
    # Get normal PDF
    n_sd = 2.5
    x = np.linspace(mean - n_sd*covar, mean + n_sd*covar, 300)
    x = x.ravel()
    y = stats.norm.pdf(x, mean, covar).ravel()
    if i == 0:
      label = 'Component PDF'
    else:
      label = None
    plt.plot(x, y, color='darkorange', label=label)
  plt.yticks(())
  plt.title(title)

# Generate data
g1 = np.random.uniform(low=-1.5, high=-1, size=(1,100))
g2 = np.random.uniform(low=1.5, high=1, size=(1,100))
X  = np.append(g1, g2)

# Shuffle data
X = shuffle(X)
X = X.reshape(-1, 1)

# Define parameters for grid search
parameters = {
  'n_components': [1, 2, 3, 4, 10],
  'weight_concentration_prior_type':['dirichlet_distribution']
}

# Create permutations of parameter settings
keys, values = zip(*parameters.items())
param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run GridSearch using parallel for loop
list_clf = [None] * len(param_grid)
num_cores = multiprocessing.cpu_count()
list_clf = Parallel(n_jobs=num_cores)(delayed(fitmodel)(X, params) for params in param_grid)

# Print best model (based on lower bound on model evidence)
lower_bounds = [x.lower_bound_ for x in list_clf] # Extract lower bounds on model evidence
idx = int(np.where(lower_bounds == np.max(lower_bounds))[0]) # Find best model

best_estimator = list_clf[idx]
print(f'Parameter setting of best model: {param_grid[idx]}')
print(f'Components weights: {best_estimator.weights_}')

# Plot data points and gaussian components
plt.figure(figsize=(8,6))

ax = plt.subplot(2, 1, 1)
if best_estimator.weight_concentration_prior_type == 'dirichlet_process':
  prior_label = 'Dirichlet process'
elif best_estimator.weight_concentration_prior_type == 'dirichlet_distribution':
  prior_label = 'Dirichlet distribution'

plot_results(X, best_estimator.means_, best_estimator.covariances_,
          f'Best Bayesian GMM | {prior_label} prior')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.legend(fontsize='small')

# Plot histogram of weights
ax = plt.subplot(2, 1, 2)
for k, w in enumerate(best_estimator.weights_):
  plt.bar(k, w,
    width=0.9,
    color='#56B4E9',
    zorder=3,
    align='center',
    edgecolor='black'
    )
  plt.text(k, w + 0.01, "%.1f%%" % (w * 100.),
            horizontalalignment='center')
ax.get_xaxis().set_tick_params(direction='out')
ax.yaxis.grid(True, alpha=0.7)
plt.xticks(range(len(best_estimator.weights_)))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
plt.ylabel('Component weight')
plt.ylim(0, np.max(best_estimator.weights_)+0.25*np.max(best_estimator.weights_))
plt.yticks(())
plt.savefig('bgmm_clustering.png')
plt.show()
plt.close()