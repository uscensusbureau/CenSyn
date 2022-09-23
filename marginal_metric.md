# N-Marginal Evaluation 
- **What**: the difference between two datasets in terms of their clustering characteristics 
- **Why**: measure the ability of a differential privacy algorithm to conserve clustering characteristics
- **How**: computed comparisons of n-column (i.e. n-feature) marginal density distributions between any two similarly structured datasets.


## Definition of terms
- **N-marginal**: set of records corresponding to a n-feature value combination.
- **N-feature combination**: set of n features chosen from the feature space.
- **N-feature value combination**: set of n values, where each value is selected from the set of possible values for one of the features in a n-feature value combination.


## To compare two data sets that share the same shape using the n-marginal metric: 
1. Bin scalar features so as to prevent a large number of possible values resulting in an explosion of feature value combinations - and thus greatly increases runtime.
2. Select n features - for example 3. For each possible 3-feature-value combination corresponding to these 3 features, sum the results of the following computation (results in score between 0 and 2):
    1. Compute density for the 3-feature-value combination, for both data sets (number of observations corresponding to the 3-feature-value combination, divided by the total number of observations). 
    2. Compute absolute difference between the two densities. 
3. Do Step 2 and average scores across either 1) a random sunset or 2) all possible combinations of 3 features to achieve an aggregate score in the range 0-2. 
    - 2 can become infeasible with high-dimensional data.


## How to interpret results - the marginal metric score (value between 0 and 2):
0 - perfectly matching density distributions (for the marginals used in the comparison).
2 - no overlap whatsoever (for the marginals used in the comparison).


## Usage Process:
0. Bin features as deemed best - fewer possible values per features means less computational complexity for marginal metric evaluation.
1. Instantiate a MarginalMetric object with the following parameters:
  - `data_frame_a` (`pandas.DataFrame`): first of two data frames to evaluate against one another.
  - `data_frame_b` (`pandas.DataFrame`): second of two data frames to evaluate against one another.
  - `marginal_dimensionality` (`int`): the number of features used to create marginals. The higher this number, the greater the number of features considered at a time when the clustering characteristics for the two data sets are evaluated against each other. 3 is the default value. Larger values can drastically increase the computational complexity of the marginal metric evaluation. 
  - `picking_strategy` (`Enum`): determines how feature combinations are selected. 
    Possible values are: 
    - `lexicographic` --> all possible combinations, in lexicographic order
    - `rolling` --> incrementally shifting sets. Example: (1,2,3), (2,3,4), ... (8,9,n)
    - `random` --> random selection from all possible combinations 
  - `sample_ratio` (`float`): sample proportion of available combinations to use, given a picking_strategy.  If value of 1 and picking strategy is lexicographic or random, then all possible combinations will be used in the marginal metric evaluation. If value of 1 and picking strategy is rolling, then all rolling combinations will be used. This value can have a significant impact on runtime.
2. Optional parameters include:
  -  `stable_features`: A list of the of the Feature names to include in the marginal metric. Default value is the empty list [].
3. Make a call to MarginalMetric.compute_results(), which returns a Result object containing the results of the evaluation
4. Pass the Result instance to a Report instance (e.g. ConsoleReport), and call the produce_report() method for the Report instance. 

## Suggested parameters for a first run:
- `marginal_dimensionality` --> 3
- `picking_strategy` --> random
- `sample_ratio` --> .001
