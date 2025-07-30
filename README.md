- train.ipynb worked better than titanic_proj.ipynb.

- Categorial features were converted to numerical features by one-hot encoding.
- For missing values, if labels were missing, they were simply removed since they couldn't be used to train the model. If feature values were missing, they were replaced
with the mode of their feature.
- Limiting depth (setting max-depth) is the easiest but not dynamic way of stopping. I used information gain (or gini-purification) limit, that is if the gain is less than 10^-5, and stop branching.
- Bootstrap based on N decision trees, selecting m = (#features)^1/2 features, majority vote.
- I was using all unique values for thresholds at first, but this caused dead kervel. So, I instead limit the number of thresholds to 10 by linspacing from min + 10^-5 to max -10^-5.
