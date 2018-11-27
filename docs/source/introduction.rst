======================
Why did we build this?
======================

When exploring a data set, for example to model one variable in terms of the others,
it is useful to summarize the dependencies between the variables, assess their significances, and
visualize the individual variable dependencies.
The ``PhiK`` correlation analyzer library contains several useful functions to help one do so.

* This library implements a novel correlation coefficient, :math:`\phi_{K}`, with properties that - taken together - form
  an advantage over existing methods.

  The calculation of correlation coefficients between paired data variables is a standard tool of analysis for every data analyst.
  Pearson's correlation coefficient is a de facto standard in most fields, but by construction only works for interval variables
  (sometimes called continuous variables). Pearson is unsuitable for data sets with mixed variable types,
  e.g. where some variables are ordinal or categorical.

  While many correlation coefficients exist, each with different features, we have not been able to find a
  correlation coefficient with Pearson-like characteristics 
  and a sound statistical interpretation that works for interval, ordinal and categorical variable types alike.

  The correlation coefficient :math:`\phi_{K}` follows a uniform treatment for interval, ordinal and categorical variables,
  captures non-linear dependencies, and is similar to Pearson's correlation coefficient in case of a bivariate normal input distribution.

* We found that, by default, popular analysis libraries such ``R`` and ``scipy`` make incorrect ("asymptotic") assumptions when assessing
  the statistical significance of the :math:`\chi^2` contingency test of variable independence. In particular, the actual number of
  degrees of freedom and the shape of the test statistic distribution can differ significantly from their theoretical
  predictions in case of low to medium statistics data samples. This leads to incorrect p-values for the hypothesis test of variable
  independence. A prescription has been implemented to fix these two mistakes.
    
* Visualizing the dependency between variables can be tricky, especially when dealing with (unordered) categorical variables. 
  To help interpret any variable relationship found, we provide a method for the detection of
  significant excesses or deficits of records with respect to the expected values in a contingency table, so-called outliers,
  using a statistically independent evaluation for expected frequency of records, accouncting for the uncertainty on the expectation.
  We evaluate the significance of each outlier frequency in a table, and normalize and visualize these accordingly.
  The resulting plots we find to be very valuable to help interpret variable dependencies,
  and work alike for interval, ordinal and categorical variables.

The ``PhiK`` analysis library is particularly useful in modern-day analysis when studying the dependencies between a set of
variables with mixed types, where often some variables are categorical.
The package has been used by us to study surveys, insurance claims, correlograms, etc.

For details on the methodology behind the calculations, please see our publication.
For the available examples on how to use the methods, please see the `tutorials <tutorials.html>`_ section.
