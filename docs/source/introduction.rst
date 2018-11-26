======================
Why did we build this?
======================

When exploring a data set, for example to model one variable in terms of the others,
it is useful to summarize and visualize the dependencies between the variables.

The calculation of correlation coefficients between paired data variables is a standard tool of analysis for every data analyst.
Pearson's correlation coefficient is a de facto standard in most fields, but by construction only works for interval variables
(sometimes called continuous or real-valued variable).
While many correlation coefficients exist, each with different features, we have not been able to find a
correlation coefficient with Pearson-like characteristics 
and a sound statistical interpretation that works for interval, ordinal and categorical variable types alike.

This library implements a novel correlation coefficient, :math:`\phi_{K}`, with properties that - taken together - form an advantage over existing methods.
The correlation coefficient follows a uniform treatment for interval, ordinal and categorical variables.
For details on the methodology behind the calculations, please see our publication.

The ``PhiK`` correlation analyzer library contains useful functions on three related topics typically encountered in data analysis:

* Calculation of the correlation coefficient, :math:`\phi_{K}`, for each variable-pair of interest.
* Evaluation of the statistical significance of each correlation, where particular attention is paid to strong correlations and low-statistics samples.
* Insights in the correlation of each variable-pair, by studying outliers and their significances.
  
This analysis library is particularly useful in modern-day analysis when studying the dependencies between a set of variables with mixed types,
where often some variables are categorical.

The Phi_K correlation analyzer package has been used to study surveys, insurance claims, correlograms, etc.
For the available examples on how to use the methods, please see the `tutorials <tutorials.html>`_ section.
