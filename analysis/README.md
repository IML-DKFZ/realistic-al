## :chart_with_upwards_trend: Analysis
The implemented analysis can be found in the folder `/analysis/` and consists of:
1. Standard performance vs. labeled data plots
2. Area Under Budget Curve (AUBC)
3. Pairwise Penalty Matrices (PPM)

The main script for the analysis is located in:
`/analysis/plot.py`

To execute the analysis on your own device the results path will need to be changed in the script.
Also all values for the AUBC plots are read out in this function.
Further it requires to have all of the models that are trained on the entire dataset to be present in the default version.
For this execute: `/analysis/obtain_metrics.py -l 1 -v {metric} -s auto -p {path_to_full_trained_models}`.

metric = `test/acc` (and `test/w_acc` for MIO-TCD and ISIC-2019)

A newly implemented Query Method needs to be introduced here.

The AUBC values and PPMS can be found in the respective jupyter notebooks.