# GMCHIIR

Scripts for the statistical analysis of the molecular gas around HII regions in nearby galaxies.

**Extract_info_plot_per_gal.py**
Algorithm to extract the needed information from the GMC and HII region catalogs.
It pairs GMCs with HII regions. It creates databases to later produce correlation plots.
COrrelation plots galaxy by galaxy plus histograms of all properties.

**get_limits.py**
Measure the limits for all variables in order to create correlation plots with the
same scale for all galaxies.

**plot_corr_point_all_gals.py**
Plot the correlations between the properties of the GMCs and the HII regions for all galaxies together.
Scatter plot, binned data, linear regression to measure the correlation.
Also, plot GMC properties: Mco vs Sigma_mol; Sigma_mol vs sigma_v/R^0.5


