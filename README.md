# Stressed Bees

This repository contains the code for the proofs and figures accompanying the manuscript, "Generalized Stressors on Social Bee Colonies", along with a few additional files.
Each file in the repository in the repository is referenced below. 

## Figure Generation 
- `LHS_prcc.py` -> runs LHS and PRCC analysis to produce `lhs_aaof.pdf` and `lhs_pop.pdf` (Figure 2)
- `Social_Inhibition_Heatmap.ipynb` -> compares social inhibition terms from Khoury 2011 and our study to produce `Social_Inhibition_Heatmap.pdf` (Figure 8)
- `bifurcation_plot.py` -> generates bifurcation plot, `bif.pdf` (Figure 3), of stressor contact rate and growth rate. 
- `stressor_comparison.py` -> compares characteristics of stressors to produce `many_stressor_comparison.pdf` (Figure 4)
- `monotonicity.py` -> confirms the monotonicity of response variables for PRCC analysis, generating `monotonicity_aaof.pdf` and `monotonicity_pop.pdf` (Figure 6 and 7)
- `temporal_comparison.py` -> compares the temporal application effects of various stressors to generate `temporal_comparison.pdf` (Figure 5)

## Proofs
- `analytical_work.nb` -> all work done for computations and proofs (divided into sections inside, very helpful for finding Equilibria and Appendix A) 

## Misc.
- `bee_model.py` -> module containing the actual ODE model (used in various scripts) 
- `parameter_est.xlsx` -> simple computations to support parameter range estimations in Appendix B
- `time_series.py` -> can be used for quick time series analysis of the model 
