# Jacobian calculation method
In the folder 'jac', we provide the notebooks for:
- Jacobian calculation and visualisation (notebook 1-3)
- Calculate, compare, and plot the difference in contact prediction accuracy for different methods (notebook 3-4)
- Calculate and plot pairwise potentials with different cutoffs (notebook 5-6) 
# Contact recovery experiments 
In the folder 'contact_recovery', we provide the notebooks for:
- Prepare the secondary structure information needed for secondary structure element (SSE) pair contact recovery (notebook 0) 
- Single segment contact recovery with different masking types (notebook 01-04). Note that notebook 01 randomly samples some segments, and then the same segments are used for different masking methods for comparison.
- Analysis and visualisation of contact recovery results for single segment contact recovery (notebook 05 -06)
- SSE pair segment contact recovery with different masking types (notebook 07-10). Note that notebook 07 randomly samples some pairs, and then the same pairs are used for different masking methods for comparison.
- Analysis and visualisation of contact recovery results for SSE pairs contact recovery (notebook 11 -12)
# Data 
Since the h5 files from Jacobian calculation and contact recovery experiments are too large, we did not include them in this repo. However, we provide the information needed to reproduce them:
- Sequences of proteins from Gremlin (full_seq_dict.json) 
- Proteins we selected after filtering based on criteria mentioned in our paper (selected_protein.json)
- Secondary structure element information for the proteins we run our analysis (ss_dict.json)
- The exact single segments and SSE pairs we used for our analysis (json files starting with the word 'reproduce')
- You can also run the analysis on proteins not included here with the code we provided in previous sections
