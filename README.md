# WasteFootprintTestCases

## The readme is not up to date! 
## look at MacroStudy-Markets/results for the new figures 


This script uses Brightway2 to run LCIA calculations, pandas for data processing and plotly for data visualisation

Calculation results are stored in /tmp

Figures are saved in /figures as .html (the best) and .svg (needs tinkering to make pretty)
Tables are stored in /tables as .csv and .xlsx

## Requirements

bw2calc
bw2analyser

pandas
plotly

## Contents

### LCIAcalculations.py 
* This script will perform LCIA calculations for a given set of 
    activities and methods defined as arguments to LCIAcalculations()
* Produces pickle files (in cwd/tmp) (and optionally dataframes) of the LCIA results (lcia_results) 
    and 'deep copies' of the full LCIA objects (lca_copies)
* In lcia_results, each cell is a list containing the following information: 
    [lca.score, lca.top_activities(), lca.to_dataframe()]

### LCIAprocessing.py 
* For processessing the .pickle files from LCIAcalculations 
* Returns pandas dataframes [df_scores, df_topacts, df_2df] which have the output of lca.values(), lca.top_activities() and lca.to_dataframe()

### LCIAvisualisation.py
Produces figures and tables from the data produced by LCIAprocessing.py
Saves files in "/figures" and "/tables". 
Figures can also appear on your browser automatically (if you uncomment 'fig.show()' in each function)
You may need to install plotly and plotly express 

### LCIAtables.py
* Makes some html tables with plotly
* NEEDS FORMATTING IMPROVEMENT

### LCIAsankeys.py
* Separate function to LCIAprocessing because the calculations are much heavier (~50 times, because of using lca.lci(factorize=True)]
* Then using bw2calc.GraphTraversal to produce dictionarys of critical impact pathways (set options for cutoff and lca calculations limits)
* NOT YET COMPLETE

### Defaults

DEFAULT ARGUMENTS:
    in LCIAcalculations(
                        activities= #see below example
                        method_keyword="Waste Footprint"
                        project="WasteFootprint", 
                        db="cutoff38")
RETURNS: lca_results_df, lca_results_path (deep copies are optional (~4GB), uncomment where needed to get that) 

DEFAULT ACTIVITY LIST: (use FindActivities() in main.py to get a different list)
activities = [["'battery production, NiMH, rechargeable, prismatic' (kilogram, GLO, None)", 'e5af5c3b833867a63f4d4c4312a0c3bd'], 
              ["'battery production, Li-ion, rechargeable, prismatic' (kilogram, GLO, None)", '3377a1ab4d9266e104f29c12b4443f31'], 
              ["'battery production, lead acid, rechargeable, stationary' (kilogram, RoW, None)", 'bca50086a52be115e70bb28f0bc0659a'], 
              ["'battery production, Li-ion, NMC111, rechargeable, prismatic' (kilogram, RoW, None)", '07ccf3e73d2fcbb25a2fc41972aba5fd'], 
              ["'battery production, NMC811, Li-ion, rechargeable, prismatic' (kilogram, RoW, None)", 'd910ceabac665cedc3f166a7964f6965'], 
              ["'battery production, NCA, Li-ion, rechargeable, prismatic' (kilogram, RoW, None)", 'c56e4eec177b9334aa4e31a350daadd4']]
