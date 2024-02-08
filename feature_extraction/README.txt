This is the main folder for extracting early-life features for paper
"Predicting Battery Lifetime Under Varying Usage Conditions from Early Aging Data"

To properly perform the feature extraction, steps are as
1. Download and unzip the "ISU-ILCC battery aging dataset" from https://doi.org/10.25380/iastate.22582234.
2. Run the code "feature_extraction.py" (note that the directories for raw/preprocessed data need to be change accordingly to the directory of downloaded data). This step generates the raw data files for features and lifetime based on the proper subset partitions, namely training set ('training.csv'), high-DoD test set ('test_in.csv'), and low-DoD test set ('test_out.csv'). 
3. Run the code "feature_selection_based_on_CV.py" to replicate the stepwise forward feature selection results in the paper.