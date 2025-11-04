## Overview
Build models that predict pasture biomass from images, ground-truth measurements, and publicly available datasets. Farmers will use these models to determine when and how to graze their livestock.

## Start

2 days ago
Close
3 months to go
Merger & Entry
Description
Farmers often walk into a paddock and ask one question: “Is there enough grass here for the herd?” It sounds simple, but the answer is anything but. Pasture biomass - the amount of feed available - shapes when animals can graze, when fields need a break, and how to keep pastures productive season after season.



Estimate incorrectly, and the land suffers; feed goes to waste, and animals struggle. Get it right and everyone wins: better animal welfare, more consistent production, and healthier soils.

Current methods make this assessment more challenging than it could be. The old-school “clip and weigh” method is accurate but slow and impossible at scale. Plate meters and capacitance meters can provide quicker readings, but are unreliable in variable conditions. Remote sensing enables broad-scale monitoring, but it still requires manual validation and can’t separate biomass by species.

This competition challenges you to bring greener solutions to the field: build a model that predicts pasture biomass from images, ground-truth measures, and publicly available datasets. You’ll work with a professionally annotated dataset covering Australian pastures across different seasons, regions, and species mixes, along with NDVI values to enhance your models.

If you succeed, you won’t just improve estimation methods. You’ll help farmers make smarter grazing choices, enable researchers to track pasture health more accurately, and drive the agriculture industry toward more sustainable and productive systems.

## Evaluation
Scoring
The model performance is evaluated using a weighted average of 
 scores across the five output dimensions. The final score is calculated as:


Where:

The term 
 represents the coefficient of determination for dimension 
The weights 
 used are as follows:
Dry_Green_g: 0.1
Dry_Dead_g: 0.1
Dry_Clover_g: 0.1
GDM_g: 0.2
Dry_Total_g: 0.5
R² Calculation
For each target, the coefficient of determination 
 is:


Residual Sum of Squares 

Measures the total error of the model’s predictions:

Total Sum of Squares 

Measures the total variance in the data:

## Terms

 ground-truth value for data point 
: model prediction for data point 
: mean of all ground-truth values
Submission File
Submit a CSV in long format with exactly two columns:

sample_id : ID constructed from image ID and target_name pair.
target: Your predicted biomass value (grams) for that sample_id (float).
The valid target names are: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g.

Your file must contain one row per (image, target) pair, i.e., 5 rows for each image in the test set.

Header and example:

sample_id,target
ID1001187975__Dry_Green_g,0.0
ID1001187975__Dry_Dead_g,0.0
ID1001187975__Dry_Clover_g,0.0
ID1001187975__GDM_g,0.0
ID1001187975__Dry_Total_g,0.0
ID1001187976__Dry_Green_g,0.0
ID1001187976__Dry_Dead_g,0.0
ID1001187976__Dry_Clover_g,0.0
ID1001187976__GDM_g,0.0
ID1001187976__Dry_Total_g,0.0



## Dataset Description
Competition Overview
In this competition, your task is to use pasture images to predict five key biomass components critical for grazing and feed management:

Dry green vegetation (excluding clover)
Dry dead material
Dry clover biomass
Green dry matter (GDM)
Total dry biomass
Accurately predicting these quantities will help farmers and researchers monitor pasture growth, optimize feed availability, and improve the sustainability of livestock systems.

## Files
test.csv

sample_id — Unique identifier for each prediction row (one row per image–target pair).
image_path — Relative path to the image (e.g., test/ID1001187975.jpg).
target_name — Name of the biomass component to predict for this row. One of: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g.
The test set contains over 800 images.

train/

Directory containing training images (JPEG), referenced by image_path.
test/

Directory reserved for test images (hidden at scoring time); paths in test.csv point here.
train.csv

sample_id — Unique identifier for each training sample (image).
image_path — Relative path to the training image (e.g., images/ID1098771283.jpg).
Sampling_Date — Date of sample collection.
State — Australian state where sample was collected.
Species — Pasture species present, ordered by biomass (underscore-separated).
Pre_GSHH_NDVI — Normalized Difference Vegetation Index (GreenSeeker) reading.
Height_Ave_cm — Average pasture height measured by falling plate (cm).
target_name — Biomass component name for this row (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, or Dry_Total_g).
target — Ground-truth biomass value (grams) corresponding to target_name for this image.
sample_submission.csv

sample_id — Copy from test.csv; one row per requested (image, target_name) pair.
target — Your predicted biomass value (grams) for that sample_id.
What you must predict
For each sample_id in test.csv, output a single numeric target value in sample_submission.csv. Each row corresponds to one (image_path, target_name) pair; you must provide the predicted biomass (grams) for that component. The actual test images are made available to your notebook at scoring time.