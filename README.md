# 3D printer error detection

## Architecture plan:

Feature extractions ideas:
    - edge detection methods 
    - semantic segmentation methods (maybe?)

Train different models for each feature extraction and create an Ensemble

Done:

- Making edge detection dataset
- Making model for classification

Todo:

- Create and run model1 on initial dataset
- Create and run model2 on edge outlined dataset
- Create an ensemble of the two models.