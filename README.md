# 3D printer error detection

For the competition https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/overview

## Architecture plan:

Edge detection feature extraction using LDC:
```
@ARTICLE{xsoria2022ldc,
  author={Soria, Xavier and Pomboza-Junez, Gonzalo and Sappa, Angel Domingo},
  journal={IEEE Access}, 
  title={LDC: Lightweight Dense CNN for Edge Detection}, 
  year={2022},
  volume={10},
  number={},
  pages={68281-68290},
  doi={10.1109/ACCESS.2022.3186344}}
```
Fine-tuned various pretrained models on the dataset, resnet50 worked best. 
