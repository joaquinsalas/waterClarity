# Water Clarity Assessment through Satellite Imagery and Machine Learning

Leveraging satellite monitoring and machine learning techniques for water clarity assessment addresses the critical need for sustainable water management.
This study uses satellite images and machine learning techniques to assess water clarity by predicting the Secchi disk depth (SDD). During data preparation, AquaSat samples were updated with Landsat 8 satellite collections that included atmospheric corrections, resulting in 33,261 multispectral and depth observations. For inferring the SSD, regressors such as SVR, NN, and XGB were trained, and an ensemble method  was used to enhance performance. The ensemble demonstrated performance with an average determination coefficient of \(R^2\) of 0.623 and a standard deviation of 0.024. Meanwhile  field data validation achieved an \(R^2\) of 0.51. The results are encouraging as they resolve depth with error values less than 0.35 m. 
This document presents the transition from semi-analytical to data-driven methods in water clarity research, using a machine learning ensemble  to  assess the clarity of a  water bodies through satellite imagery.


## Input data
We start with AquaSat, a dataset with about 600,000 paired observations of water quality running from 1984-2019. Aquasat can be downloaded from https://figshare.com/articles/dataset/AquaSat/8139383

Since 




## Ensemble

Our program fine-tunes an ensemble model by estimating the optimal architecture to aggregate individual regressors. Utilizing the script `ensemble_ft.py`, we integrate the following models: a neural network (`../models/NN_model_20240227.h5`), XGBoost (`../models/best_xgb_model_20240227.pkl`), and support vector regression (`../models/best_svr_model.pkl`). The dataset used for this process is located at `../data/20230930_aquasat_L2_C2.csv`. Normalization parameters for the dataset, obtained from the training split, are stored in `../data/normalization_scaler_20240227.pkl`. Our approach involves a neural network-based architecture for aggregating the regressors, where hyperparameters are meticulously fine-tuned to achieve the best performance. The resulting optimized model is saved at `../models/best_ens_model_20240227.h5`. Before determining the best hyperparameters for the ensemble neural network, predictions from the individual regressors are normalized using parameters stored in `../data/ensemble_normalization_scaler.pkl`.


The program 'ensemble.py' uses these files to train the ensemble 20 times, on different data partitions. The performance is evaluated on the testing partition.



AquaSat: A Data Set to Enable Remote Sensing of Water Quality for Inland Waters



@article{ross2019aquasat,
  title={AquaSat: A data set to enable remote sensing of water quality for inland waters},
  author={Ross, Matthew RV and Topp, Simon N and Appling, Alison P and Yang, Xiao and Kuhn, Catherine and Butman, David and Simard, Marc and Pavelsky, Tamlin M},
  journal={Water Resources Research},
  volume={55},
  number={11},
  pages={10012--10025},
  year={2019},
  publisher={Wiley Online Library}
}


### BibTeX

@article{salas2024water, <br>
   author = {Joaquín Salas, Rodrigo Sepúlveda, and~Pablo Vera}, <br>
   title = {Water Clarity Assessment through Satellite Imagery and Machine Learning}, <br>
   journal = {IEEE LATAM Transactions},  <br>
   year = {2024}<br>
} 



