In this README file we explain how to run our DeepER model. We divide this in three sections that we explain in general terms. More details will be provided inside different folders with additional readme files. Environment libraries, analysis, current approach, as explained in our paper, and previous approaches.


##########		ENVIRONMENT 		##########
- The libraries installed for our framework were:
- You can also installed them all at once from our DeepERenv.yml through the creation of an environment from this file.
- Then the variable PYTHONPATH should be set to "./emergency_model_runing/emergency_model"



##########		ANALYSIS 		##########
We describe some previous analysis done in the dataset:
- PreAnalysis.ipynb:Incident type and subtype frequency
- Disribution per NA close date.ipynb: histograms on response time by Incident Type: Utility, Law, Structural and Fire. We also used this to define the outlier limit. Graphically show missing values in the time series
- Replace with Distribution.ipynb: base code to implement preprocessingPeriods.py
- Some of the plots are left in the subfolders of this folder


##########		CURRENT APPROACH ##########
Our current approach consist on a module for preprocessing, a training script and a testing script:
1. preprocessing/preprocessingPeriods.py implements all the preprocessing and more details on the parameters and modules can be found in a readme file inside the preprocessing folder. As the replacement step could be done randomly, it saves the dataset with an specific name format in order to just create it once.

2. model/pytorchRNN_unguidedPeriods.py implements the training and starts by receiving the preprocessing dataset from preprocessingPeriods.getData(). Then it defines the model in model/modePytorch.py and writes down the log needed for experiment comparisons such as best epoch (evaluated in the validation set).

3. model/pytorchRNN_unguided_testBestPeriods.py takes the best epoch from the logs left by the training and runs the model on the test set to report the final evaluation. To be able to read the exact same preprocessed dataset, it uses similar hyperparameters and names as the training script.


On the other hand, we run the baselines in the following way:

1. model/arima_response.py
2. model/linear_response.py


##########		PREVIOUS APPROACH ##########
Before we stay with the replacement by distribution we tried several approaches and preprocessing options.
1. Drop missing values and replace outliers with a constnat value (quantile 90 of each Incident type)
2. Replacce missing and outlier values with a random distribution
3. Guided approach:
	python model/pytorchRNN_guided.py --epochs 50000 --hidden_layer 5 --type_training AD --mini_batch 200 --learning_rate 0.001 --percentage 0.5 --type_rnn lstm --data_path ./Data/With_Features/fire.csv 
4. We also tried attention mechanism but it seemed not to improve results
