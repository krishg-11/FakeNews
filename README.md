# FakeNews
A Novel Approach of Using Machine Learning to Detect Textual Fake News Using Deep Recurrent Neural Networks. 

Using a Bidirectional Recurrent Neural Network with LSTM Gates, our model is able to achieve 92.4% accuracy in predicting fake news (F1-score = 0.92)

# Data
Raw and formatted datasets are available in folders. All these datasets have been compiled into data.json. The models below use data.json for training and test data.

# Models
Four models were used: Support Vector Machines (SVMs), Vanilla Recurrent Neural Networks (RNNs), Recurrent Neural Networks with LSTM gates (LSTMs), and Bidirectional Recurrent Neural Networks (BRNNs).

A seperate file is available to train and test each of these models (SVM.py, RNN.py, LSTM.py, and BRNN.py, respectively).

# Grid Search
Grid Search was used for hyperparameter tuning. 

GridSearch.py was used for tuning the different types of Recurrent Neural Networks. Results of the grid search is available in hyperparameterData.csv.

GridSearchSVM.py was used for tuning the SVM. Results of the grid search is available in SVMhyperparameterData.csv.
