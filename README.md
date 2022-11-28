# ITU-ML5G-PS-005-KDDI-UT-NakaoLab-AI

# Environment
Please use Google Colab.

Runtime type : GPU

GPU type : Nvidia Tesla K80

# How to evaluate and train
Please download this repository and place it under your google drive directory.

If you want to evaluate the model I've created, please open and run evaluate.ipynb.

If you want to create your own model, please open and run train.ipynb.

# Description of Evaluation class
1. evaluation.score()
   
   This function prints the confusion matrix and calculates precision, recall and F1 score.

2. evaluation.MSE()

   This function calculates MSE(mean squared error).

3. evaluation.visualization()

   This function illustrates the predicted results and the actual number of UE registration failures from the last 20 to 10 cycles of the test data.