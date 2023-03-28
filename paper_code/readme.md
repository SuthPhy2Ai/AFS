Part1: Data exploration;   
Part2: Cost-sensitive learning;   
Part3: Hyperparameter tuning training and pre-trained DNN loading;   
Part4: explain in detail how to load the trained model and forecast the superconducting transition temperature of a virtual system (or an experimenter's system of interest);  
Part5: Manifold Observation of Hidden Layers of DNN.  

In particular, we give additional code for when the model goes hyperparameter tuning, which is based on optuna. The initial definition of the model ground and the tuning intervals are consistent with the representation of the model framework in the supporting material, and once the model has been trained, the residual analysis portion of the code provided in Part3 can be used easily to determine if the model has preferences on the test set.
