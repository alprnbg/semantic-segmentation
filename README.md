# Semantic Segmentation Training
- It is recommended to use torch 1.11
- Run train.py directly to start training on the sample dataset provided in the repo.
- For command line arguments, check parse_args in utils.py.
- Hyperparameter search is implemented in this repo with the help of Optuna library. Search space can be defined as a yaml file (see search.yml)
- Augmentation can be removed or new augmentation can be added by changing the source code. (Check train_transform_list and val_transform_list in dataset.py)
- Trained models can be directly exported to jit or onnx. Check the corresponding scripts in the scripts folder. 
- For vino export, first do onnx export, then run this command "mo --input_model model.onnx".
"# semantic-segmentation" 
