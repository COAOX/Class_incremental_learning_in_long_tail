# Class_incremental_learning_in_long_tail
Class_incremental_learning_in_long_tail
Simple Implementation of an algorithm in Class-incremental learning in long tail environment in PyTorch.
# Download Dataset
## Cifar-100
You may need git bash to run this command
```
cd src
sh download_cifar.sh 
```
# Run Model
Use `Python` command to run this code, like this.
```
cd src
python train.py --stage=5 --cuda -nep=55 --batch_size=256 --lossF='NCM'  --Data_file='train_meta'  --pushR=0.0001 --pillR=0.001 --Bias_epoch=0 --mix
```
