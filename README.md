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
## Incremental dataset
You can use `Cifar100.py` to serialize the cifar-100 dataset to 5 stage and each stage contains 20 class.
## Long-tail environment
You may need to define you own function to cut the balanced dataset to imbalanced dataset. 
<br>
For me, I use probability function $f(x)=\frac{1}{20^x}$ and $f(x)=\frac{1}{40^x}$ to cut the dataset.
# Model
This model implements two kinds of feature extractors: simple implementation and resnet.
## Feature Extractor
You can switch two model in `train.py - func main() - line:424`
### Sample Extractor Implementation with 4 conv Layer
Use `model = init_protonet(options)` in `train.py` to set this implementation as extractor.
### ResNet-32 as Extractor
Use `model = PreResNet(32,options.total_cls).cuda()` in `train.py` to set ResNet-32 as extractor.<br>
You can choose and test the effects of the two implementations according to the needs of the experiment.
## Sample Mix
Sampe mix is a method to balance the gap between large sample class and small sample class in feature extractor, and parameters can be seted with operation `--mix` in command to run model. Sample Mix method was defined in `train.py -- line:315`.
***
<br>
Sample mix is based on the fact that the feature extractor will be biased due to the original number difference of data categories in the process of training and classification, which makes it easier for a large number of unknown samples to be classified into the large sample class area and marginalize the small sample class. The method of sample mix can reduce this difference to a certain extent.


## Remeasure —— Bias Layer
Bias Layer was described in file `prototypical_loss.py`, According to the general code specification, this module should be defined in the `model.py`, and you can finish it in your free time.
<br>
Bias Layer defines two parameters which is `alpha` and `beta`, this two paramters works with the distance computed out by Euclidean distance between feature extractor output and prototype center of each category. Core idea was inspired by `Prototypical Network` and `BIC`. You may need to read it to better understand the content of this model.<br>
## Prototype Distance Distillation
Experiments show that avoiding the relative position change of `prototype` in feature space helps to avoid `Catastrophic Forgetting`. In this model, we use `push` and `pull` lose to check and balance the special space reconstruction caused by Incremental learning in the training process.
___
You can use command `--pushR ` and `--pullR` to adjust.
<br>
 `--pushR` : set the lose rate of `push` loss used in `train.py -- line:533`
<br>
 `--pullR` : set the lose rate of `pull` loss used in `train.py -- line:534`
# How to Run
Use `Python` command to run this code, like this.
<br>
`--nep`: number of epoch default 32
<br>
`--batch_size`: size of batch default 
<br>
`--Data_file`: the data file you need to train, which could be original cifar data or the long_tail data cut by yourself
<br>
`--pushR`: the push loss rate defined in Prototype Distance Distillation
<br>
`--pullR`: the pull loss rate defined in Prototype Distance Distillation
<br>
`--Bias_epoch`: the number of epoch to train bias layer
<br>
`--mix`: to spicify whether use Sample Mix
<br>
`--lr`: learning rate
<br>
`--lossF`: to spicify the loss function, which is `NCM` or `CrossEntropyLoss`, check in file `prototypical_loss.py--line:164`
<br>
Other important parameter you can find in file `parser_util.py`
<br>
Command shoule like this.
```
cd src
python train.py --stage=5 --cuda -nep=55 --batch_size=256 --lossF='NCM'  --Data_file='train_meta'  --pushR=0.0001 --pullR=0.001 --Bias_epoch=0 --mix
```
