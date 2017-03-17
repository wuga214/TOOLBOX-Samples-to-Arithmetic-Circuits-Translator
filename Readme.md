Toolbox: Samples to Arithmetic Circuits
===

# Introduction

When you have some data from a conditional distribution P(Y|X), how can you create an AC to capture this distribution and do inference on it? This code solve the problem. 

We assume the data is sampled from a Bayes Network and, for each conditional probability P(x|par(x)), we train an Neural network and transform it into Arithmetic Circuit.

## Author 
[GA WU](mailto:wuga@mie.utoronto.ca), D3M Lab, MIE, University of Toronto

## Language
Python 2.7 + ipython


## Instruction
```Python
	#Variable Names
	parents_names = ['Binary_Variable_088','Binary_Variable_089']
	child_name = 'Binary_Variable_084'
	varindex =['088','089','084']

	#Train Network
	parents = df[parents_names].as_matrix()
	child = df[[child_name]].as_matrix()
	dnn = FullyConnectedNetwork(2,2,4,1)
	dnn.train(parents,child)
	layers = dnn.getLayers()

	#Convert to AC
	conv = NetToAC(parents_names,child_name,varindex,layers)
	ac_stream,_ = conv.getACStream()
	print ac_stream
```

## Related Packages
1. [Bayes Network Generator](https://github.com/wuga214/TOOLBOX-Random-Bayes-Net-Generator)
2. [Arithmetic Circuits to Tensorflow Tensor Compiler](https://github.com/wuga214/TOOLBOX-Arithmetic-Circuits-to-Tensorflow-Compiler)

## Cite
```
@misc{wu_2017, 
title={TOOLBOX:Random Bayes Net Generator}, 
url={https://github.com/wuga214/TOOLBOX-Samples-to-Arithmetic-Circuits-Translator}, 
journal={GitHub}, 
publisher={D3M Lab, MIE, UofT}, 
author={Wu, Ga}, 
year={2017}, 
month={Mar}}
```




