Learning and Interpreting Multi-Multi-Instance Learning Networks - Code
----
This repository contains the python code for reproducing the experiments described in the paper Learning and Interpreting Multi-Multi-Instance Learning Networks, by Alessandro Tibo, Manfred Jaeger, and Paolo Frasconi.

Python Requirments
----
* `natsort`
* `numpy`
* `sklearn`
* `tensorflow (version >= 2.0)`
* `tqdm`

Preparation
----
CoreNLP and GloVe installation

	$ cd tools
	$ ./download_tools.sh

	
Run the Experiments for MNIST
----
	$ cd example/mnist
	$ python generate_dataset.py
	$ python train.py 
	
Run the Experiments for IMDB
----
	$ cd example/imdb/dataset
	$ ./create_dataset.sh
	$ cd ..
	$ python train.py 
