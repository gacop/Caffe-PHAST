#!/bin/bash

if (( $# == 0 )); then
	echo "please, provide a mode parameter: time, test, or train"
	exit 1
fi
JOB=$1

if [ "$JOB" != "time" && "$JOB" != "test" && "$JOB" != "train" ]; then
	echo "please, provide a mode parameter: time, test, or train"
	exit 1
fi


LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/build/lib/
if [ "$JOB" == "train" ]; then
	./build/tools/caffe $JOB -model phast/mnist/lenet_train_test.prototxt -weights phast/mnist/lenet_iter_10000.caffemodel -phast_conf_file phast/mnist/conf_file.yml -iterations 10 --solver=phast/mnist/lenet_solver.prototxt
else
	./build/tools/caffe $JOB -model phast/mnist/lenet_train_test.prototxt -weights phast/mnist/lenet_iter_10000.caffemodel -phast_conf_file phast/mnist/conf_file.yml -iterations 10
	#cuda-gdb --tui --args ./build/tools/caffe $JOB -model phast/mnist/lenet_train_test.prototxt -weights phast/mnist/lenet_iter_10000.caffemodel -phast_conf_file phast/mnist/conf_file.yml -iterations 1
	#cuda-memcheck ./build/tools/caffe $JOB -model phast/mnist/lenet_train_test.prototxt -weights phast/mnist/lenet_iter_10000.caffemodel -phast_conf_file phast/mnist/conf_file.yml -iterations 1
fi
