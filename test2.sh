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

LD_LIBRARY_PATH=$PWD/build/lib/:$LD_LIBRARY_PATH
if [ "$JOB" == "train" ]; then
	./build/tools/caffe $JOB -model phast/cifar/cifar10_quick_train_test.prototxt -phast_conf_file phast/cifar/conf_file.yml -iterations 1000 --solver=phast/cifar/cifar10_quick_solver.prototxt
else
	./build/tools/caffe $JOB -model examples/cifar10/cifar10_full.prototxt -phast_conf_file phast/cifar/conf_file.yml -iterations 1000
	#cuda-gdb --tui --args ./build/tools/caffe $JOB -model phast/cifar/cifar10_quick_train_test.prototxt -weights phast/cifar/cifar10_quick_iter_5000.caffemodel -phast_conf_file phast/cifar/conf_file.yml -iterations 1
fi
