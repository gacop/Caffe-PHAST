#!/bin/bash

#- algo1:
#  major_block_size: 2
#  minor_block_size: 32
#  scheduling_strategy: 2
#  tiling_strategy: 1
#  shared_pre_load: 0
#  n_thread: 8

if (( $# < 2 )); then
	echo "Usage: ./autotune.sh <arch> <conf_file> <script_file> <algo_list_file>"
	echo "arch: CUDA or MULTI"
	exit 1
fi

ARCH=$1
CONF_FILE=$2
SCRIPT_FILE=$3
algos=$(cat $4 | sed 's/\n/ /')

for algo in $algos
do
	echo $algo
	START_LINE_NUM=$(grep -n "\- $algo" $CONF_FILE | grep -oE [0-9]+\:- | sed 's/:-//')
	if [ -z "$START_LINE_NUM" ]; then
		echo "- $algo:" >> $CONF_FILE
		echo "  major_block_size: 1" >> $CONF_FILE		
		echo "  minor_block_size: 1" >> $CONF_FILE		
		echo "  scheduling_strategy: 1" >> $CONF_FILE		
		echo "  n_thread: 1" >> $CONF_FILE		

		START_LINE_NUM=$(grep -n "\- $algo" $CONF_FILE | grep -oE [0-9]+\:- | sed 's/:-//')
	fi

	Mbs_LINE=$(( START_LINE_NUM + 1 ))
	mbs_LINE=$(( START_LINE_NUM + 2 ))
	sched_LINE=$(( START_LINE_NUM + 3 ))
	n_thr_LINE=$(( START_LINE_NUM + 4 ))

	best_Mbs=1
	best_mbs=1
	best_sched=1
	best_n_thr=1
	LONG_TIME=1000000000000
	BEST_TIME=$LONG_TIME
	if [ $ARCH == 'CUDA' ]; then
		for Mbs in 1 2 4 8 16 32 64 128 256
		do
			for mbs in 1 2 4 8 16 32
			do
				if (( $mbs * $Mbs < 32 || $mbs * $Mbs > 512 )); then
					continue
				fi

				for sched in 1 2
				do
					
					sed -i "${Mbs_LINE}s/.*/  major_block_size: $Mbs/" $CONF_FILE
					sed -i "${mbs_LINE}s/.*/  minor_block_size: $mbs/" $CONF_FILE
					sed -i "${sched_LINE}s/.*/  scheduling_strategy: $sched/" $CONF_FILE

					bash $SCRIPT_FILE time &> out
					TIME=$(cat out | grep -oE "Total Time\:.*" | sed "s/Total Time: //" | sed "s/ ms//")
                    if [ -z "$TIME" ]; then
                        TIME=$LONG_TIME
                    fi

					if (( $( echo "$TIME < $BEST_TIME" | bc -l ) )); then
						best_Mbs=$Mbs
						best_mbs=$mbs
						best_sched=$sched
						BEST_TIME=$TIME
					fi
				done
			done
		done

		sed -i "${Mbs_LINE}s/.*/  major_block_size: $best_Mbs/" $CONF_FILE
		sed -i "${mbs_LINE}s/.*/  minor_block_size: $best_mbs/" $CONF_FILE
		sed -i "${sched_LINE}s/.*/  scheduling_strategy: $best_sched/" $CONF_FILE
	else
		for n_thr in 1 2 4 8 16
		do
			sed -i "${n_thr_LINE}s/.*/  n_thread: $n_thr/" $CONF_FILE

			bash $SCRIPT_FILE time &> out
			TIME=$(cat out | grep -oE "Total Time\:.*" | sed "s/Total Time: //" | sed "s/ ms//")
			if (( $( echo "$TIME < $BEST_TIME" | bc -l ) )); then
				best_n_thr=$n_thr
				BEST_TIME=$TIME
			fi
		done

		sed -i "${n_thr_LINE}s/.*/  n_thread: $best_n_thr/" $CONF_FILE
	fi
done
