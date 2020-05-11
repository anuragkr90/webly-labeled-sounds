#!/bin/bash

devices=0,1,2,3
optm='adam'
net='newNETAutoEncoder'
wt_decay=0.0001
segmapping=avg
schedule=N
preload='AUD'
epoch=100
beta=0.1

lr=0.0001
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0002
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0003
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0004
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0005
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0006
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0007
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0008
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0009
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.001
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0011
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0012
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0013
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0014
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.0015
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.00055
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.00065
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.000075
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.00008
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.00009
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.000095
echo $lr
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.000092
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  

lr=0.000098
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  
lr=0.000085
./run_filtering_ffnet4k_noisy_baseline_method_no_sigmoid.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $beta  