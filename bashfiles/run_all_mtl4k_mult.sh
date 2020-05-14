#!/bin/bash

devices=2,3
optm='sgd'
net='newNET2'
wt_decay=0.0
segmapping=avg
schedule=N
preload='AUD'
div_apply=Y+KL.B+0.5
epoch=250


lr=0.01
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0002
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0003
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0004
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0005
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0006
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0007
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0008
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0009
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.001
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0011
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0012
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0013
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.0014
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.00007
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply  

lr=0.00095
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.00085
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.000075
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.00008
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

lr=0.00009
./run_filtering_ffnet_mtl4k_mult.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 
