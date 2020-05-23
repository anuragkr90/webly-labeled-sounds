#!/bin/bash

devices=2,3
optm='adam'
net='newNET2'
wt_decay=0.0
segmapping=avg
schedule=N
preload='AUD'
div_apply=Y+KL.B+0.3
epoch=250


lr=0.0001
./run_filtering_ffnet_mtl.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch $div_apply 

