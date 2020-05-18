#!/bin/bash

devices=0,1
optm='adam'
net='newNET2'
wt_decay=0.0
segmapping=avg
schedule=N
preload='AUD'
epoch=100


lr=0.0001
./run_filtering_ffnet4k.sh $devices $optm $net $wt_decay $segmapping $schedule $preload $lr $epoch  

