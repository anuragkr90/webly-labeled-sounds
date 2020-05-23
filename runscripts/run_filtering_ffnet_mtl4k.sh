#!/bin/bash

echo $@

devices=$1
optm=$2
net=$3
wt_decay=$4
segmapping=$5
schedule=$6
preload=$7
lr=$8
nepoch=$9
div_apply=${10}


exp=4k-$optm'-'$net'-'$wt_decay'-'$lr'-'$segmapping'-'$schedule'-'$preload'-'$div_apply

trtssplit=youtube_top_100_40

feat_model=model_path.10_embedding_audioset_feat_architecture_new_dataloader_0.0005_10.pkl
#../models/output_youtube_top_50_40_embedding_new_dataloader_0.0002/model_path.youtube_top_50_40_embedding_new_dataloader_0.0002_6.pkl


echo $exp
CUDA_VISIBLE_DEVICES=$devices stdbuf -oL python cnn_filtering_ffnet_mtl.py --training_features_directory ../features/embedding/youtube_top_40_all_original_embedding/youtube_top_40_all_original --validation_features_directory ../features/embedding/validation_set_original_segmented_downloaded/ --testing_features_directory ../features/embedding/eval_segments_original_segmented_downloaded/ --classCount 40 --spec_count 128 --learning_rate $lr --val_test_list  10 9 8 7 6 5 --train_list 2 3 4 5 6 7 8 9 10 15 20 25 30 35 36 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 249   --train_test_split_path "../train_test_split_"$trtssplit"_embedding/" --num_epochs $nepoch --run_number $exp --sgd_batch_size 48 --feat_model_path $feat_model --segmapping $segmapping --optimizer $optm --newnet $net --weight_decay $wt_decay --schedular $schedule --preload $preload --div_apply $div_apply > logfiles/$exp.log

#CUDA_VISIBLE_DEVICES=$devices stdbuf -oL python cnn_filtering_ffnet_mtl.py --training_features_directory ../features/embedding/youtube_top_40_original/youtube_top_40_original --validation_features_directory ../features/embedding/validation_set_original_segmented_downloaded/ --testing_features_directory ../features/embedding/eval_segments_original_segmented_downloaded/ --classCount 40 --spec_count 128 --learning_rate $lr --val_test_list  10 9 8 7 6 5 --train_list 2 3 4 5 6 7 8 9 10 15 20 25 30 35 36 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 249   --train_test_split_path "../train_test_split_"$trtssplit"_embedding/" --num_epochs $nepoch --run_number $exp --sgd_batch_size 48 --feat_model_path $feat_model --segmapping $segmapping --optimizer $optm --newnet $net --weight_decay $wt_decay --schedular $schedule --preload $preload --div_apply $div_apply --div_type $div_type --preloading2 Y --preloading_model2_path  ../models/output_youtube_top_50_40_embedding_new_dataloader_0.0002/model_path.youtube_top_50_40_embedding_new_dataloader_0.0002_6.pkl #> logfiles/$exp.log
