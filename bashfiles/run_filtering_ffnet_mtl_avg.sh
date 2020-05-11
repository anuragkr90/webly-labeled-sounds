#!/bin/bash



devices=0,1
optm='avg_test'
net='newNET2'
segmapping=avg
preload=AUD
feat_model=model_path.10_embedding_audioset_feat_architecture_new_dataloader_0.0005_10.pkl
#../models/output_youtube_top_50_40_embedding_new_dataloader_0.0002/model_path.youtube_top_50_40_embedding_new_dataloader_0.0002_6.pkl
trtssplit=youtube_top_50_40



exp=4k-$optm'-'$net'-'$segmapping'-'$preload


#top 50
#preload1=output/adam-newNET2-0.0-0.0006-avg-N-AUD/model_path.adam-newNET2-0.0-0.0006-avg-N-AUD_8.pkl
#preload2=../models/output_youtube_top_50_40_embedding_new_dataloader_0.0002/model_path.youtube_top_50_40_embedding_new_dataloader_0.0002_6.pkl

#top 100
preload1=output/4k-adam-newNET2-0.0-0.0007-avg-N-AUD/model_path.4k-adam-newNET2-0.0-0.0007-avg-N-AUD_4.pkl
preload2=model_path.youtube_top_50_40_all_embedding_new_dataloader_0.00002_26.pkl #model_path.youtube_top_50_40_all_embedding_new_dataloader_0.00005_9.pkl

echo $exp

CUDA_VISIBLE_DEVICES=$devices stdbuf -oL python cnn_filtering_ffnet_mtl_avg.py --training_features_directory ../features/embedding/youtube_top_40_original/youtube_top_40_original --validation_features_directory ../features/embedding/validation_set_original_segmented_downloaded/ --testing_features_directory ../features/embedding/eval_segments_original_segmented_downloaded/ --classCount 40 --spec_count 128 --val_test_list  10 9 8 7 6 5 --train_list 2 3 4 5 6 7 8 9 10 15 20 25 30 35 36 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 249   --train_test_split_path "../train_test_split_"$trtssplit"_embedding/" --run_number $exp --sgd_batch_size 48 --feat_model_path $feat_model --segmapping $segmapping --newnet $net --preload $preload --preloading2 Y --preloading_model2_path $preload2 --preloading1 Y --preloading_model1_path $preload1  > logfiles/$exp.log

