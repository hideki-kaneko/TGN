#!/bin/sh

python train.py \
    --expname "vgg-c24"\
    --train_list_path "./train-c24.lst"\
    --train_dir "../dataset/TGN/cropped"\
    --test_list_path "./test-c24.lst"\
    --test_dir "../dataset/TGN/cropped" \
    --pickle_path "./dict_class_neuron.pkl" \
    --lr 0.0001\
    --batch_size 16\
    --n_epochs 100\
    --model_path "model.pth"\
    --num_workers 4\
    --tensorboard\
    --model_type vgg 
