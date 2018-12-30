#!/bin/sh

python infer.py \
    --test_list_path "./test-c24.lst"\
    --test_dir "../dataset/TGN/cropped" \
    --pickle_path "./dict_class_neuron.pkl" \
    --batch_size 16\
    --model_path "./checkpoint-vgg-c24/model-vgg-c24-ep90.pth"\
    --num_workers 4
