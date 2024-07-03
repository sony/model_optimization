#!/bin/bash
rm -rf imagenet/val
mkdir imagenet/val
tar -xf imagenet/ILSVRC2012_img_val.tar -C imagenet/val
wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt -O imagenet/val/imagenet_2012_validation_synset_labels.txt
cd imagenet/val; find . -name "*.JPEG" | sort > images.txt
cd imagenet/val; function zip34() { while read word3 <&3; do read word4 <&4 ; echo $word3 $word4 ; done }; zip34 3<images.txt 4<imagenet_2012_validation_synset_labels.txt | xargs -n2 -P8 bash -c 'mkdir -p $2; mv $1 $2' argv0
cd imagenet/val; rm *.txt