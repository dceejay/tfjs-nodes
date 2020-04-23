#!/bin/bash

echo "Get CoCo SSD model"
rf -f model/coco-ssd/*
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/group1-shard1of5
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/group1-shard2of5
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/group1-shard3of5
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/group1-shard4of5
wget -q --show-progress -P models/coco-ssd/ https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/group1-shard5of5

echo "Get MobileNetv1 model"
rf -f model/mobilenetv1/*
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/model.json
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/group1-shard1of5 
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/group1-shard2of5 
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/group1-shard3of5 
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/group1-shard4of5 
wget -q --show-progress -P models/mobilenetv1/ https://storage.googleapis.com/tfhub-tfjs-modules/google/imagenet/mobilenet_v1_100_224/classification/1/group1-shard5of5 