# node-red-contrib-tjfs-nodes
[![platform](https://img.shields.io/badge/platform-Node--RED-red)](https://nodered.org)
<br>
[![JavaScript Style Guide](https://img.shields.io/badge/code_style-standard-brightgreen.svg)](https://standardjs.com)
[![GitHub license](https://img.shields.io/github/license/dceejay/tfjs-nodes)](https://github.com/dceejay/tfjs-nodes/blob/master/LICENSE)

A `Node-RED` node for `tensorflow.js`. A simple function like node preloaded with tensorflowjs and some other useful libraries.

**NOTE**: Tensorflow.js is only available in 64bit - so will not run on 32 bit operating systems like Raspbian.

## Install
Either use the Node-RED Menu - Manage Palette option, or run the following command in your Node-RED user directory - typically `~/.node-red`

```
npm i node-red-contrib-tfjs
```

## Overview
This package contains 3 nodes

 - **tf predict** - Runs tf.predict against a model you import
 - **tf mobilenet** - Runs the pretrained mobilenet classifier
 - **tf coco ssd** - Runs the CoCo Single Shot object detector against either a file or a buffer of a jpg image.
 - **tf posenet** - Runs the Posenet body parts model against either a file or a buffer of a jpg image.
