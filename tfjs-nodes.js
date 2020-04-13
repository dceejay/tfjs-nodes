module.exports = function (RED) {
  // Common stuff for all nodes
  var tf = require('@tensorflow/tfjs-node')
  var fs = require('fs')
  global.fetch = require('node-fetch')

  function setNodeStatus (node, status) {
    switch (status) {
      case 'modelReady':
        node.status({ fill: 'green', shape: 'dot', text: 'ready' })
        break
      case 'modelLoading':
        node.status({ fill: 'yellow', shape: 'ring', text: 'loading model...' })
        break
      case 'infering':
        node.status({ fill: 'blue', shape: 'ring', text: 'infering...' })
        break
      case 'modelError':
        node.status({ fill: 'red', shape: 'dot', text: 'model error' })
        break
      case 'error':
        node.status({ fill: 'red', shape: 'dot', text: 'error' })
        break
      case 'close':
        node.status({})
        break
      default:
        node.status({ fill: 'grey', shape: 'dot', text: status })
    }
  }

  function inputNodeHandler (node, msg) {
    try {
      if (node.ready) {
        var image = msg.payload
        if (typeof image === 'string') { image = fs.readFileSync(image) }
        node.inferImage(image).then(
          function (results) {
            msg.payload = results
            setNodeStatus(node, 'modelReady')
          }
        )
      } else {
        node.error('model is not ready')
      }
    } catch (error) {
      node.error(error)
      setNodeStatus(node, 'error')
    }
  }

  // Specific implementations for each of the nodes
  function tensorflowPredict (config) {
    RED.nodes.createNode(this, config)
    this.modelUrl = config.model
    // this.modelfile = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
    var node = this

    async function loadModel (modelUrl) {
      setNodeStatus(node, 'modelLoading')
      await tf.loadLayersModel(modelUrl).then(function (model) {
        var shape = model.inputs[0].shape
        shape.shift()
        // node.log('input model shape: ' + shape)
        shape.unshift(1)
        model.predict(tf.zeros(shape)).dispose()
        node.model = model
        node.shape = shape
        node.ready = true
        setNodeStatus(node, 'modelReady')
      })
    }

    async function imgToTensor (p) {
      // if it's a string assume it's a filename
      if (typeof p === 'string') { p = fs.readFileSync(p) }
      const img = tf.node.decodeImage(p, node.shape[3])
      // rescale the image to fit the wanted shape
      const scaled = tf.image.resizeBilinear(img, [node.shape[1], node.shape[2]], true)
      const offset = tf.scalar(127.5)
      // Normalize the image from [0, 255] to [-1, 1].
      const normalized = scaled.sub(offset).div(offset)
      // extend the tensor to 4d
      return normalized.reshape(node.shape)
    }

    loadModel(node.modelUrl)

    node.on('input', function (msg) {
      msg.inputShape = node.shape
      try {
        imgToTensor(msg.payload)
          .then(
            function (tensorImage) {
              var tensorResult = node.model.predict(tensorImage)
              msg.maxIndex = tensorResult.argMax(1).dataSync()[0]
              msg.payload = Array.from(tensorResult.dataSync())
              node.send(msg)
            }
          )
      } catch (error) {
        node.error(error, msg)
      }
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  function tensorflowMobilenet (config) {
    var mobilenet = require('@tensorflow-models/mobilenet')

    RED.nodes.createNode(this, config)
    this.modelUrl = config.modelUrl || 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json'
    this.threshold = config.threshold
    this.saveImage = config.saveImage || false

    var node = this

    async function loadModel (modelUrl) {
      setNodeStatus(node, 'modelLoading')
      try {
        node.model = await mobilenet.load()
        node.ready = true
        setNodeStatus(node, 'modelReady')
      } catch (error) {
        setNodeStatus(node, 'modelError')
        node.error(error)
      }
    }

    node.inferImage = async function (image) {
      setNodeStatus(node, 'infering')
      var tensorImage = tf.node.decodeImage(image)
      var results = await node.model.classify(tensorImage)
      return results
    }

    loadModel(node.modelUrl)

    node.on('input', function (msg) {
      inputNodeHandler(node, msg)
      node.send(msg)
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  function tensorflowCocoSsd (config) {
    var cocoSsd = require('@tensorflow-models/coco-ssd')
    var express = require('express')
    var compression = require('compression')

    RED.nodes.createNode(this, config)
    this.scoreThreshold = config.scoreThreshould
    this.maxDetections = config.maxDetections

    var node = this

    RED.httpNode.use(compression())
    RED.httpNode.use('/coco', express.static(__dirname + '/models/coco-ssd'))

    async function loadModel () {
      setNodeStatus(node, 'modelLoading')
      node.model = await cocoSsd.load({ modelUrl: 'http://localhost:1880/coco/model.json' })
      node.ready = true
      setNodeStatus(node, 'modelReady')
    }

    loadModel()

    node.on('input', function (msg) {
      async function inferTensorImage (image) {
        setNodeStatus(node, 'infering')
        msg.maxDetections = msg.maxDetections || node.maxDetections || 20
        msg.payload = await node.model.detect(image, msg.maxDetections)
        msg.shape = image.shape
        msg.classes = {}
        msg.scoreThreshold = msg.scoreThreshold || node.scoreThreshold || 0.5

        for (var i = 0; i < msg.payload.length; i++) {
          if (msg.payload[i].score < msg.scoreThreshold) {
            msg.payload.splice(i, 1)
            i = i - 1
          }
          msg.classes[msg.payload[i].class] = (msg.classes[msg.payload[i].class] || 0) + 1
        }

        node.send(msg)
        setNodeStatus(node, 'modelReady')
      }
      try {
        if (node.ready) {
          var image = msg.payload
          if (typeof image === 'string') { image = fs.readFileSync(image) }
          inferTensorImage(tf.node.decodeImage(image))
        }
      } catch (error) {
        node.error(error, msg)
      }
    })

    node.on('close', function () {
      setNodeStatus(node, 'close')
    })
  }

  function tensorflowPosenet (config) {
    var posenet = require('@tensorflow-models/posenet')

    RED.nodes.createNode(this, config)
    this.scoreThreshold = config.scoreThreshould
    this.maxDetections = config.maxDetections

    var node = this

    async function loadModel () {
      setNodeStatus(node, 'modelLoading')
      node.model = await posenet.load()
      node.ready = true
      setNodeStatus(node, 'modelReady')
    }

    loadModel()

    node.on('input', function (msg) {
      async function inferTensorImage (image) {
        setNodeStatus(node, 'infering')
        msg.scoreThreshold = msg.scoreThreshold || node.scoreThreshold || 0.5
        msg.maxDetections = msg.maxDetections || node.maxDetections || 4
        var poses = await node.model.estimateMultiplePoses(image, {
          flipHorizontal: false,
          maxDetections: msg.maxDetections,
          scoreThreshold: msg.scoreThreshold,
          nmsRadius: 20
        })
        msg.payload = poses
        for (var i = 0; i < msg.payload.length; i++) {
          if (msg.payload[i].score < msg.scoreThreshold) {
            msg.payload.splice(i, 1)
            i = i - 1
          }
        }
        msg.shape = image.shape
        msg.classes = { person: msg.payload.length }
        node.send(msg)
        setNodeStatus(node, 'modelReady')
      }
      try {
        if (node.ready) {
          var image = msg.payload
          if (typeof image === 'string') { image = fs.readFileSync(image) }
          inferTensorImage(tf.node.decodeImage(image))
        }
      } catch (error) {
        node.error(error, msg)
      }
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  // Register Node-RED nodes
  RED.nodes.registerType('tensorflowPredict', tensorflowPredict)
  RED.nodes.registerType('tensorflowMobilenet', tensorflowMobilenet)
  RED.nodes.registerType('tensorflowCocoSsd', tensorflowCocoSsd)
  RED.nodes.registerType('tensorflowPosenet', tensorflowPosenet)
}
