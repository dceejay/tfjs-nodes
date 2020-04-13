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

  async function inputNodeHandler (node, msg) {
    try {
      if (node.ready) {
        var image = msg.payload
        // If it's a string assume it's a filename
        if (typeof image === 'string') { image = fs.readFileSync(image) }
        var results = await node.inferImage(image)
        setNodeStatus(node, 'modelReady')
        return results
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
    this.modelUrl = config.modelUrl
    this.threshold = config.threshold

    var node = this

    async function loadModel (modelUrl) {
      setNodeStatus(node, 'modelLoading')
      try {
        node.model = await tf.loadLayersModel(modelUrl)
        var shape = node.model.inputs[0].shape
        shape.shift()
        // node.log('input model shape: ' + shape)
        shape.unshift(1)
        node.model.predict(tf.zeros(shape)).dispose()
        node.shape = shape
        node.ready = true
        setNodeStatus(node, 'modelReady')
      } catch (error) {
        setNodeStatus(node, 'modelError')
        node.error(error)
      }
    }

    node.inferImage = async function (image) {
      setNodeStatus(node, 'infering')
      var tensorImage = tf.node.decodeImage(image, node.shape[3])
      // Rescale the image to fit the wanted shape
      var scaledTensorImage = tf.image.resizeBilinear(tensorImage, [node.shape[1], node.shape[2]], true)
      var offset = tf.scalar(127.5)
      // Normalize the image from [0, 255] to [-1, 1].
      var normalized = scaledTensorImage.sub(offset).div(offset)
      tensorImage = normalized.reshape(node.shape)

      var tensorResult = node.model.predict(tensorImage)
      var argMax = tensorResult.argMax(1).dataSync()[0]
      var results = Array.from(tensorResult.dataSync())
      results = [results, argMax]
      return results
    }

    loadModel(node.modelUrl)

    node.on('input', function (msg) {
      inputNodeHandler(node, msg).then(
        function (results) {
          msg.payload = results[0]
          msg.argMax = results[1]
        })
      node.send(msg)
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  function tensorflowMobilenet (config) {
    var mobilenet = require('@tensorflow-models/mobilenet')

    RED.nodes.createNode(this, config)
    this.modelUrl = config.modelUrl
    this.threshold = config.threshold

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
      results = [results]
      return results
    }

    loadModel(node.modelUrl)

    node.on('input', function (msg) {
      inputNodeHandler(node, msg).then(
        function (results) {
          msg.payload = results[0]
        })
      node.send(msg)
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  function tensorflowCocoSsd (config) {
    var cocoSsd = require('@tensorflow-models/coco-ssd')

    RED.nodes.createNode(this, config)
    this.modelUrl = config.modelUrl
    this.threshold = config.threshold
    this.maxDetections = config.maxDetections

    var node = this

    async function loadModel (modelUrl) {
      setNodeStatus(node, 'modelLoading')
      try {
        node.model = await cocoSsd.load()
        node.ready = true
        setNodeStatus(node, 'modelReady')
      } catch (error) {
        setNodeStatus(node, 'modelError')
        node.error(error)
      }
    }

    loadModel(node.modelUrl)

    node.inferImage = async function (image) {
      setNodeStatus(node, 'infering')
      var tensorImage = tf.node.decodeImage(image)
      var results = await node.model.detect(tensorImage, node.maxDetections)
      var classes = {}

      for (var i = 0; i < results.length; i++) {
        if (results[i].score < node.threshold) {
          results.splice(i, 1)
          i = i - 1
        }
        classes[results[i].class] = (classes[results[i].class] || 0) + 1
      }
      results = [results, classes]
      return results
    }

    node.on('input', function (msg) {
      inputNodeHandler(node, msg).then(
        function (results) {
          msg.payload = results[0]
          msg.classes = results[1]
        })
      node.send(msg)
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  function tensorflowPosenet (config) {
    var posenet = require('@tensorflow-models/posenet')

    RED.nodes.createNode(this, config)
    this.modelUrl = config.modelUrl
    this.threshold = config.threshold
    this.maxDetections = config.maxDetections

    var node = this

    async function loadModel (modelUrl) {
      setNodeStatus(node, 'modelLoading')
      try {
        node.model = await posenet.load()
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
      var poses = await node.model.estimateMultiplePoses(tensorImage, {
        flipHorizontal: false,
        maxDetections: node.maxDetections,
        scoreThreshold: node.threshold,
        nmsRadius: 20
      })
      var results = poses
      for (var i = 0; i < poses.length; i++) {
        if (results[i].score < node.threshold) {
          results.splice(i, 1)
          i = i - 1
        }
      }
      results = [results]
      return results
    }

    loadModel(node.modelUrl)

    node.on('input', function (msg) {
      inputNodeHandler(node, msg).then(
        function (results) {
          msg.payload = results[0]
          msg.classes = { person: msg.payload.length }
        })
      node.send(msg)
    })

    node.on('close', function () { setNodeStatus(node, 'close') })
  }

  // Register Node-RED nodes
  RED.nodes.registerType('tensorflowPredict', tensorflowPredict)
  RED.nodes.registerType('tensorflowMobilenet', tensorflowMobilenet)
  RED.nodes.registerType('tensorflowCocoSsd', tensorflowCocoSsd)
  RED.nodes.registerType('tensorflowPosenet', tensorflowPosenet)
}
