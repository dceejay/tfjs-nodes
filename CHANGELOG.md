# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
 - New `tf mobilenets` node
 - Standarized node status defined globally for all nodes
 - `JavaScript` code has been standarized following [Standard JS](https://standardjs.com/index.html)
 - Badges in `README` (platform Node-RED, code style, licence)
 - `CHANGELOG` file
 - New contributor `Yair Bonastre` in `package.json`
 - `google` keyword to `package.json` 
 - New `Parameters` section in configuration
 - Added `Online` and `Local` options for all nodes (`tf coco ssd` is the only supported for now)

### Changed
 - General polish `JavaScript` code across all nodes
 	- Variable names
 	- Refining general reusable functions
 - Tensorflow node logo updated to 2.0
 - Node color matches official Tensorflow orange color
 - Node-RED example (improved)
 - Better icons in nodes configuration
 - Standard configuration templates
 - Revised information help menus 
 - Name of npm module to `node-red-contrib-tfjs-nodes` and its description
 - JavaScript file from `tfjs.js` to `tfjs-nodes.js`
 - Set all used libraries to `const` variables
 - `Min. Threshold` and `Max. Detections` are now selected from an easier spinner
 - `Min. Threshold` is now defined from 0 to 100 in %

### Fixed
 - Correct model initial loading for `tf coco ssd` node 

### Removed
 - Dependancies from `express` and `compression` npm modules
 - Override of messages properties - will come back on a future release


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).