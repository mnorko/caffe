# SSD For Text Detection and Reading

### Overview

1. For SSD text detection testing code, see /examples/ssd/python. Further documentation is presented within that folder

2. For prototxt files for text reading, see /examples/ocr/90ksynth

3. For prototxt files and testing code for the combined text detection/word reading network, see /examples/ocr. This folder also contains the code for performing net surgery to combine the pre-trained weights for the detection and reading network. as well as the net surgery code to combine detection and read. Further documentation is presented within that folder

4. For ipython notebooks for testing text detection, tex reading, and the full combined network, see /examples. Further documentation is provided within that folder.

5. For networks and code that test the spatial transformer network on MNIST, see examples/mnist. Further documentation provided within the folder

### Python Layers for Caffe

All of the new caffe layers that were written for the combined text detector and reader are in /examples/pycaffe/layers. 

1. Add `$CAFFE_ROOT/examples/pycaffe/layers` to your PYTHONPATH

2. Set up the PyYAML package which is used to read the parameters for the python layers. Information about the set up for this layer is found [here](http://pyyaml.org/wiki/PyYAML). If you are planning on reading parameters in the layer, `import yaml` in the layer .py file and load params in using `yaml.load(self.param_str)`

3. To successfully use the python layers in caffe, you will need to update you Makefile.config and uncomment `WITH_PYTHON_LAYER := 1`. Build caffe after making this change. 

4. To define a python layer in a prototxt file, set the following: 
  ```Shell
  type: "Python"
  bottom: bottom_blob
  top: top_blob
  python_param {
    module: "name of .py file containing the layer definition"
    layer: "name of layer class defined in the module file"
    param_str: "{'parameter1': param1_val, 'parameter2': param2_val}"
  }
  ```
