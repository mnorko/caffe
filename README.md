# SSD For Text Detection and Reading

### Overview


1. For the test scripts, tools, and prototxt files for the ocr pipeline and the combined text detection/word reading network, see `$CAFFE_ROOT/examples/ocr`. This folder also contains the code for performing net surgery to combine the pre-trained weights for the detection and reading network. Further documentation is presented within that folder

2. Code for checking the backpropagation for python caffe layers is contained in  `$CAFFE_ROOT/python/caffe/test/'. For additional information, see the description in the "Gradient Checker for Python Layers" section below.

3. Files for creating the lmdb data for COCO-Text is found in `$CAFFE_ROOT/examples/coco/create_data`. The documentation on how to create the lmdb folders is included within that folder.

4. The python layers are contained in `$CAFFE_ROOT/examples/pycaffe/layers'. More information on those layers is included in the "Python Layers for Caffe" section below.

5. For networks and code that test the spatial transformer network on MNIST, see examples/mnist. Further documentation provided within the folder



### Python Layers for Caffe

All of the new caffe layers that were written for the combined text detector and reader are in `$CAFFE_ROOT/examples/pycaffe/layers'. 

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

Several layers were created for the combined text reader network.

1. `spatialTransformerMulti.py` - spatial transformer layer that can handle multiple transforms per image. The back propopagation for for bottom[0] using compute_dU is not optimized for speed. However, because we aren't interested in the gradient on the image, I set propagate_down[0] to 0 in the prototxt file.

2. `spatialTransformer.py` - spatial transform layer when there's only a single transformation per image

3. `multiboxRoutingLayer.py` - routes the predicted bbox outputs from the SSD network by matching the ground truth with predicted networks and subselecting the outputs

4. `multibox_util.py` - contains functions called in mutliboxRoutingLayer

5. `bboxToTransformLayer.py` - Converts the bounding box parameters into the theta transform parameters and converts the input images to grayscale.

### Gradient Checker for Python Layers

I updated a gradient checker for python layers that was found [here](https://github.com/tnarihi/caffe/tree/python-gradient-checker). 

1. This code added a "create_layer" option which allows you to create a specific layer and run a forward and backward pass on only that layer.  

2. `$CAFFE_ROOT/python/caffe/gradient_check_util.py` - contains the main gradient checking functions 

3. `$CAFFE_ROOT/python/caffe/test/test_gradient_checker.py` - the file sets up and runs the gradient check for a specified layer. I set up this code to run on the spatial transformer layer. $CAFFE_ROOT/python/caffe/test/test_gradient_checker_multibox_route.py` runs the gradient check for the multibox routing layer which requires inputs from a network to make sure that the label inputs make sense.
