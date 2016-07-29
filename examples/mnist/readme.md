# MNIST Spatial Transformer Tests

### Prototxt files

1. `deploy_lenet_spatial_transform.prototxt` - takes an input image and theta and performs the spatial transform. This was used to test the forward pass of the spatial transformer

2. `lenet_train_test_spatial_transform.prototxt` and `solver_spatial_transform.prototxt` are set up to learn a mapping from an input image to the transform parameters theta prior to transforming the image. The localization network was copied from the mnist tests presented [here](https://github.com/daerduoCarey/SpatialTransformerLayer)

### Test Scripts

1. `test_spatial_transformer.py` and `test_spatial_transformer_multi.py` test the forward pass of the spatial transformer layer (for one transform and mutliple transforms respectively) when you input an image and theta parameters.

2. `mnist_spatial_transform_test.py'- tests the spatial transformer layer performance after training on the mnist network 