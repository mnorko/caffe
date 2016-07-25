/home/marissac/ssd/caffe/build/tools/caffe train \
--solver=examples/mnist_tests/ST_CNN/solver.prototxt
--gpu 7 2>&1 | tee /home/marissac/ssd/caffe/examples/mnist/spatial_transform/mnist_spatial_transform.log