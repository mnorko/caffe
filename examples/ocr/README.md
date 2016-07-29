# OCR Tests

### Overview

This folder contains protoxt files for the combined network, a script for combined weights from the text detector and word reader networks, as well as code for testing the end-to-end text reading pipeline.

Note: In several of the test scripts I use a function from coco_evaluation called reduceDetections. This function is included in the ["reduce_detect" branch of the coco-text repository on my corp git](https://git.corp.yahoo.com/marissac/coco-text/tree/reduce_detect) 

### Code for vision pipeline

```detect_read_ssd.py``` contains the functions that can be included in the vision OCR pipeline and they should be generally compatible with Rob's current inputs and outputs. ```ssd_detect_box.py``` uses the SSD network to read an image and output bounding boxes. ```synth_read_words.py``` uses the 90ksynth network to read the words in the cropped text images.

### Testing OCR Pipeline

runTextReader files are used for testing the OCR pipeline using the detector and word reader separately.

1. ```runTextReader.py``` uses the COCO-Text API to load images and uses ```detect_read_ssd.py``` to detect and read words. This outputs two pickle files: one contains the annotations as a list of dictionaries, the second specifies the imgToAnns values. This is a list where each entry corresponds to an image and contains a new list of the annotation ids associated with the given image. 

2. ```runTextReader_ICDAR.py``` is similar to ```runTextReader.py``` but it also compute pickle files for the ground truth ICDAR data to allow for evaluation

3. ```runTextReader_fromLmdb.py``` this is the code used to test the results on the gpu machines. It reads the images from the lmdb folder rather than the individual image files. 

When testing, you will need to specify the coco_labelmap_file, the model_def, and model weights for the SSD network, the synth_model and the synth_model_weights for the text reader network, the 90ksynth mean file, the name_size_file, and the lmdb directory.

You can specify the number of the number of test images you want to run through the network (```numImgTest```). You can also specify the confidence threshold for detections (```thresh_use```). In all my tests I set the threshold to 0 so I get all the detections, and once I have the pickle files, I can play with the threshold in the future to get PR curves. 

There are a few ipython notebooks to get a closer view of the outputs of the text detector and reader networks.

1. `ssd_compute_detections_newNet.ipynb` - visualizes examples of text detection with COCO-text data. The first cell loads the network. The second cell computes the detections and matches predicted and ground truth detections. The thrid cell outputs the cropped words and the predicted text associated with each one. The fourth cell plots the ground truth and predicted detections on top of the image. In the first cell, imgIds are defined using ct.getImgIds. This is where you can specify if you want images from the training or validation set and what categories of annotations you want the images to have. Vary thresh_use to see the detections for different confidence threshold.

2. `ssd_inputImage.ipynb` - visualizes cropped words and predicted text for any input image that you specify in the varialble "fileName"

3. `ssd_detect_ICDAR.ipynb` - this provides similar information to `ssd_compute_detections_newNet.ipynb` but the inputs are from ICDAR

### Computing Precision-Recall Metrics

1. `coco_text_createPRCurve_ICDAR.py` and `coco_text_createPRCurve_cocoText.py` - use the pickle outputs from the runTextReader scripts to compute PR curves for ICDAR and COCO-Text respectively. I ran these scripts to find the threshold that results in the highest f-score. 

2. `eval_ICDAR_results.py` and `eval_cocoText_results.py` - use the modified COCO-Text API to compute the detections and the precision and recall metrics. These scripts also use the pickle outputs from the runTextReader scripts. You can set the confidence threshold using the `confidence` variable. To get the best f-score results, I used the threshold from the PRCurve scripts

3. `create_coco_dist_plots.py` - plots a variety of distributions from the coco-Text dataset to try to understand patterns to why we get errors

### Combined Text Detector and Word Reader Network

This folder contains prototxt files and code used to test and train the combined text detection and word reading network.

1. The first task in creating the combined network is combining the weights from the pretrained detection and reading networks. ```combineNetworkWeights.py``` performs the net surgery to combine the weights from the two networks. Specify the caffemodel files for the two pretrained networks (model_weights and synth_model_weights) and the text reader network and combination network prototxt files (synth_model and model_combo)

2.  test_detectReadCombo.protxt, train_detectReadCombo.prototxt, and solver_detectReadCombo.prototxt are all set up for the combination network. The train network has all three loss layers at the top. You should be able to change the weights for each loss layer to vary its contribution during training by changing the "loss_weight" in the prototxt file. Additionally, the "ignore_label" in the SoftmaxWithLoss layers indicates which class label should be ignored.

3. The combined weights that I got after running ```combineNetworkWeights.py``` is included in detect_2444000_read_140000_combo_final.caffemodel

4. `fullComboNet.ipynb` is a ipython notebook that contains examples of the forward pass of the combined text detector and reader. 

### ICDAR

```eval_ICDAR_results.py```- uses the COCO-Text API to find precision-recall values for ICDAR datasets. Prior to running this you need to create the pickle files with annotations and imgToAnns using ```runTextReader_ICDAR.py``` 

### Text Reader Network

The folder ```90ksynth``` contains the files needed for the text reader network. I have been using deploy_2.prototxt to peform word reading for input images. deploy_ocr_spatial_transform_tullTest.prototxt is used to test the spatial transform on the word reader network.
