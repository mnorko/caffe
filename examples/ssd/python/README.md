# Text Detector testing scripts

1. eval_ICDAR_results.py - uses the COCO-Text API to find precision-recall values for ICDAR datasets. Prior to running this you need to create the pickle files with annotations and imgToAnns using runTextReader_ICDAR.py

2. ssd_read.py - This file contains the functions to run the vision pipeline with the two different networks. "ssd_detect_box" runs the SSD text detection network. "synth_read_words" runs the word reader network on cropped text inputs

3. 