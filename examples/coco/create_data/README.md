## Preparation
1. Download Images and Annotations from [MSCOCO](http://mscoco.org/dataset/#download). By default, we assume the data is stored in `$HOME/data/coco`

2. Get the coco code. We will call the directory that you cloned coco into `$COCO_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/coco.git
  cd coco
  git checkout dev
  ```
3. Build the coco code.
  ```Shell
  cd PythonAPI
  python setup.py build_ext --inplace
  ```
4. Download Annotations for [COCO-Text](http://vision.cornell.edu/se3/coco-text/)

5. Get the COCO-Text code [here](https://github.com/andreasveit/coco-text)

6. Add directories for COCO and COCO-Text code to PYTHONPATH

7. Read annotation files and create .json files for each image which contain image and annotation information
  ```Shell
  # To create json files for COCO-Text/COCO combined annotations which splits up legible and illegible text into two different classes
  python examples/coco/create_data/split_annotations_cocoCombo_splitLegible.py
  # To create json files for only COCO-Text annotations of legible, english text for the join network
  python examples/coco/create_data/split_annotations_cocoText_legibleOnly.py
  ```
8. Create name_size file which prints the image name, height, and width. This file should be the same no matter what the annotations are as long as the set of training and validation images is the same
  ```Shell
  python examples/coco/create_data/get_image_size_COCOText.py
  ```

9. Create text file that contains image file, annotation file pairings. This should also be the same no matter what the annotations look like. Specify the "train_list_file", the "val_list_file", and the datasets for training and validation
  ```Shell
  python examples/coco/create_data/create_list_cocoCombo.py
  ```
10. Create lmdb files for both training and validation sets. Depending on the dataset being create, you will need to update the mapfile to point to the labelmap with the correct class-label association and specify the subsets. If you want to created annotated word data, change the label_type to "json_word"
  ```Shell
  ./examples/coco/create_data/create_data_cocotext_combo_legible.sh
  ```

