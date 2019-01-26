# COCO data preparation

1. Download train and validation images from [here](http://cocodataset.org/#download).

2. Install [COCO API](https://github.com/cocodataset/cocoapi).

3. Run `prepare_COCO.ipynb` to prepare annotations.

4. Create tfrecords:  
  ```
  python create_tfrecords.py \
      --image_dir=/home/dan/datasets/COCO/images/train2017/ \
      --annotations_dir=/home/dan/datasets/COCO/train_annotations/ \
      --output=/home/dan/datasets/COCO/train_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1000

  python create_tfrecords.py \
      --image_dir=/home/dan/datasets/COCO/images/val2017/ \
      --annotations_dir=/home/dan/datasets/COCO/val_annotations/ \
      --output=/home/dan/datasets/COCO/val_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1
  ```  

# Visualization notebooks

1. Use `test_input_pipeline.ipynb` to see how my data augmentation works.
2. Use `visualize_anchor_boxes.ipynb` to see what my anchor boxes look like.
3. Use `visualize_anchor_matching.ipynb` to see how training target creation works.
