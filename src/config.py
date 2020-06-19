dim = (256, 256, 1)  # gray images
SEED = 1

# Labels/n_classes
labels = ['gun']
# count background
n_classes = len(labels)
# number filters
n_filters = 16
# model name
model_name = '%s_model.hdf5'
# annotation file name by default
# all files should have the same name
ann_file_name = 'coco_annotation.json'
# Batch size
batch_size = 16
