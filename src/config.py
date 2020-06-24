dim = (256, 256, 1)  # gray images
SEED = 1

# Labels/n_classes
labels = sorted(['gun', 'knife'])
# count background
n_classes = len(labels) + 1 # background
# number filters
n_filters = 24
# model name
model_name = '%s_model.hdf5'
# annotation file name by default
# all files should have the same name
ann_file_name = 'coco_annotation.json'
# Batch size
batch_size = 32
