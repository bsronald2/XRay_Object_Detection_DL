dim = (512, 512, 1)  # gray images
SEED = 1
multiply_by = 10
# Labels/n_classes
labels = sorted(['gun', 'knife'])
hues_labels = {'gun':0, 'knife': 25, 'background': 45}
# count background
n_classes = len(labels) + 1 # background
# number filters
n_filters = 32
# model name
model_name = '%s_model.hdf5'
# annotation file name by default
# all files should have the same name
ann_file_name = 'coco_annotation.json'
# Batch size
batch_size = 4
reports_path = "reports/figures"
