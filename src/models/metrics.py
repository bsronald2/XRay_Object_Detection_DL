import tensorflow.keras.backend as K
import numpy as np


def iou_metric(y_exp, y_pre):
    assert y_exp.shape == y_pre.shape, f'Masks have different shapes: expected ({y_exp.shape}) vs ' \
                                                  f'predicted ({y_pre.shape})'
    n_classes = K.shape(y_exp)[-1]

    # One-hot encoding
    y_pre = K.one_hot(K.argmax(y_pre), n_classes)
    y_exp = K.one_hot(K.argmax(y_exp), n_classes)

    # Calculate metrics
    axes = (1, 2)  # W,H axes of each image
    intersection = K.sum(K.abs(y_exp * y_pre), axis=axes)
    mask_sum = K.sum(K.abs(y_exp), axis=axes) + K.sum(K.abs(y_pre), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)

    print('intersection, union')
    print(K.eval(intersection), K.eval(union))
    print(K.eval(intersection / union))

    return iou


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric2(y_true_in, y_pred_in, verbose=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

        # Loop over IoU thresholds

    prec = []
    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)