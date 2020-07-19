
import tensorflow.keras.backend as K


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

