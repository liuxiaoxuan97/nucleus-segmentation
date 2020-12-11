from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

#y_pred is predicted probability of classes 
def dice_coef_numpy(y_true,y_pred):
    y_true_f = K.flatten(y_true).numpy()
    y_pred_f= K.flatten((y_pred>0.5).astype('uint8')).numpy()
    intersection=sum(y_true_f*y_pred_f)
    k=(sum(y_true_f) + sum(y_pred_f))
    return (2.0*intersection+1.0)/ (k+1)
