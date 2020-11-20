#History of accuracy and loss
import matplotlib.pyplot as plt
def plot_history(history1):
    tra_loss1 = history1.history['loss']
    tra_dice_coef = history1.history['dice_coef']
    val_loss1 = history1.history['val_loss']
    val_dice_coef = history1.history['val_dice_coef']

    # Total number of epochs training
    epochs1 = range(1, len(tra_dice_coef)+1)
    end_epoch1 = len(tra_dice_coef)

    # Epoch when reached the validation loss minimum
    opt_epoch1 = val_loss1.index(min(val_loss1)) + 1

    # Loss and accuracy on the validation set
    opt_val_loss1 = val_loss1[opt_epoch1-1]
    opt_val_dice_coef = val_dice_coef[opt_epoch1-1]
    print("Model 1\n")
    print("Epoch [opt]: %d" % opt_epoch1)
    print("Valid dice coef [opt]: %.4f" % opt_val_dice_coef)
    # print("Test dice coef [opt]:  %.4f" % opt_test_dice_coef)
    print("Valid loss [opt]: %.4f" % opt_val_loss1)
    # print("Test loss [opt]:  %.4f" % opt_test_loss1)
    fig,ax=plt.subplots(1,2,figsize= (8,4))
    ax[0].plot(epochs1, tra_dice_coef, label='Training set')
    ax[0].plot(epochs1, val_dice_coef, label='Validation set')
    ax[0].set_title('Model dice coefficient')
    ax[0].set_ylabel('Dice coefficient')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(epochs1, tra_loss1, label='Training set')
    ax[1].plot(epochs1, val_loss1, label='Validation set')
    ax[1].set_title('Model binary crossentropy loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    plt.legend()
