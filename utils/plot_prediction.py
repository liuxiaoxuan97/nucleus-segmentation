
types_3 = np.load('data/Fold 3/images/fold3/types.npy')

maskNames = ["neoplastic", "inflammatory", "connective", "dead", "epithelial", "background"]
def plot_prediction(i, images, masks, model, x_test, figsize = (20,10)):
    global maskNames
    global types_3
    preds_test = model.predict(x_test, verbose=0)
    dice_test= np.round(dice_coef_numpy(test_y,preds_test),4)
    plt.figure(figsize=figsize)
    plt.subplot(181)
    plt.imshow(images[i])
    plt.title(types_3[i] + " " + str(i))
    plt.axis("off")
    
    for j in range(6):
        plt.subplot(182+j)
        plt.imshow(masks[i,:,:,j])
        plt.title(maskNames[j])
        plt.axis("off")
    plt.subplot(188)
    plt.imshow(preds_test[i,:,:,:])
    plt.title('prediction')
    plt.show()

#for i in range(8):
 #   plot_prediction(i, images_3, masks_3)