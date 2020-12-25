#images_1 = np.load('data/Fold 1/images/fold1/images.npy').astype(np.uint8) 
#masks_1 = np.load('data/Fold 1/masks/fold1/masks.npy').astype(np.uint8) 

types_1 = np.load('data/Fold 1/images/fold1/types.npy')

maskNames = ["neoplastic", "inflammatory", "connective", "dead", "epithelial", "background"]
def plot_images_mask(i, images, masks, figsize = (20,10)):
    '''i -- index of samle
       images: numpy array (uint8) format of image data (,256,256,3)
       masks: corresponsing array (uint8) format of image data (,256,256,6)'''
    global maskNames
    global types_1
    plt.figure(figsize=figsize)
    plt.subplot(171)
    plt.imshow(images[i])
    plt.title(types_1[i] + " " + str(i))
    plt.axis("off")
    
    for j in range(6):
        plt.subplot(172+j)
        plt.imshow(masks[i,:,:,j])
        plt.title(maskNames[j])
        plt.axis("off")
        
    plt.show()

#for i in range(500, 510):
#    plot_images_mask(i, images_1, masks_1)
