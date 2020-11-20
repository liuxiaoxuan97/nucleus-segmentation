def train_val_split(images,masks,train_perc=0.7):
    '''
    user case: use fold1/2/3 to get splited train/val/test set
    parameters:
       images: numpy array (uint8) format of image data (,256,256,3)
       masks: corresponsing array (uint8) format of image data (,256,256,6)
       train_perc <1, train_val_split, train_perc==1, user how dataset as train/va/test
       '''
    x = images_1
    y = masks_1[:,:,:,5]
    #converting the inputs to 0 and 1 with 0 the background and 1 is the nuclei
    y = np.where((y==0)|(y==1), y^1, y) # convert 0 to 1; 1 to 0 
            
    # Number of images
    no_img = y.shape[0]
    #print("Number of images : %d \t " % (no_img))

    # Compute width and height of images
    img_ht = x.shape[1]
    img_wd = y.shape[2]
    #print("Image size: %dx%d" % (img_wd, img_ht))
    y = y.reshape(no_img, img_ht, img_ht, 1)

    n_train = int(train_percentage*no_img)
    x_train = x[0:n_train]
    x_val   = x[n_train:]
    y_train = y[0:n_train]
    y_val   = y[n_train:]
    
    return x_train,x_val,y_train,y_val