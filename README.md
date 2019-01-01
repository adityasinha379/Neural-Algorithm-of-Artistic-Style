This folder consists of two subfolders, 'Code' and 'Images'. Since our project, "A Neural Algorithm of Artistic Style" doesn't need a huge dataset, we instead have our own images (private dataset) that we have uploaded here. The folder 'Images' has three subfolders: 'Content' contains content images, 'Style' contains style images and 'Results' has all our artistic images.
 
In the Code folder, the main Python notebook 'NeuralStyleTransfer.ipynb' has the main code for taking input, running the model, and training and generating artistic output images. This is empowered by the following three python scripts:  
1: **loss.py** - has all the functions to calculate content loss, style loss and total variation loss. These are then summed up in the function _total_loss_.   
2: **vggnet.py** - contains the VGG class that defines the VGG network configuration.  
3: **utils.py** - has functions _load_image_ and _imsave_ to load and save the images (perform I/O).  
 
We use a pre-trained VGG network, with the network weights being stored in _imagenet-vgg-verydeep-19.mat_, which can be downloaded from https://www.kaggle.com/teksab/imagenetvggverydeep19mat. This is a large file, and along with vggnet.py, it acts as a proxy for the CNN model.
