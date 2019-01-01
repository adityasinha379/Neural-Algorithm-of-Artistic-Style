import tensorflow as tf
import numpy as np

# Compute total loss
def total_loss(target_im, content_shape, tv_weight, target_net, style_layer, style_feats, style_weight,
               content_layer, content_feats, content_weight):

    # Compute style loss
    styleloss = style_loss(target_net, style_layer, style_feats)
    
    # Compute content loss
    contentloss = content_loss(target_net, content_layer, content_feats)
    
    # Compute total variation loss
    tvloss = tv_loss(target_im, content_shape)
    
    # Compute weighted total loss
    totalloss = style_weight*styleloss + content_weight*contentloss + tv_weight*tvloss
    
    return totalloss

def content_loss(target_net, content_layer, content_feats):
    
    # Get features for target image
    target_feats = []
    for layer in content_layer:
        target_feats.append(target_net[layer])
        
    # Add content loss for each feature
    # Content loss is just 2-norm of difference between features
    contentloss = 0
    for num in range(len(content_feats)):
        contentloss += 2*tf.nn.l2_loss(target_feats[num]-content_feats[num]) / tf.size(content_feats[num], out_type=tf.float64)
    
    return contentloss

def style_loss(target_net, style_layer, style_feats):
    
    # Get features for target image
    target_feats = []
    for layer in style_layer:
        target_feats.append(target_net[layer])
        
    styleloss = 0
    for num in range(len(style_feats)):
        # Get features for current layer
        style_feat = style_feats[num]
        target_feat = target_feats[num]
        
        # Get feature dimensions
        [_, height, width, chan] = style_feat.get_shape().as_list()
        feat_size = height*width*chan
        
        # Compute Gram matrix for each feature
        style_vec = tf.reshape(style_feat, (-1, chan))
        style_gram = tf.matmul(tf.transpose(style_vec), style_vec) / feat_size
        
        target_vec = tf.reshape(target_feat, (-1, chan))
        target_gram = tf.matmul(tf.transpose(target_vec), target_vec) / feat_size
        
        # Add style loss for each feature
        styleloss += 2*tf.nn.l2_loss(target_gram-style_gram) / tf.size(style_gram, out_type=tf.float64)
    
    return styleloss

def tv_loss(image, shape):  
	# Define size of matrices minus the dimension over which total variation is calculated
    tv_y_size = np.prod(image[:,1:,:,:].get_shape().as_list())
    tv_x_size = np.prod(image[:,:,1:,:].get_shape().as_list())

    # Calculate the total variation loss as
    # 2-norm between adjacent pixels in x and y directions
    tv_loss = 2 * (
            (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                tv_y_size) +
            (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                tv_x_size))
    return tv_loss