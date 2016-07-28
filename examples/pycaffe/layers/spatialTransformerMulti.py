# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:47:38 2016

@author: marissac
"""

import caffe 
import numpy as np
import yaml

class SpatialTransformerMultiLayer(caffe.Layer):
    """
    Transform an input using the transformation parameters computed by the 
    localization network. This requires the creation of a mesh grid, a 
    transformation of this gird, and an interpolation of the image along the grid
    """     
    def setup(self, bottom, top):
        self.grid_global = []
        self.x_s = []
        self.y_s = []
        
    def reshape(self,bottom,top):
        params = yaml.load(self.param_str)
        numDetect = params["num_detections"]
        N = bottom[0].shape[0] # number of batches
        C = bottom[0].shape[1] # number of channels
        
       # shape = np.array((N, C, params["output_H"], params["output_W"]))
        top[0].reshape(N*numDetect,C,params["output_H"], params["output_W"])
        
        
    def forward(self,bottom,top):
        """
        Transform the mesh grid
        
        Paremters
        ---------
        top: Transformed image based on the transformer
        bottom: [0] - input image or convolutional outputs [num_batch,num_channels,H,W]
                [1] - theta [num_batch*max_matches,6] - [theta_1_1,theta_1_2,theta_1_3, theta_2_1, theta_2_2, theta_2_3]
        """
        theta = bottom[1].data
        img = bottom[0].data
        
        
        params = yaml.load(self.param_str)
        num_detect = bottom[1].shape[0]/bottom[0].shape[0]
        N = bottom[0].shape[0]*num_detect # number of batches

        predefined_theta = {}
        # Set the theta values to the predefined parameters if they are specified
        if params.has_key("theta_1_1"):
            predefined_theta["theta_1_1"] = params["theta_1_1"]
            theta[:,0] = predefined_theta["theta_1_1"]*np.ones((N))
        if params.has_key("theta_1_2"):
            predefined_theta["theta_1_2"] = params["theta_1_2"]
            theta[:,1] = predefined_theta["theta_1_2"]*np.ones((N))
        if params.has_key("theta_1_3"):
            predefined_theta["theta_1_3"] = params["theta_1_3"]
            theta[:,2] = predefined_theta["theta_1_3"]*np.ones((N))
        if params.has_key("theta_2_1"):
            predefined_theta["theta_2_1"] = params["theta_2_1"]
            theta[:,3] = predefined_theta["theta_2_1"]*np.ones((N))
        if params.has_key("theta_2_2"):
            predefined_theta["theta_2_2"] = params["theta_2_2"]
            theta[:,4] = predefined_theta["theta_2_2"]*np.ones((N))
        if params.has_key("theta_2_3"):
            predefined_theta["theta_2_3"] = params["theta_2_3"]
            theta[:,5] = predefined_theta["theta_2_3"]*np.ones((N))
        
        # Reshape theta so it is num_batchx2x3
        theta = np.reshape(theta,[-1,2,3])
        
        out_height = params["output_H"]
        out_width = params["output_W"]
        out_size = np.array([out_height,out_width])
        
        global grid
        # Create the grid of (x_t, y_t, 1), eq (1)
        grid = create_meshgrid(out_height,out_width)
        # The initial grid is 3 x (out_height*out_width), to work with several 
        # batches, set this to num_batch x 3 x (out_height*out_width)
        grid = np.expand_dims(grid,0)
        grid = np.reshape(grid,[-1])
        grid = np.tile(grid,N)
        grid = np.reshape(grid,[N,3,-1])
        self.grid_global = grid
        
        # Transform the mesh grid using the transformations specified by theta
        T_g =  batch_matmul(theta,grid)
        # Isolate and flatten x_s and y_s

        self.x_s = T_g[:,0,:]
        self.y_s = T_g[:,1,:]
        x_s_flat = np.reshape(self.x_s,[-1])
        y_s_flat = np.reshape(self.y_s,[-1])
    
        top[0].data[...] = interpolate(img, x_s_flat,y_s_flat,out_size,num_detect)

    def backward(self, top, propagate_down, bottom):
        """
        Backpropagate the gradients for theta and the input U
        Parameters
        ---------
        top: information from the above layer with the transformed image
        bottom: [0] - input image or convolutional layer input [num_batch x num_channels x input_height x input_width]
                [1] - theta parameters [num_batch*max_matches x Num_transform_param] - in this case the num_transform_param = 6
        """
        img = bottom[0].data
        theta = bottom[1].data
        
        params = yaml.load(self.param_str)
        out_height = params["output_H"]
        out_width = params["output_W"]
        out_size = np.array([out_height,out_width])
        num_detect = bottom[1].shape[0]/bottom[0].shape[0]
        
        top_diff = top[0].diff
        
        x_s_flat = np.reshape(self.x_s,[-1])
        y_s_flat = np.reshape(self.y_s,[-1])
        
        if propagate_down[0]: # Propagate convolutional input gradients down
            bottom[0].diff[...] = compute_dU(img,x_s_flat,y_s_flat,out_size,top_diff,num_detect)
            
        if propagate_down[1]: # Propagate theta gradients down
            bottom[1].diff[...] = compute_dTheta(img,x_s_flat,y_s_flat,out_size,theta,top_diff,num_detect,self.grid_global)
def check_params(params):
    assert params["transform_type"] == "affine","Only supports affine transformations"
    assert params["sampler_type"] == "bilinear","Only supports bilinear interpolation"   
    
def create_meshgrid(height,width):
    x_t, y_t = np.meshgrid(np.linspace(0, width-1, width),
                          np.linspace(0, height-1, height))
    x_t = x_t*2/(width-1) - 1
    y_t = y_t*2/(height-1) -1
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    
    return grid

def batch_matmul(theta,grid):
    """
    Multiply the tensors containing the theta and grid for every input in the batch
    Parameters
    ---------
    theta: tensor containing the transformation parameters [num_batch x 2 x 3]
    grid:  tensor contraining the target coordinates for the output grid
    """
    N = theta.shape[0]
    num_out = grid.shape[2]
    output_grid = np.zeros((N,2,num_out))
    for k in range(0,N):
        output_grid[k,:,:] = np.dot(theta[k,:,:],grid[k,:,:])
        
    return output_grid

def interpolate(img,x,y,out_size,num_detect):
    """
    Interpolate between values in the original image to create the transformed output
    
    Parameters
    -----------
    img:        Input image or convolutional layer output [num_batch x num_channels x height x width]
    x:          x_s (source coordinates) [num_batch*output_height*output_width]
    y:          y_s (source coordinates) [num_batch*output_height*output_width]
    out_size:   [output_height, output_width]
    
    Output
    ------
    interp_image: interpolated image [num_batch*max_matches x num_channels x output_height x output_width]
    
    The interpolation in the spatial transformer multi layer needs to make sure to keep track of images appropriately
    """
    N = img.shape[0]*num_detect

    C = img.shape[1]
    H = img.shape[2]
    W = img.shape[3]
    
    x = x.astype(float)
    y = y.astype(float)
    H = float(H)-1.0
    W = float(W)-1.0
    out_height = out_size[0]
    out_width = out_size[1]
    
    output_combo = out_height*out_width
    
    #max_x = W - 1
    max_x = W
    max_x = int(max_x)
    #max_y = H - 1
    max_y = H
    max_y = int(max_y)
    
    # scale indices from [-1,1] to [0, width/height]
    x = (x+1.0)*W/2.0
    y = (y+1.0)*H/2.0

    # Get the closest x and y grid samples to the x,y locations
    x0 = np.floor(x)
    x0 = x0.astype(int)
    x1 = x0 + 1
    
    y0 = np.floor(y)
    y0 = y0.astype(int)
    y1 = y0 + 1
   
    
    # Ensure that the x0,x1,y0,y1 values are in the image size
    x1_weight = np.clip(x1,0,max_x+1)
    y1_weight = np.clip(y1,0,max_y+1)
    x0 = np.clip(x0,0,max_x)
    x1 = np.clip(x1,0,max_x)
    y0 = np.clip(y0,0,max_y)
    y1 = np.clip(y1,0,max_y)
    
    
    
    # Find the pixel values at each of these locations and how close they
    # are to the real points
    interp_image = np.zeros((N,C,out_height,out_width))
    for k in range(0,N):
        x0_batch = x0[output_combo*k:output_combo*(k+1)]
        x1_batch = x1[output_combo*k:output_combo*(k+1)]
        y0_batch = y0[output_combo*k:output_combo*(k+1)]
        y1_batch = y1[output_combo*k:output_combo*(k+1)]
        x1_w_batch = x1_weight[output_combo*k:output_combo*(k+1)]
        y1_w_batch = y1_weight[output_combo*k:output_combo*(k+1)]
        
        xs_batch = x[output_combo*k:output_combo*(k+1)]
        ys_batch = y[output_combo*k:output_combo*(k+1)]
        

        
        w0_0 = np.clip(1-abs(xs_batch-x0_batch),0,1000)*np.clip(1-abs(ys_batch-y0_batch),0,1000)
        w0_1 = np.clip(1-abs(xs_batch-x0_batch),0,1000)*np.clip(1-abs(ys_batch-y1_w_batch),0,1000)
        w1_0 = np.clip(1-abs(xs_batch-x1_w_batch),0,1000)*np.clip(1-abs(ys_batch-y0_batch),0,1000)
        w1_1 = np.clip(1-abs(xs_batch-x1_w_batch),0,1000)*np.clip(1-abs(ys_batch-y1_w_batch),0,1000)
        
        
        w0_0 = np.transpose(np.repeat(np.expand_dims(w0_0,1),C,axis=1))
        w0_1 = np.transpose(np.repeat(np.expand_dims(w0_1,1),C,axis=1))
        w1_0 = np.transpose(np.repeat(np.expand_dims(w1_0,1),C,axis=1))
        w1_1 = np.transpose(np.repeat(np.expand_dims(w1_1,1),C,axis=1))
        
        idx_use_0_0 = y0_batch*int(W+1) + x0_batch
        idx_use_0_1 = y1_batch*int(W+1) + x0_batch
        idx_use_1_0 = y0_batch*int(W+1) + x1_batch
        idx_use_1_1 = y1_batch*int(W+1) + x1_batch
        
        # Only use a new image when you've run through all the thetas for a single image
        if k % num_detect == 0:
            img_idx = k/num_detect
            img_batch = np.reshape(img[img_idx,:,:,:],[C,-1])
            
        img_0_0 = img_batch[:,idx_use_0_0]
        img_0_1 = img_batch[:,idx_use_0_1]
        img_1_0 = img_batch[:,idx_use_1_0]
        img_1_1 = img_batch[:,idx_use_1_1]

        img_sum = np.multiply(img_0_0,w0_0) + np.multiply(img_0_1,w0_1) + np.multiply(img_1_0,w1_0) + np.multiply(img_1_1,w1_1) 
        interp_image[k,:,:,:] = np.reshape(img_sum,[C,out_height,out_width])
        
    return interp_image
    
def compute_dU(img,x,y,out_size,top_diff,num_detect):
    """ 
    Compute the derivative of the objective function in terms of the input image or convolutional layer U
    This back propagation function is not optimized to be fast. If you want to use it, you should update this piece to get rid of some of the for loops
    
    Parameters
    ----------
    img:        Input image or convolutional layer output [num_batch x num_channels x height x width]
    x:          x_s (source coordinates) [num_batch*output_height*output_width]
    y:          y_s (source coordinates) [num_batch*output_height*output_width]
    out_size:   [output_height, output_width]
    top_diff:   Gradient from the top layer [num_batch x channels x output_height x output_width]
    """
    # First compute dV/dx and dV/dY
    N = img.shape[0]
    C = img.shape[1]
    H = img.shape[2]
    W = img.shape[3]
    
    x = x.astype(float)
    y = y.astype(float)
    
    # scale indices from [-1,1] to [0, width/height]
    x = (x+1.0)*(W-1)/2.0 # x_s
    y = (y+1.0)*(H-1)/2.0 # y_s

    
    out_height = out_size[0]
    out_width = out_size[1]
    
    output_combo = out_height*out_width
    
    total_derivative = np.zeros((N,C,H,W))
    for k in range(0,N):
        for d_idx in range(0,num_detect):
            dervTemp = np.zeros((C,H,W))
            x_batch = x[output_combo*d_idx+num_detect*k:output_combo*(d_idx+1)+num_detect*k]
            y_batch = y[output_combo*d_idx+num_detect*k:output_combo*(d_idx+1)+num_detect*k]
            for c_idx in range(0,C):
                top_derv = top_diff[k*num_detect+d_idx,c_idx,:,:]
                top_derv = np.reshape(top_derv,-1)
                for n in range(0,H):
                    for m in range(0,W):
                        # Find all the x and y values close to this location
                        x_vals = np.clip(1-abs(x_batch-m),0,1000)
                        y_vals = np.clip(1-abs(y_batch-n),0,1000)
    
                        xy_prod = np.multiply(x_vals,y_vals)
                        
                        # Multiply the dervivative by dE/dV from the layer above
                        xy_prod = np.multiply(xy_prod,top_derv)
        
                        dervTemp[c_idx,n,m] = np.sum(xy_prod)
            total_derivative[k,:,:,:] = total_derivative[k,:,:,:] + dervTemp        
    return total_derivative
                
def compute_dTheta(img,x,y,out_size,theta,top_diff,num_detect,grid):
    """ 
    Compute the derivative of the objective function in terms of theta
    
    Parameters
    ----------
    img:        Input image or convolutional layer output [num_batch x num_channels x height x width]
    x:          x_s (source coordinates) [num_batch*output_height*output_width]
    y:          y_s (source coordinates) [num_batch*output_height*output_width]
    out_size:   [output_height, output_width]
    theta:      transformation params [num_batch x num_transfor_params]
    top_diff:   Gradient from the top layer [num_batch x channels x output_height x output_width]
    """ 
    # First compute dV/dx and dV/dY
    N = img.shape[0]*num_detect
    C = img.shape[1]
    H = img.shape[2]
    W = img.shape[3]
    
    x = x.astype(float)
    y = y.astype(float)
    
    # scale indices from [-1,1] to [0, width/height]
    x = (x+1.0)*(W-1)/2.0 # x_s
    y = (y+1.0)*(H-1)/2.0 # y_s
    
    out_height = out_size[0]
    out_width = out_size[1]
    
    output_combo = out_height*out_width
    
    total_derivative = np.zeros((N,theta.shape[1]))
    for k in range(0,N):
        if np.all(top_diff[k,:,:,:]== 0) != True:
            x_batch = x[output_combo*k:output_combo*(k+1)]
            y_batch = y[output_combo*k:output_combo*(k+1)]
            dx_total = np.zeros(x_batch.shape)
            dy_total = np.zeros(y_batch.shape)
            
            grid_mult_x= np.zeros((theta.shape[1],output_combo))
            grid_mult_x[0:3,:] = grid[k,0:3,:]
            
            grid_mult_y  = np.zeros((theta.shape[1],output_combo))
            grid_mult_y[3:6,:] = grid[k,0:3,:]
            
            w_vals = range(0,W)
            h_vals = range(0,H)
            
            # Find all the x and y values close to this location
            x_sub = np.expand_dims(x_batch,1)- w_vals # The result of this should be output_combo x W
            y_sub = np.expand_dims(y_batch,1) - h_vals # The result of this should be ouput_combo x H
            
            x_vals = np.clip(1-abs(x_sub),0,1000)
            y_vals = np.clip(1-abs(y_sub),0,1000)
            
            # Find where the values of abs(m-x_batch) > 1
            mult_val_y = np.clip(-y_sub,-1, 1)
            mult_val_y = np.sign(mult_val_y)
            
            mult_val_x = np.clip(-x_sub,-1, 1)
            mult_val_x = np.sign(mult_val_x)
            
            abs_y = abs(-y_sub)
            abs_x = abs(-x_sub)
    
            del x_sub
            del y_sub
            
            large_state_y = np.where(abs_y>=1)
            mult_val_y[large_state_y] = 0
            
            large_state_x = np.where(abs_x>=1)
            mult_val_x[large_state_x] = 0
    
            del abs_x
            del abs_y
            del large_state_x
            del large_state_y
            
            # Create tensors with the values of y_vals,x_vals, mult_val_x, and mult_val_y for easy multiplication
            # We want each tensor to have the size output_comboxHxW
            x_vals = np.repeat(np.expand_dims(x_vals,1),H,axis =1)
            mult_val_x = np.repeat(np.expand_dims(mult_val_x,1),H,axis =1)
            
            y_vals = np.repeat(np.expand_dims(y_vals,2),W,axis =2)
            mult_val_y= np.repeat(np.expand_dims(mult_val_y,2),W,axis =2)
            
            dx_tensor = np.multiply(y_vals,mult_val_x)
            dy_tensor = np.multiply(x_vals,mult_val_y)
    
            del x_vals
            del y_vals
            del mult_val_x
            del mult_val_y
            
            for c_idx in range(0,C):
                top_derv_tensor = top_diff[k,c_idx,:,:]
                top_derv_tensor = np.reshape(top_derv_tensor,-1)
                
                top_derv_tensor = np.repeat(np.expand_dims(top_derv_tensor,1),H,axis = 1)
                top_derv_tensor = np.repeat(np.expand_dims(top_derv_tensor,2),W,axis = 2)
                
                dx_tensor = np.multiply(top_derv_tensor,dx_tensor)
                dy_tensor = np.multiply(top_derv_tensor,dy_tensor)
                
                del top_derv_tensor
    
                img_idx = k/num_detect
                
                img_tensor = np.repeat(np.expand_dims(img[img_idx,c_idx,:,:],0),output_combo,axis = 0)
                
                dx_tensor_total = np.multiply(dx_tensor,img_tensor)
                dy_tensor_total = np.multiply(dy_tensor,img_tensor)
                
                dx_tensor_total = np.reshape(dx_tensor_total,(output_combo,H*W))
                dy_tensor_total = np.reshape(dy_tensor_total,(output_combo,H*W))
                
                
                dx_total = dx_total + np.sum(dx_tensor_total,axis=1)
                dy_total = dy_total + np.sum(dy_tensor_total,axis=1)
    
                del img_tensor
                del dx_tensor_total
                del dy_tensor_total
            
            dx_total = dx_total*(H-1)/2
            dy_total = dy_total*(W-1)/2
            total_derivative[k,:] = np.dot(grid_mult_x,dx_total) + np.dot(grid_mult_y,dy_total)      
    return total_derivative