import torch
import numpy as np
import hdf5storage
import time


def get_reconstruction_cpu(input, model, axis):
    
    ####
    model.eval()
    
    input_split = torch.split(input,  int(input.shape[axis]/16), dim=axis)
    output_split = []
    start_time = time.time()
    for i in range(16):
        var_input = input_split[i]
        with torch.no_grad():
            var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=axis)
    end_time = time.time()
        
    return end_time-start_time, output


def reconstruction_whole_image_cpu(rgb, model, axis=3):
    all_time, img_res = get_reconstruction_cpu(torch.from_numpy(rgb).float(), model, axis)
    img_res = img_res.cpu().numpy() * 1.0
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    #img_res_limits = np.minimum(img_res, 1.0)
    #img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
