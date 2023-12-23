import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import random
from utils import *

#-------The lib below is not functional----------#
from pprint import pprint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sliceAlongAxis(array,axis):
    dims = array.shape
    if (axis==0):
        return array[0,:,:].reshape(1,dims[1],dims[2])
    elif (axis==1):
        return array[:,0,:].reshape(dims[0],1,dims[2])
    elif (axis==2):
        return array[:,:,0].reshape(dims[0],dims[1],1)
    else:
        raise ValueError(f"sliceAlongAxis: Axis {axis} not valid, should be [0,2]")
    
def complementZero(array,complementAxis):
    complementArray = np.zeros(sliceAlongAxis(array,complementAxis).shape)
    array = np.concatenate((complementArray,array),axis=complementAxis)
    return array

class MultiVarDataset():
    def __init__(self,args):
        self.data_settings = args['data_setting']
        self.mode = args['mode']
        self.batch_size = self.data_settings['batch_size']
        self.datasetInfoPath = "./dataInfo/localDataInfo.json" if self.mode in ['debug', 'local_inf'] else './dataInfo/CRCDataInfo.json'
        self.datasetInfo = parseDatasetInfo(self.data_settings['dataset'],self.datasetInfoPath)
        self.dataset_varList = self.data_settings['dataset']
        self.unsupervised = self.data_settings['unsupervised']
        self.scale,self.interval = self.parseScale_Interval(self.data_settings['scale'], self.data_settings['interval'])
        self.splitTimes = self.data_settings['splitTimes']

        self.completeDatasetInfo()
       
        
    def ReadData(self):
        self.data = {dataset_var:[] for dataset_var in self.dataset_varList}
        for d_v in self.dataset_varList:
            d,v = d_v.split('_')
            train_time_steps = self.datasetInfo[d_v]['train_time_steps']
            data_path = self.datasetInfo[d_v]['data_path']
            dim = self.datasetInfo[d_v]['dim']
            self.space_sampled_indices = self.Space_Subsample(dim,d_v)
            for i in train_time_steps:
                if d in ['supercurrent', 'Tornado']:
                    v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype = '<f')
                elif d in ['Cylinder', 'Earthquake', 'Tangaroa']:
                    v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
                elif d in ['Jet']:
                    v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
                else:
                    v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
                
                v = 2*(v-np.min(v))/(np.max(v)-np.min(v)) - 1
                v = v[self.space_sampled_indices]
                self.data[d_v].append(v)
                
    def parseScale_Interval(self,scale_in_setting,interval_in_setting):
        interval = {}
        scale = {}
        if type(interval_in_setting) == int:
            interval = {dataset_var:interval_in_setting for dataset_var in self.dataset_varList}
        if type(scale_in_setting) == int:
            scale = {dataset_var:scale_in_setting for dataset_var in self.dataset_varList}
        if type(interval_in_setting) == list:
            for idx,dataset_var in enumerate(self.dataset_varList):
                interval[dataset_var] = interval_in_setting[idx]    
        if type(scale_in_setting) == list:
            for idx,dataset_var in enumerate(self.dataset_varList):
                scale[dataset_var] = scale_in_setting[idx]  
        return scale, interval
                    
    def completeDatasetInfo(self):
        #* set train and inf time steps for each variable
        for dataset_var in self.datasetInfo:
            total_samples = self.datasetInfo[dataset_var]['total_samples']
            self.datasetInfo[dataset_var]['train_time_steps'] = [i for i in range(1,total_samples+1,self.interval[dataset_var]+1)]
            self.datasetInfo[dataset_var]['total_time_steps'] = [i for i in range(1,total_samples+1)]
            self.datasetInfo[dataset_var]['embeddingIndex'] = self.dataset_varList.index(dataset_var)
        # pprint(self.datasetInfo)
        

    def Space_Subsample(self,dim,d_v):
        space_sampled_indices = []
        for z in range(0,dim[2],self.scale[d_v]):
            for y in range(0,dim[1],self.scale[d_v]):
                for x in range(0,dim[0],self.scale[d_v]):
                    index = (((z) * dim[1] + y) * dim[0] + x)
                    space_sampled_indices.append(index)
        space_sampled_indices = np.asarray(space_sampled_indices)
        return space_sampled_indices

    
    def getMultiVarTrainLoader(self):
        #* should work even dims are not same for all dataset
        training_data_input = []
        training_data_output = []
        for d_v in self.dataset_varList:
            train_time_steps = self.datasetInfo[d_v]['train_time_steps']
            total_samples = self.datasetInfo[d_v]['total_samples']
            embeddingIndex = self.datasetInfo[d_v]['embeddingIndex']
            HR_dim = self.datasetInfo[d_v]['dim']
            LR_dims = [d//self.scale[d_v] for d in HR_dim]
            for index,sampled_time in enumerate(train_time_steps):
                output_val = self.data[d_v][index].reshape(-1,1)
                split_output,reverseBuffer = self.splitVolumes(data=output_val,splitTimes=self.splitTimes,dims=LR_dims)
                if index == 0:
                    HR_split_dims = [dim*self.scale[d_v] for dim in reverseBuffer['split_dims']]
                    reverseBuffer['HR_split_dims'] = HR_split_dims
                    space_sampled_indices = self.Space_Subsample(HR_split_dims,d_v)
                    split_coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
                    split_coords = split_coords[space_sampled_indices]
                    if self.unsupervised:
                        reverseBuffer['HR_split_dims'] = [int(d*1.75) for d in HR_split_dims] #* d times unsample factor
                    self.datasetInfo[d_v]['reverseBuffer'] = reverseBuffer
                
                time_col = (2*(sampled_time-1)/(total_samples-1)-1)*np.ones((split_coords.shape[0],1))
                embeddingIndex_col = embeddingIndex*np.ones((split_coords.shape[0],1))
                input_coord = np.concatenate((embeddingIndex_col,time_col,split_coords),axis=1)

                training_data_input += list(input_coord)
                training_data_output += list(split_output)
        
        training_data_input = torch.FloatTensor(np.array(training_data_input))
        training_data_output = torch.FloatTensor(np.array(training_data_output))
        data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
        train_loader = DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True)

        return train_loader  

    def getMultiVarInfLoader(self):
        test_input_coords = {dataset_var:{"t":[],"coords":None} for dataset_var in self.dataset_varList}
        for d_v in self.dataset_varList:
            # dim = self.datasetInfo[d_v]['dim']
            reverseBuffer = self.datasetInfo[d_v]['reverseBuffer']
            # print(reverseBuffer)
            HR_split_dims = reverseBuffer['HR_split_dims']
            if not self.unsupervised:
                coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
            else:
                coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
            total_samples = self.datasetInfo[d_v]['total_samples']
            total_time_steps = self.datasetInfo[d_v]['total_time_steps']
            test_input_coords[d_v]['coords'] = coords
            for t in total_time_steps:
                test_input_coords[d_v]['t'].append((2*(t - 1)/(total_samples - 1)-1))
        return test_input_coords 
    
    def splitVolumes(self,data,splitTimes,dims):
        data_ls = [data.reshape(dims[2],dims[1],dims[0]).transpose()]
        reverseBuffer = {"split_dims":[],"split_axis_order":[],"splitTimes":splitTimes,"complemtAxis":[-1 for i in range(splitTimes*3)]}
        split_dims = dims.copy()
        # for j in range(splitTimes*3): #* this is split accross 3 axis 
        for j in range(splitTimes): 
            res = []
            split_axis = j%3
            reverseBuffer['split_axis_order'].append(split_axis)
            complementHappen = False
            for data_idx in range(len(data_ls)):
                if(data_ls[data_idx].shape[split_axis]%2 != 0):
                    data_ls[data_idx] = complementZero(data_ls[data_idx],split_axis)
                    complementHappen = True
                
                split_arrays = np.split(data_ls[data_idx],indices_or_sections=2,axis=split_axis)
                res.append(split_arrays[0])
                res.append(split_arrays[1])
            data_ls = res
            
            if complementHappen:
                reverseBuffer['complemtAxis'][j] = split_axis
                split_dims[split_axis] = (split_dims[split_axis]+1)//2
            else:
                split_dims[split_axis] = split_dims[split_axis]//2

        data_ls = [v.flatten('F').reshape(-1,1) for v in data_ls] 
        
        reverseBuffer['split_dims'] = split_dims
        reverseBuffer['split_axis_order'].reverse()
        reverseBuffer['complemtAxis'].reverse()
        res = np.concatenate(data_ls,axis=-1)
        return res,reverseBuffer  
    

def multiSplitFlips(v_ls,axis):
    for i in range(0,len(v_ls),2):
        v_ls[i] = volumeMirrorFlip(v_ls[i], axis)
    return v_ls

def volumeMirrorFlip(v,axis):
    v = np.flip(v,axis)
    return v
        

if __name__ == "__main__":
   settings = yaml_loader('./tasks/configs/multilass.yml')
   D = MultiVarDataset(settings)
   
    
    