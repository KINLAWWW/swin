import numpy as np
import torch
from copy import deepcopy
from torcheeg.transforms import EEGTransform
from typing import Dict, Union



def get_swap_pair(location_dict):
    location_values = list(location_dict.values())
    eeg_width = np.array(location_values)[:,1].max()
    visited = [False for _ in location_values]
    swap_pair = []
    for i,loc in enumerate(location_values):
        if visited[i]:
            continue
        x,y = loc
        target_loc = [x,eeg_width - y]
        for j,loc_j in enumerate(location_values[i:]):
            #print(loc_j)
            if target_loc == loc_j:
                if target_loc[1] >=y:
                    swap_pair.append((i,j+i))
                else:
                    swap_pair.append((j+i,i))
                visited[i] = True
                visited[j] = True
                break
    return swap_pair


def horizontal_flip(eeg,eeg_channel_dim,pair):
    eeg = deepcopy(eeg)
   
    for i,(index1,index2) in enumerate(pair):
        slice_tuple1 = tuple(slice(None) if i != eeg_channel_dim else index1 for i in range(eeg.ndim))
        slice_tuple2 = tuple(slice(None) if i != eeg_channel_dim else index2 for i in range(eeg.ndim))
        t =  deepcopy(eeg[slice_tuple1])
        eeg[slice_tuple1] = eeg[slice_tuple2]
        eeg[slice_tuple2] = t
    return eeg 


def get_align_eeg(eeg,eeg_channel_dim,pair):
    new_shape = [2]
    new_shape.extend(list((s if i!=eeg_channel_dim else len(pair)  for i,s in enumerate(eeg.shape))))
    if isinstance(eeg,np.ndarray):
        eeg_align = np.empty(new_shape)
    else:
        eeg_align = torch.empty(*new_shape)
    for index, (left_index,right_index) in enumerate(pair):
        slice_tuple1 = tuple(slice(None) if i != eeg_channel_dim else left_index for i in range(eeg.ndim))
        slice_tuple2 = tuple(slice(None) if i != eeg_channel_dim else right_index for i in range(eeg.ndim))

        target_slice_tuple = tuple(slice(None) if i != eeg_channel_dim else index for i in range(eeg.ndim))
        left_eeg = eeg_align[0]
        right_eeg = eeg_align[1]
        left_eeg[target_slice_tuple] = deepcopy(eeg[slice_tuple1])
        right_eeg[target_slice_tuple] = deepcopy(eeg[slice_tuple2])
    
    # eeg_align[0] = left_eeg
    # eeg_align[1] = right_eeg

    return eeg_align

class HorizontalFlip(EEGTransform):
    r'''
    Flip the EEG signal horizontally based on the electrode’s position.

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets.constants.motor_imagery import BCICIV2A_LOCATION_DICT

        # numpy based example
        eeg = np.random.randn(32,4,22,128 )   # batchsize, num_inchannels, num_electrodes, num_time_points
        t = transforms.HorizontalFlip(
            location_dict =  BCICIV2A_LOCATION_DICT,
            channel_dim = 2
        )
        t(eeg = eeg)['eeg'].shape
        >>> (32,4,22,128)

        # torch based example
        eeg = torch.randn(4,22,128 )   # batchsize, num_inchannels, num_electrodes, num_time_points
        t = transforms.HorizontalFlip(
            location_dict =  BCICIV2A_LOCATION_DICT,
            channel_dim = 1
        )
        t(eeg = eeg)['eeg'].shape
        >>> (4,22,128)

    Args:
        location_dict (dict): The dict of electrodes and their postions. 
        channel_dim (int): The dim of electrodes in EEG data.

    .. automethod:: __call__
    '''

    def __init__(self, 
                 location_dict: Union[dict,None],
                 channel_dim: int =0):
        
        super().__init__(apply_to_baseline = False)
        self.location_dict = location_dict
        self.swap_pair =get_swap_pair(location_dict)
        self.channel_dim = channel_dim
        
    def apply(self, eeg: any, **kwargs) -> any:
        eeg = horizontal_flip(eeg,self.channel_dim,self.swap_pair)
        return eeg 
    
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'apply_to_baseline': self.apply_to_baseline,
                                          'loaction_dict':self.location_dict,
                                          'channel_dim':self.channel_dim})



class SymmetricAlign(EEGTransform):
    def __init__(self,location_dict, 
                 channel_dim:int =0,
                 retain_middle:bool = False,
                 apply_to_baseline: bool = False):
        
        super().__init__(apply_to_baseline)
        self.channel_dim= channel_dim
        self.location_dict = location_dict
        self.symetry_pair = get_swap_pair(location_dict)
        self.retain_middle = retain_middle
        
        if not retain_middle:
            symetry_pair_ = []
            for (index1,index2) in self.symetry_pair:
                if index1 != index2:
                    symetry_pair_.append((index1,index2))
            self.symetry_pair = symetry_pair_

    def apply(self, eeg: any, **kwargs) -> any:
        eeg = get_align_eeg(eeg,self.channel_dim,self.symetry_pair)
        return eeg 
    


class BSI(EEGTransform):
    def __init__(self,
                 location_dict,
                 channel_dim:int = 0,
                 time_dim:int= 1,
                apply_to_baseline: bool = False):
        
        super().__init__(apply_to_baseline)
        self.location_dict = location_dict
        self.channel_dim = channel_dim 
        self.time_dim = time_dim
        
        self.symetry_pair = get_swap_pair(location_dict)
        symetry_pair_ = []
        for (index1,index2) in self.symetry_pair:
            if index1 != index2:
                symetry_pair_.append((index1,index2))
        self.symetry_pair = symetry_pair_

    


    @EEGTransform.numpy_based
    def apply(self, eeg: any,**kwargs) -> any:
        
        eeg = np.fft.fft(eeg,axis=self.time_dim)
        align_eeg = get_align_eeg(eeg,self.channel_dim,self.symetry_pair)
        left_eeg= align_eeg[0]
        right_eeg= align_eeg[1]
        
        L =left_eeg.mean(self.channel_dim,keepdims=True)
        R = right_eeg.mean(self.channel_dim,keepdims=True)
        BSI = np.abs((R-L)/(R+L)).mean(self.time_dim)
        return BSI
    

    







