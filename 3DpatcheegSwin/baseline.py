from typing import Dict, Union, List

from torcheeg.transforms import EEGTransform


class BaselineRemoval(EEGTransform):
    r'''
    A transform method to subtract the baseline signal (the signal recorded before the emotional stimulus), the nosie signal is removed from the emotional signal unrelated to the emotional stimulus.
    
    TorchEEG recommends using this class in online_transform for higher processing speed. Even though, this class is also supported in offline_transform. 
    Usually, the baseline needs the same transformation as the experimental signal, please add :obj:`apply_to_baseline=True` to all transforms 
    before this operation to ensure that the transformation is performed on the baseline signal

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToTensor(apply_to_baseline=True),
            transforms.BaselineRemoval(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])

        t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg'].shape
        >>> (4, 9, 9)
    
    .. automethod:: __call__
    '''
    def __init__(self):
        super(BaselineRemoval, self).__init__(apply_to_baseline=False)

    def __call__(self, *args, eeg: any, baseline: Union[any, None] = None, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            eeg (any): The input EEG signal.
            baseline (any) : The corresponding baseline signal.

        Returns:
            any: The transformed result after removing the baseline signal.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: any, **kwargs) -> any:
        if kwargs['baseline'] is None:
            return eeg

        assert kwargs[
            'baseline'].shape == eeg.shape, f'The shape of baseline signals ({kwargs["baseline"].shape}) need to be consistent with the input signal ({eeg.shape}). Did you forget to add apply_to_baseline=True to the transforms before BaselineRemoval so that these transforms are applied to the baseline signal simultaneously?'
        return eeg - kwargs['baseline']

    @property
    def targets_as_params(self):
        return ['baseline']
    
    def get_params_dependent_on_targets(self, params):
        return {'baseline': params['baseline']}

class BaselineCorrection(EEGTransform):
    r'''
    A transform method to subtract the mean of baseline signal from EEG.
    
    TorchEEG recommends using this class in online_transform for higher processing speed. Even though, this class is also supported in offline_transform. Usually, the baseline needs the same transformation as the experimental signal, please add :obj:`apply_to_baseline=True` to all transforms before this operation to ensure that the transformation is performed on the baseline signal

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.BaseCorrection()

        t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg'].shape
        >>> (32,128)
    
    .. automethod:: __call__
    '''
    def __init__(self,axis=-1):
        super(BaselineCorrection, self).__init__(apply_to_baseline=False)
        self.axis=axis

    def __call__(self, *args, eeg: any, baseline= None, **kwargs) :
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)


    def apply(self, eeg, **kwargs) -> any:
        
         if kwargs['baseline'] is None:
            return eeg
         return eeg - kwargs['baseline'].mean(self.axis,keepdims= True)
    
    @property
    def targets_as_params(self):
        return ['baseline']
    
    def get_params_dependent_on_targets(self, params):
        return {'baseline': params['baseline']}
