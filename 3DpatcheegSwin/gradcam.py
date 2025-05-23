import numpy as np

from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image (Batch, Features, Width, Height)
        if len(grads.shape) == 4: 
            return np.mean(grads, axis=(2, 3))
        
        # 3D image (Batch, Features, Times, Width, Height)
        elif len(grads.shape) == 5:
            # print(grads.shape)  (32, 1536, 4, 3, 3)
            return np.mean(grads, axis=(2, 3, 4)) # 对每个通道的 feature map 上所有像素求平均，
                                                  # 代表该通道对目标类别的整体“重要性” —— 
                                                  # 这是经典 Grad-CAM 的权重计算方式。
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")