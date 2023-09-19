import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
from sr.utils.utils import get_config

class Metrics():
    def __init__(self, config, log = None):
        metrics = config.metrics
        self.log = log
        implemented = ['psnr','mse', 'ssim']
        for value in metrics:
            assert value in implemented , f"sorry the {value} metrics is not imlemented"
        
    def _psnr(self, original, distorted):
        mse = torch.mean((original - distorted) ** 2, dim=(1,2,3))
        max_pixel = torch.max(original)
        return 10 * torch.log10((max_pixel ** 2) / mse)
    
    def _ssim(self, original, distorted):
        original = original.permute(0, 2, 3, 1).cpu().detach().numpy()  # Convert to numpy
        distorted = distorted.permute(0, 2, 3, 1).cpu().detach().numpy()  # Convert to numpy
        ssim_values = []
        for i in range(original.shape[0]):
            data_range = max(np.max(original[i]) , np.max(distorted[i])) - min(np.min(original[i]) , np.min(distorted[i]))
            ssim_values.append(ssim(original[i], distorted[i], multichannel=True, channel_axis = 2, data_range = data_range ))
        return torch.tensor(ssim_values)
    
    def _mse(self, original, distorted):
        mse = torch.mean((original - distorted) ** 2)
        return mse
    
    def result(self, original, distorted):
        psnr_value = self._psnr(original, distorted).mean()
        ssim_value = self._ssim(original, distorted).mean()
        mse_value = self._mse(original, distorted)
        
        return {
            'PSNR': psnr_value.item(),
            'SSIM': ssim_value.item(),
            'MSE': mse_value.item()
        }

# def test():
#     config = get_config('config.json')
#     met = Metrics(config)
#     input_tensor = torch.randn(4,4,256,56)
#     distorted_tensor = torch.randn(4,4,256,56)
#     result = met.result(input_tensor, distorted_tensor)
    
#     print(result)
    
# test()