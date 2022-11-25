import numpy as np
import torch
import torchvision
from utils import *


# mnist_trainset = torchvision.datasets.MNIST(root = "datasets/", train = True, transform = None, target_transform = None, download = False)
# sample_image, _ = mnist_trainset[0]
# # sample_image.show()

# transform = torchvision.transforms.ToTensor()
# img_tensor = transform(sample_image)
# img_tensor.unsqueeze_(0)
# img_tensor = img_tensor.repeat(4, 3, 1, 1)


def build_model():
    # input_image = img_tensor
    input_image = torch.zeros([7, 3, 256, 256])

    analysis_transform = AnalysisTransform()
    hyper_analysis_transform = HyperAnalysisTransform()
    hyper_synthesis_transform = HyperSynthesisTransform()

    feature = analysis_transform(input_image)   # encoder for encoding inputs into their latent vectors
    z = hyper_analysis_transform(feature)   # hyper-latents
    compressed_z = torch.round(z)   # quantized hyper-latents
    recon_sigma = hyper_synthesis_transform(compressed_z)   # hyper-decoder for decoding the hyper-latents

    '''
    We need to replace the decoder with a diffusion model here to reconstruct the input and train the model jointly 
    to optimize the parameters all the models!
    '''
    
    compressed_feature_renorm = feature / recon_sigma
    compressed_feature_renorm = torch.round(compressed_feature_renorm)
    compressed_feature_denorm = compressed_feature_renorm * recon_sigma
    
    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("z : ", z.size())
    print("recon_sigma : ", recon_sigma.size())

    
if __name__ == '__main__':
    build_model()