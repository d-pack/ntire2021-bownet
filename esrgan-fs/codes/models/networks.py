from copy import copy

import torch
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.USRGAN_arch as USRGAN_arch
import models.modules.USRGANLarge_arch as USRGANLarge_arch
import models.modules.USRGAN_Connections_arch as USRGAN_Connections_arch
import models.modules.BOWGAN_arch as BOWGAN_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = copy(opt['network_G'])
    which_model = opt_net.pop('which_model_G')
    if 'scale' in opt_net:
        scale = opt_net.pop('scale')

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(**opt_net, upscale=scale)
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(**opt_net)
    elif which_model == 'USRGAN':
        netG = USRGAN_arch.USRGAN(**opt_net)
    elif which_model == 'USRGANLarge':
        netG = USRGANLarge_arch.USRGANLarge(**opt_net)
    elif which_model == 'USRGAN_conns':
        netG = USRGAN_Connections_arch.USRGANLarge(**opt_net)
    elif which_model == 'BOWGAN':
        netG = BOWGAN_arch.BOWGAN(**opt_net)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_128_patch':
        netD = SRGAN_arch.Discriminator_VGG_128_Patch(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
