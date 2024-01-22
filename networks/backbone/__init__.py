from networks.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm,nInputChannels,pretrained):
    if backbone == 'resnet18':
        return resnet.ResNet18(output_stride, BatchNorm,nInputChannels=nInputChannels,pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm,nInputChannels=nInputChannels,pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm,nInputChannels=nInputChannels,pretrained=pretrained)         
    else:
        raise NotImplementedError
