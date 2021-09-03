
from . import biggan, densenet, resnet18, vgg_face
import os

def get_model(model_name, exp_dict, data_path=None):
    if model_name == "biggan":
        model = biggan.VAE(exp_dict=exp_dict)
    elif model_name == "densenet":
        model = densenet.DenseNet121()
    elif model_name == "resnet18":
        model = resnet18.ResNet18()
    elif model_name == "vgg_face":
        weights_path = os.path.join(data_path, 'pretrained_models/resnet50_128.pth')
        model = vgg_face.resnet50_128(exp_dict=exp_dict, weights_path=weights_path)

    return model
