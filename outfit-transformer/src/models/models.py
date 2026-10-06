import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

timm_model_names = timm.list_models(pretrained=True)


model_info = {'resnet18':{'num_features': 512, 'torch_hub':'pytorch/vision:1.10.0', 'last_layer':'fc', 'dropblock_position':7, 'last_pool':'avg_pool'},
                'resnet32':{'num_features':512, 'torch_hub':'pytorch/vision:1.10.0', 'last_layer':'fc', 'dropblock_position':7, 'last_pool':'avg_pool'},
                'resnet50':{'num_features':512*4, 'torch_hub':'pytorch/vision:1.10.0', 'last_layer':'fc', 'dropblock_position':7, 'last_pool':'avg_pool'},
                'resnet101':{'num_features':512*4, 'torch_hub':'pytorch/vision:1.10.0', 'last_layer':'fc', 'dropblock_position':7, 'last_pool':'avg_pool'},
                'resnet152':{'num_features':512*4, 'torch_hub':'pytorch/vision:1.10.0', 'last_layer':'fc', 'dropblock_position':7, 'last_pool':'avg_pool'},
                'resnest50':{'num_features':512*4, 'torch_hub':'zhanghang1989/ResNeSt', 'last_layer':'fc', 'last_pool':'avg_pool'},
                'resnest101':{'num_features':512*4, 'torch_hub':'zhanghang1989/ResNeSt', 'last_layer':'fc', 'last_pool':'avg_pool'},
                'mobilenetv2_100':{'num_features':1280, 'torch_hub':'rwightman/gen-efficientnet-pytorch', 'last_layer':'classifier', 'last_pool':'adaptive_avg_pool'},
                'efficientnet_b2':{'num_features':1408, 'torch_hub':'rwightman/gen-efficientnet-pytorch', 'last_layer':'classifier', 'last_pool':'adaptive_avg_pool'},
                'efficientnet_b3':{'num_features':1536, 'torch_hub':'rwightman/gen-efficientnet-pytorch', 'last_layer':'classifier', 'last_pool':'adaptive_avg_pool'},
                'swin_large_patch4_window12_384':{'num_features':1536, 'torch_hub':'timm', 'last_layer':'head', 'last_pool':'adaptive_avg_pool'},
                'tf_efficientnet_b7_ns':{'num_features':2560, 'torch_hub':'timm', 'last_layer':'classifier', 'last_pool':'adaptive_avg_pool'},
                'tf_efficientnet_b5_ns':{'num_features':2048, 'torch_hub':'timm', 'last_layer':'classifier', 'last_pool':'adaptive_avg_pool'}}
class BackModel(nn.Module):
    def __init__(self,
                 model_name,
                 embedding_size=[99]):
        super(BackModel, self).__init__()


        self.num_features = 0
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.get_backbone()
        self.make_branch()

    def forward(self, x):
        x = self.model(x)
        if model_info[self.model_name]['last_pool'] == 'avg_pool':
            f = F.avg_pool2d(x, int(x.size()[-1]), stride=1)
            f = f.view(f.size(0), -1)
        elif model_info[self.model_name]['last_pool'] == 'adaptive_avg_pool':
            f = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        pred = self.fc(f)
        return pred


    def get_backbone(self):
        if self.model_name.startswith('resnet'):
            ResNet = models.resnet.__dict__[self.model_name]
            self.model = ResNet(pretrained=True)
            if '50' in self.model_name or '101' in self.model_name or '152' in self.model_name:
                self.num_features = 512 * 4
            else:
                print('model : {}'.format(self.model_name))
                self.num_features = 512
            self.model.fc = None

        elif self.model_name in model_info:
            if model_info[self.model_name]['torch_hub']=='timm':
                self.model = timm.create_model(self.model_name, pretrained=True)
            else:
                self.model = torch.hub.load(model_info[self.model_name]['torch_hub'], self.model_name, pretrained=True)
            self.num_features = model_info[self.model_name]['num_features']
            # print('*** *** *** MODEL *** *** *** ')
            # print('model name : {} / {}'.format(self.model_name, self.model))
            # sys.exit()
            if model_info[self.model_name]['last_layer'] == 'fc':
                self.model.fc = None
            elif model_info[self.model_name]['last_layer'] == 'classifier':
                self.model.classifier = None
            elif model_info[self.model_name]['last_layer'] == 'head':
                self.model.head = None 
            
        else:
            raise ValueError(" [!] Invaild argument: %s" % self.model_name)


    def make_branch(self):
        fcs = {}
        
        modules = list(self.model.children())
        self.model = nn.Sequential(*modules[:-1])
        self.fc =  nn.Linear(self.num_features, self.embedding_size)
        
        # print('*** *** *** MODEL *** *** *** ')
        # print('model name : {} / {}'.format(self.model_name, self.model))
        # sys.exit()


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)

    net = BackModel(model_name='resnet50', embedding_size=64)
    outputs = net(x)
       
    print(outputs.size())
