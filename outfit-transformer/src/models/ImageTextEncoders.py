import torch
import torch.nn as nn
from models.models import BackModel

class ImageTextEncoder(nn.Module):
    def __init__(self, model_name, embedding_size, text_dim):
        super().__init__()

        self.text_encoder = TextBranch(feat_dim=text_dim, embedding_dim=embedding_size)

        self.image_encoder = BackModel(model_name=model_name, embedding_size=embedding_size)

    def image_forward(self, images):
        """ images: outfit items images
        """
        image_output = self.image_encoder(images)

        return image_output

    def text_forward(self, texts):
        """ texts: outfit items text descriptions
        """
        text_output = self.text_encoder(texts)

        return text_output


    def forward(self, images, texts):
        """ images: outfit items images tensor of shape [batch_size, 3, 224,224]
            texts: outfit items corresponding text description of shape [batch_size, 768] 
            outfit_sizes: list of outfit lengths of size [batch_size]

        """
        # image_output and text_output are of shape [batch_size, embedding_size]
        image_output = self.image_forward(images)
        # print(image_output.shape)
        text_output = self.text_forward(texts)
        # print(text_output.shape)

        # concatenate text embedding and image embedding to generate item embeddings of shape [batch_size, embedding_size * 2]
        item_output = torch.cat((image_output, text_output), 1)

        return item_output
        
class TextBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(feat_dim, embedding_dim)
        
    def forward(self, x):
        x = self.fc(x)
        
        # L2 normalize each feature vector
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        # x = x / norm.expand_as(x) # python 2
        x = x / torch.unsqueeze(norm,1).expand_as(x)
        return x
