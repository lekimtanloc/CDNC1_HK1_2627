import sys
import torch
from torch import nn
import torch.nn.functional as F
from models import OutfitTransformer

class CIRMLP(nn.Module):
    """
    A two layer MLP head that generates the target item embedding

    Parameters:

    d_model: the token embedding size
    n_hidden: the size of the hidden layer
    """
    def __init__(self, d_model: int, n_hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, d_model)


    def forward(self, x: torch.Tensor):

        # x is the transformer encoding for the outfit token
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

class CIROutfitTransformer(nn.Module):
    """
    Combine the image and text embeddings, transformer encoder and the classification head.


    Parameters:

    cp_pretrained_transformer: a pretrained OutfitTransformer on the CP (Compatibility prediction) task
    mlp: a MLP that generates the target item embedding 
    """
    def __init__(self, 
                cp_pretrained_transformer: OutfitTransformer,
                mlp: CIRMLP):
        super().__init__()
        self.cp_pretrained_transformer = cp_pretrained_transformer
        self.mlp = mlp

    def forward(self, 
                images: torch.Tensor, 
                texts: torch.Tensor, 
                outfit_lengths: list[int],
                bs: int):
        """
            images: outfit items images tensor of shape [batch_size, 3, 224,224]
            texts: outfit items corresponding text description of shape [batch_size, text_feature_dim]
            outfit_lengths: a list of corresponding outfits initial lengths, used to compute the MHA masks
           
        """

        x = self.cp_pretrained_transformer.image_text_encoders(images, texts)
        _, d_model = x.shape 
        # print(x.shape)
        # print(x)
        x = torch.permute(x.view(bs, max(outfit_lengths), d_model), (1, 0, 2))
        # x is the input embeddings of shape [outfit_len, batch_size, d_model]


        # mha_maks: attention masks of shape [outfit_len, outfit_len, batch_size], 
        # mask[i, j, b] indicates whether for batch b , query at position i has access to key-value at position j
        mha_mask = make_sample_mask(outfit_lengths)
        # keep mask on the same device as inputs (CPU, CUDA, or MPS)
        mha_mask = mha_mask.to(dtype=torch.int64, device=images.device)
        # print(mha_mask[:,:,0])
        # print(mha_mask[:,:,1])


        # Pass through transformer layers with attention masking, the result is the global outfit representation
        x = self.cp_pretrained_transformer.transformer_encoder(x, mask=mha_mask)
        # print('output of transformer shape: {}'.format(x.shape))
        # MLP, to get the target item embedding
        x = self.mlp(x)
        # print('output of cls shape: {}'.format(x.shape))
        return x


def make_sample_mask(outfit_len_list: list[int]):
    max_len = max(outfit_len_list)
    sample_masks_list = []
    for ln in outfit_len_list:
        sample_mask = torch.ones(ln, ln)
        for i in range(max_len-ln):
            sample_mask = F.pad(sample_mask, (0,1,0,1))
            # set one value to 1 in the padded sample in mask to avoid nan
            sample_mask[ln+i,ln+i-1]=1
        # print(ln)
        # print(sample_mask)
        sample_masks_list.append(sample_mask)
    return torch.stack(sample_masks_list, 2)    
    

    
