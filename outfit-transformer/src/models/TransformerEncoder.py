import torch
import torch.nn as nn
from labml import tracker
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list

# Some labml versions don't expose tracker.debug; add a no-op to avoid crashes.
if not hasattr(tracker, 'debug'):
        tracker.debug = lambda *_, **__: None

class TransformerLayer(nn.Module):
        """
        

        Parameters:

        d_model: the token embedding size
        self_attn: the self attention module
        feed_forward: the feed forward module
        dropout_prob: the probability of dropping out after self attention and FFN
        """
        def __init__(self,
                     d_model: int,
                     self_attn: MultiHeadAttention,
                     feed_forward: FeedForward,
                     dropout_prob: float):
                super().__init__()
                self.size = d_model
                self.self_attn = self_attn
                self.feed_forward = feed_forward
                self.dropout = nn.Dropout(dropout_prob)
                self.norm_self_attn = nn.LayerNorm([d_model])
                self.norm_ff = nn.LayerNorm([d_model])

                # whether to save input to the feed forward layer
                self.is_save_ff_input = False

        def forward(self,
                    x: torch.Tensor,
                    mask: torch.Tensor):
        

                # Normalize the vectors before doing self attention
                z = self.norm_self_attn(x)
                # print('query shape {}'.format(z.shape))
                # print('query : {}'.format(z))
                # Run through self attention, i.e. keys and values are from self
                self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)

                # print('output of self attn of shape {} : {}'.format(self_attn.shape, self_attn))
                # Add the self attention results
                x = x + self.dropout(self_attn)

                # Normalize for feed-forward
                z = self.norm_ff(x)
                # print('output of norm shape: {}'.format(z.shape))
                # Save the input to the feed forward if specified
                if self.is_save_ff_input:
                        self.ff_input = z.clone()
                
                
                # Pass through the feed-forward network
                ff = self.feed_forward(z)
                # print('output of ffn shape: {}'.format(ff.shape))
                # Add the feed-forward results back
                x = x + self.dropout(ff)
                
                return x
           
class TransformerEncoder(nn.Module):

        def __init__(self, layer: TransformerLayer, n_layers: int):
                super().__init__()

                self.layer = layer

                # Make copies of the transformer layer
                self.layers = clone_module_list(layer, n_layers)

                # Final normalization layer
                self.norm = nn.LayerNorm([layer.size])

        def forward(self, x: torch.Tensor, mask: torch.Tensor):

                # Run through each transformer layer
                i = 0
                for layer in self.layers:
                        x = layer(x=x, mask=mask)
                        # print('layer {} x = {}'.format(i, x))
                # Get the transformer output of the outfit token (which is the first in the sequence)
                x = x[0]
                # print('x token only output shape: {}'.format(x.shape))
                # Finally normalize the vectors
                x = self.norm(x)
                return x


            

            

