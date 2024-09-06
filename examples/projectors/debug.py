import torch
import re
from open_flamingo.src.helpers import PerceiverResampler, VisionTokenizer
from einops_exts import rearrange_many
import torch.nn as nn
torch.set_printoptions(precision=6)
# from open_flamingo.src.helpers import Forward, 
DIR = '/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct'
ckpt = torch.load(DIR + '/xgenmm.projector')

# projector = PerceiverResampler(dim=1152, dim_inner=3072, depth=6, dim_head=96,heads=16,num_latents=128)
# projector.load_state_dict(ckpt, strict=True)
from torch import einsum, nn
class MyPerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        # print('latents:', latents.shape)
        # print('before ln:', latents)
        latents = self.norm_latents(latents)
        # print('after ln:', latents)
        # print(latents)
        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2) # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat((vision_attn_masks, 
                                            torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device)),
                                            dim=-1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # print('q:', q.shape, 'k:', k.shape)
        # print('q * self.scale:', q * self.scale)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        # print('q:', q.shape, 'k:', k.shape)
        # print('q', q)
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # print('sim:', sim.shape)
        # print('sim:', sim)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros((q.size(0), 1, 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
            print('vision_attn_masks:', vision_attn_masks.shape)
            vision_attn_masks = repeat(vision_attn_masks, 'b n -> b 1 1 l n', l=q.size(-2))
            print('vision_attn_masks:', vision_attn_masks.shape)
            print('attn_bias:', attn_bias.shape, 'sim:', sim.shape)
            # return q, k, vision_attn_masks
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias
            
            
        # print('remove safe softmax')
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        # print('attn:', attn.shape)
        # print('attn:', attn)
        
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)

def MyFeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(approximate='tanh'),
        nn.Linear(inner_dim, dim, bias=False),
    )



class MyModel(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=64,
        ff_mult=4,
    ):
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        print('use MyPerceiverAttention')
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MyPerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        MyFeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, up_layer_idx, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions

        # blocks
        # FIXME: extending query tokens proportional to the vision sequence length. Hard-coded as dfn5b token_len=729.
        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers[up_layer_idx]:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents
        return self.projection(self.norm(latents)) 
   
   
projector = MyModel(dim=1152, dim_inner=3072, depth=6, dim_head=96,heads=16,num_latents=128)
projector.load_state_dict(ckpt, strict=True)   

from einops import rearrange, repeat
x = torch.load('torch_input.pt')
b, T, F, v = x.shape[:4]
x = rearrange(
    x, "b T F v d -> b T (F v) d"
)  # flatten the frame and spatial dimensions

# blocks
# FIXME: extending query tokens proportional to the vision sequence length. Hard-coded as dfn5b token_len=729.
latents = projector.latents
latents = repeat(latents, "n d -> b T n d", b=b, T=T)
x.shape, latents.shape

vision_attn_masks = torch.zeros((latents.shape[0], 729), dtype=latents.dtype, device=latents.device)
vision_attn_masks[:,0] = 1.0
vision_attn_masks[:,1] = 1.0
vision_attn_masks[:,-1] = 1.0
vision_attn_masks[:,-2] = 1.0
# torch.save(vision_attn_masks, 'test_data/vision_attn_masks.pt')
# vision_attn_masks = torch.cat((vision_attn_masks, 
#                                 torch.ones((latents.shape[0], latents.shape[-2]))),
#                                 dim=-1)
vision_attn_masks.shape
latents = projector.latents
latents = repeat(latents, "n d -> b T n d", b=b, T=T)
for attn, ff in projector.layers[:7]:
    print('one layer')
    ffn_input = attn(x, latents, vision_attn_masks) + latents
    ffn_output = ff(ffn_input) + ffn_input
    latents = ffn_output
llm_input = projector.projection(projector.norm(ffn_output))
print(llm_input.shape)
print(llm_input)