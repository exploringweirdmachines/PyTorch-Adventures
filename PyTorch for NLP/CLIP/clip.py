import torch
import torch.nn as nn
import math
from transformers import ResNetModel, RobertaModel

class CLIP(nn.Module):
    def __init__(self, tgt_embed_dim=512):
        super(CLIP, self).__init__()

        ### Vision Encoder ###
        self.vision_encoder = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.vision_proj = nn.Linear(2048, tgt_embed_dim)

        ### Text Encoder ###
        self.text_encoder = RobertaModel.from_pretrained("distilbert/distilroberta-base")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, tgt_embed_dim)

        ### Learnable Temperature Parameter ###
        self.tau = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def compute_image_embeds(self, pixel_values):
        # ResNet-50 outputs logits; get features from pooler_output
        image_features = self.vision_encoder(pixel_values)["pooler_output"].squeeze(-1).squeeze(-1)
        image_embeds = self.vision_proj(image_features)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

    def compute_text_embeds(self, input_ids, attention_mask):

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_embeds = self.text_proj(text_outputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds
    
    def forward(self, pixel_values, input_ids, text_attention_mask):
        image_embeds = self.compute_image_embeds(pixel_values)
        text_embeds = self.compute_text_embeds(input_ids, text_attention_mask)
        return image_embeds, text_embeds, self.tau
    
model = CLIP()
rand = torch.randn(2,3,224,224)
model.compute_image_embeds(rand)