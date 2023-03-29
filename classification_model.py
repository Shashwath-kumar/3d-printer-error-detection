import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import MobileViTForImageClassification, MobileViTConfig
from transformers.models.mobilevit.modeling_mobilevit import MobileViTConvLayer

class BEiTImageClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, in_channels=4):
        super(BEiTImageClassifier, self).__init__()

        self.beit = timm.create_model("beit_base_patch16_224", pretrained=pretrained)

        # Modify the input embedding layer to accept 4-channel input
        original_embedding = self.beit.patch_embed.proj
        new_embedding = nn.Conv2d(in_channels, original_embedding.out_channels,
                                  kernel_size=original_embedding.kernel_size,
                                  stride=original_embedding.stride, padding=original_embedding.padding)

        # Copy the original weights and initialize the weights for the additional channel
        with torch.no_grad():
            new_embedding.weight[:, :3] = original_embedding.weight
            new_embedding.weight[:, 3] = original_embedding.weight.mean(dim=1)

        self.beit.patch_embed.proj = new_embedding

        # Modify the final classification layer
        self.beit.head = nn.Linear(self.beit.head.in_features, num_classes)

    def forward(self, x):
        return self.beit(x).squeeze()
    
class MobileViTImageClassifier(nn.Module):
    def __init__(self, num_classes=1, in_channels=4):
        super(MobileViTImageClassifier, self).__init__()

        self.mobilevit = MobileViTForImageClassification.from_pretrained('apple/mobilevit-small')

        original_embedding : nn.Conv2d = self.mobilevit.mobilevit.conv_stem.convolution

        new_embedding = nn.Conv2d(in_channels, original_embedding.out_channels,
                                  kernel_size=original_embedding.kernel_size,
                                  stride= original_embedding.stride,
                                  padding= original_embedding.padding,
                                  bias = original_embedding.bias,
                                  groups= original_embedding.groups,
                                  dilation= original_embedding.dilation,
                                  padding_mode= original_embedding.padding_mode)

        # Copy the original weights and initialize the weights for the additional channel
        with torch.no_grad():
            new_embedding.weight[:, :3] = original_embedding.weight
            new_embedding.weight[:, 3] = original_embedding.weight.mean(dim=1)

        self.mobilevit.mobilevit.conv_stem.convolution = new_embedding

        self.mobilevit.classifier = nn.Linear(self.mobilevit.classifier.in_features, num_classes)

    def forward(self, x):
        return self.mobilevit(x).logits.squeeze()
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)  # Reshape and transpose for MultiheadAttention
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = attn_output.permute(1, 2, 0)  # Transpose back to the original shape
        return attn_output.view(x.size(1), -1)  # Flatten the output

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=1, embed_dim=512, num_heads=4, input_channels = 4, dropout_rate = 0.5):
        super(ImageClassificationModel, self).__init__()
        
        # Load the pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self. dropout1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.self_attention(x)
        x = self.dropout1(x)
        x = self.fc(x)

        return x.squeeze()

if __name__ == '__main__':
    # Create the model
    model = ImageClassificationModel(num_classes=1)
