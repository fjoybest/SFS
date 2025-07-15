import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 5, 1, 2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size,affine=False))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size,affine=False),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, out_channels, 3, padding=1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)  # 8,768,28,28
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return x

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce_low = nn.Conv2d(768, 384, 1)
        self.reduce_high = nn.Conv2d(768, 384, 1)

        self.conv1 = ConvBNR(768, 384, 3)
        self.bn1 = nn.BatchNorm2d(384)
        self.relu = nn.ReLU()
        self.conv2 = ConvBNR(384, 192, 3)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 1, 1)


    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce_low(x1)
        x4 = self.reduce_high(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out
class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):

        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = wei.squeeze(-1)
        wei = wei.transpose(-1, -2).contiguous()
        wei = self.conv1d(wei)
        wei = wei.transpose(-1, -2).contiguous()
        wei = wei.unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x



# High Feature Exploring
class HFEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bor_explore = EAM()
        self.fusion = EFM(dim)
    def forward(self, high_fre_feat ,output_feat):
        bor_feat = self.bor_explore(high_fre_feat ,output_feat)
        fusion_feat = self.fusion(output_feat, bor_feat)

        return fusion_feat

class WSSSNetwork(nn.Module):
    def __init__(self, encoder, decoder, num_classes=20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_pool2d = nn.AdaptiveMaxPool2d((1, 1))
        _dim = self.encoder.embed_dim
        self.top_classifier = nn.Conv2d(_dim, num_classes - 1, kernel_size=1, bias=False)
        self.aux_classifier = nn.Conv2d(_dim, num_classes - 1, kernel_size=1, bias=False)

        self.reconstruction = GeneratorUNet(_dim, 3) ### USR
        self.hfem = HFEM(_dim)  ### BPM

    def get_param_groups(self):
        param_groups = [[], [], [], []]

        skip_list = self.encoder.no_weight_decay()
        for name, param in self.encoder.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                param_groups[0].append(param)  # no weight decay
            else:
                param_groups[1].append(param)

        for name, param in self.decoder.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[2].append(param)  # no weight decay
            else:
                param_groups[3].append(param)
        for name, param in self.reconstruction.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[2].append(param)  # no weight decay
            else:
                param_groups[3].append(param)
        for name, param in self.hfem.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[2].append(param)  # no weight decay
            else:
                param_groups[3].append(param)

        param_groups[3].append(self.top_classifier.weight)
        param_groups[3].append(self.aux_classifier.weight)

        return param_groups

    def to_2D(self, x, h, w):
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], h, w)
        return x

    def forward(self, x, mask=None, cam_only=False, istrain=False):
        multi_views = True if isinstance(x, (tuple, list)) else False
        if multi_views:
            x1 = torch.cat(x[:2], dim=0)
            x2 = torch.cat(x[2:], dim=0) if len(x[2:]) > 0 else None
        else:
            x1 = x
            x2 = None
        H, W = x1.shape[-2:]

        # encoder + projector
        top_enc_out, aux_enc_out, att_list, low_layer_feat = self.encoder(x1)
        top_enc_out = top_enc_out[:, 1:, :]
        aux_enc_out = aux_enc_out[:, 1:, :]
        low_layer_feat = low_layer_feat[:, 1:]

        # class attention
        att = torch.stack(att_list, dim=1)
        att_all = att
        l = att_all.shape[1]

        att = torch.mean(att, dim=1)
        att = att[:,0,1:]
        ps = self.encoder.patch_size
        B = top_enc_out.shape[0]
        h = H // ps
        w = W // ps
        att = att.view(B, h, w)
        # att_all = att_all[:,:,0,1:]
        # att_all = att_all.view(B,l,h,w)

        if multi_views:
            high_layer_feat = top_enc_out.chunk(2)[1]
            re_out = top_enc_out.chunk(2)[0]
            top_enc_out = top_enc_out.chunk(2)[1]
            aux_enc_out = aux_enc_out.chunk(2)[1]
            low_layer_feat = low_layer_feat.chunk(2)[1]
            high_layer_feat = self.to_2D(high_layer_feat, h, w)
            low_layer_feat = self.to_2D(low_layer_feat, h, w)
            re_out = self.to_2D(re_out, h, w)
            high_sq = self.hfem(high_layer_feat, low_layer_feat)
            att = att.chunk(2)[1]
            re_img = self.reconstruction(re_out)

        top_enc_out = self.to_2D(top_enc_out, h, w)
        aux_enc_out = self.to_2D(aux_enc_out, h, w)
        top_fmap = top_enc_out
        aux_fmap = aux_enc_out

        with torch.no_grad():
            top_cam = F.relu(F.conv2d(top_fmap, self.top_classifier.weight))
            aux_cam = F.relu(F.conv2d(aux_fmap, self.aux_classifier.weight))
        if cam_only:
            return top_cam, aux_cam

        # segmentation
        seg_out = self.decoder(top_enc_out)

        # classification
        top_cls_out = self.max_pool2d(top_enc_out)
        top_cls_out = self.top_classifier(top_cls_out).flatten(1)

        aux_cls_out = self.max_pool2d(aux_enc_out)
        aux_cls_out = self.aux_classifier(aux_cls_out).flatten(1)
        if istrain:
            return {
                "top_cls_out": top_cls_out, #[8,20]
                "aux_cls_out": aux_cls_out, #[8,768,1,1]
                "seg_out": seg_out, #[8,21,28,28]
                "top_cam": top_cam, #[8,21,28,28]
                "aux_cam": aux_cam, #[8,21,28,28]
                "top_fmap": top_fmap, #[8,768,28,28]
                #"aux_fmap": aux_fmap, #[8,768,28,28]
                "high_sq": high_sq,
                "att": att, #[8,28,28]
                #"att_all": att_all #[8,12,28,28] only view2
                "re_img":re_img
            }
        else:
            return {
                "top_cls_out": top_cls_out,  # [8,20]
                "aux_cls_out": aux_cls_out,  # [8,768,1,1]
                "seg_out": seg_out,  # [8,21,28,28]
                "top_cam": top_cam,  # [8,21,28,28]
                "aux_cam": aux_cam,  # [8,21,28,28]
                "top_fmap": top_fmap,  # [8,768,28,28]
            }


def build_model(args, pretrained=False):
    from .decoder import LargeFOV
    from .projector import DINOHead
    from .vit import vit_base_patch16_224

    encoder = vit_base_patch16_224(
        pretrained=pretrained,
        img_size=args.input_size,
        cls_depth=args.cls_depth,
        drop_path_rate=args.drop_path_rate,
    )
    decoder = LargeFOV(encoder.embed_dim, args.num_classes)
    projector = DINOHead(encoder.embed_dim, args.out_dim)

    model = WSSSNetwork(encoder=encoder, decoder=decoder, num_classes=args.num_classes)
    return model
