import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import open3d as o3d


class Encoder(nn.Module):  
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Predicted(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.mlp1 = Mlp(in_features=1024 * 3, hidden_features=512 * 3, out_features=256 * 3)
        self.mlp2 = Mlp(in_features=128 * 3, hidden_features=64 * 3, out_features=15 * 3)

    def forward(self, xyz):
        xyz = self.mlp1(xyz.view(64, 1024 * 3))
        xyz = self.mlp2(xyz).view(64, 15, 3)
        return xyz


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        # self.pos_embed = PositionEmbeddingCoordsSine(6, self.trans_dim, 2.0)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 6)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class MiniPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        # x shape: [batch_size, num_points, 3]
        x = x.transpose(2, 1)  # 转换为 [batch_size, 3, num_points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]  # 全局最大池化，形状为 [batch_size, 1024]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        # self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        self.vector_emd = MiniPointNet()
        self.build_loss_func()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def convert_to_custom_cylindrical_coordinates_and_sort(self, point_cloud, unit_vectors):
        """
        将点云的三维坐标转换为相对于指定向量的柱坐标 (rho', phi', z') 并按照 z' 排序

        参数:
            point_cloud (torch.Tensor): 输入点云张量，形状为 [batch_size, num_points, 3]
            vectors (torch.Tensor): 指定向量，形状为 [batch_size, 3]

        返回:
            sorted_cylindrical_coords (torch.Tensor): 按 z' 排序的自定义柱坐标张量，形状为 [batch_size, num_points, 3]
        """
        device = point_cloud.device
        batch_size = point_cloud.shape[0]

        # # 归一化指定向量
        # unit_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

        # 将点云投影到单位向量上，得到新的高度 z'
        z_prime = torch.einsum('bik,bk->bi', point_cloud, unit_vectors)

        # 计算点到单位向量直线的垂直距离
        projection = torch.einsum('bi,bk->bik', z_prime, unit_vectors)
        perpendicular_vector = point_cloud - projection
        rho_prime = torch.norm(perpendicular_vector, dim=2)

        # 计算新的方位角 phi'
        x_coords = perpendicular_vector[..., 0]
        y_coords = perpendicular_vector[..., 1]
        phi_prime = torch.atan2(y_coords, x_coords)

        # 将柱坐标组合成一个张量
        custom_cylindrical_coords = torch.stack((rho_prime, phi_prime, z_prime), dim=-1)

        # 按照 z' 进行排序
        sorted_indices = torch.argsort(custom_cylindrical_coords[..., 2], dim=1)
        sorted_cylindrical_coords = torch.gather(custom_cylindrical_coords, 1,
                                                 sorted_indices.unsqueeze(-1).expand(-1, -1, 3))
        sorted_pointcloud = torch.gather(point_cloud, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))

        return sorted_cylindrical_coords, sorted_pointcloud

    def extract_points(self, ptss, num_points=2048, window_size=128, step=64):
        """
        Extracts points from a batch of point clouds using a sliding window approach,
        ensuring all tensors are operated on the same device to prevent errors.

        Parameters:
            ptss (torch.Tensor): Input tensor of size (batch_size, n_points, 3) representing
                                 point clouds where each batch contains n_points in 3D.
            num_points (int): Number of points to extract from each point cloud in the batch.
            window_size (int): Number of points to include in each window.
            step (int): Step size for the sliding window.

        Returns:
            torch.Tensor: The tensor of extracted points of size (batch_size, num_points, 3).
        """
        # 获取输入张量的设备类型, 以确保所有张量操作都在同一个设备上
        device = ptss.device
        # 创建一个空的结果张量，并确保它在与输入张量相同的设备上
        result_points = torch.empty((ptss.size(0), 0, 3), device=device)

        # 初始化滑动窗口的起始索引
        start = 0
        start_idx = 0
        # 循环直到每个批次的结果张量中的点数达到所需的点数
        while result_points.size(1) < num_points:
            # 计算滑动窗口的结束索引
            end_idx = start_idx + window_size
            # 如果结束索引超出输入张量的点数，需要从头开始取点
            if end_idx > ptss.size(1):
                end_idx -= ptss.size(1)
                # 由于结束索引超出范围，需要分两部分从输入张量中取点
                points = torch.cat((ptss[:, start_idx:], ptss[:, :end_idx]), dim=1)
            else:
                # 如果结束索引在范围内，直接从输入张量中取点
                points = ptss[:, start_idx:end_idx]
            # 将取出的点添加到结果张量中
            result_points = torch.cat((result_points, points), dim=1)
            # 更新滑动窗口的起始索引
            start_idx += step
            # 如果更新后的起始索引超出输入张量的点数，从头开始
            if start_idx >= ptss.size(1):
                start_idx -= ptss.size(1)
            start = start + 1
        return result_points

    def forward(self, pts):
        bs, N, _ = pts.shape
        batch_center = torch.mean(pts, dim=1)
        pts_temp = pts - batch_center.unsqueeze(1)
        vectors = self.vector_emd(pts_temp)
        vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
        pts_ccd, pts_yuan = self.convert_to_custom_cylindrical_coordinates_and_sort(pts, vectors)
        vectors = vectors.unsqueeze(1).repeat(1, self.num_group, 1)
        neighborhood_ccd = self.extract_points(pts_ccd, self.num_group * self.group_size, self.group_size,
                                               int(N / self.num_group)).view(bs, self.num_group, self.group_size, 3)
        neighborhood_yuan = self.extract_points(pts_yuan, self.num_group * self.group_size, self.group_size,
                                                int(N / self.num_group)).view(bs, self.num_group, self.group_size, 3)
        center_ccd = torch.mean(neighborhood_ccd, dim=2)
        center_yuan = torch.mean(neighborhood_yuan, dim=2)
        center = torch.cat((vectors, center_ccd), dim=-1)
        neighborhood = neighborhood_yuan - center_yuan.unsqueeze(2)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


#pretrain
@MODELS.register_module()
class PointCut(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        # self.decoder_pos_embed = PositionEmbeddingCoordsSine(6, self.trans_dim, 2.0)
        self.vector_emd = MiniPointNet()
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        print_log(f'[PointCut] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCut')
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
    def convert_to_custom_cylindrical_coordinates_and_sort(self, point_cloud, unit_vectors):
        """
        将点云的三维坐标转换为相对于指定向量的柱坐标 (rho', phi', z') 并按照 z' 排序

        参数:
            point_cloud (torch.Tensor): 输入点云张量，形状为 [batch_size, num_points, 3]
            vectors (torch.Tensor): 指定向量，形状为 [batch_size, 3]

        返回:
            sorted_cylindrical_coords (torch.Tensor): 按 z' 排序的自定义柱坐标张量，形状为 [batch_size, num_points, 3]
        """

        # # 归一化指定向量
        # unit_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

        # 将点云投影到单位向量上，得到新的高度 z'
        z_prime = torch.einsum('bik,bk->bi', point_cloud, unit_vectors)

        # 计算点到单位向量直线的垂直距离
        projection = torch.einsum('bi,bk->bik', z_prime, unit_vectors)
        perpendicular_vector = point_cloud - projection
        rho_prime = torch.norm(perpendicular_vector, dim=2)

        # 计算新的方位角 phi'
        x_coords = perpendicular_vector[..., 0]
        y_coords = perpendicular_vector[..., 1]
        phi_prime = torch.atan2(y_coords, x_coords)

        # 将柱坐标组合成一个张量
        custom_cylindrical_coords = torch.stack((rho_prime, phi_prime, z_prime), dim=-1)

        # 按照 z' 进行排序
        sorted_indices = torch.argsort(custom_cylindrical_coords[..., 2], dim=1)
        sorted_cylindrical_coords = torch.gather(custom_cylindrical_coords, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))
        sorted_pointcloud = torch.gather(point_cloud, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))

        return sorted_cylindrical_coords, sorted_pointcloud

    def extract_points(self, ptss, num_points=2048, window_size=128, step=64):
        """
        Extracts points from a batch of point clouds using a sliding window approach,
        ensuring all tensors are operated on the same device to prevent errors.

        Parameters:
            ptss (torch.Tensor): Input tensor of size (batch_size, n_points, 3) representing
                                 point clouds where each batch contains n_points in 3D.
            num_points (int): Number of points to extract from each point cloud in the batch.
            window_size (int): Number of points to include in each window.
            step (int): Step size for the sliding window.

        Returns:
            torch.Tensor: The tensor of extracted points of size (batch_size, num_points, 3).
        """
        # 获取输入张量的设备类型, 以确保所有张量操作都在同一个设备上
        device = ptss.device
        # 创建一个空的结果张量，并确保它在与输入张量相同的设备上
        result_points = torch.empty((ptss.size(0), 0, 3), device=device)

        # 初始化滑动窗口的起始索引
        start = 0
        start_idx = 0
        # 循环直到每个批次的结果张量中的点数达到所需的点数
        while result_points.size(1) < num_points:
            # 计算滑动窗口的结束索引
            end_idx = start_idx + window_size
            # 如果结束索引超出输入张量的点数，需要从头开始取点
            if end_idx > ptss.size(1):
                end_idx -= ptss.size(1)
                # 由于结束索引超出范围，需要分两部分从输入张量中取点
                points = torch.cat((ptss[:, start_idx:], ptss[:, :end_idx]), dim=1)
            else:
                # 如果结束索引在范围内，直接从输入张量中取点
                points = ptss[:, start_idx:end_idx]
            # 将取出的点添加到结果张量中
            result_points = torch.cat((result_points, points), dim=1)
            # 更新滑动窗口的起始索引
            start_idx += step
            # 如果更新后的起始索引超出输入张量的点数，从头开始
            if start_idx >= ptss.size(1):
                start_idx -= ptss.size(1)
            start = start + 1
        return result_points

    def forward(self, pts, vis=False, **kwargs):
        bs, N, _ = pts.shape
        batch_center = torch.mean(pts, dim=1)
        pts_temp = pts - batch_center.unsqueeze(1)
        vectors = self.vector_emd(pts_temp)
        vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
        pts_ccd, pts_yuan = self.convert_to_custom_cylindrical_coordinates_and_sort(pts, vectors)
        vectors = vectors.unsqueeze(1).repeat(1, self.num_group, 1)
        neighborhood_ccd = self.extract_points(pts_ccd, self.num_group * self.group_size, self.group_size, int(N / self.num_group)).view(bs, self.num_group, self.group_size, 3)
        neighborhood_yuan = self.extract_points(pts_yuan, self.num_group * self.group_size, self.group_size, int(N / self.num_group)).view(bs, self.num_group, self.group_size, 3)
        center_ccd = torch.mean(neighborhood_ccd, dim=2)
        center_yuan = torch.mean(neighborhood_yuan, dim=2)
        center = torch.cat((vectors, center_ccd), dim=-1)
        neighborhood = neighborhood_yuan - center_yuan.unsqueeze(2)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_points = neighborhood_yuan[mask].reshape(B * M, -1, 3)
        loss2 = self.loss_func_p2(rebuild_points, gt_points)
        return loss2