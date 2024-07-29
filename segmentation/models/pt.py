import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation

class MiniPointNet(nn.Module):
    def __init__(self):
        super(MiniPointNet, self).__init__()
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


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


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
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
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
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 64
        self.num_group = 64
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.vector_emd = MiniPointNet()

        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

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
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

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

    def forward(self, pts, cls_label):
        pts = pts.transpose(-1, -2)
        B, N, C = pts.shape
        batch_center = torch.mean(pts, dim=1)
        pts_temp = pts - batch_center.unsqueeze(1)
        vectors = self.vector_emd(pts_temp)
        vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
        pts_ccd, pts_yuan = self.convert_to_custom_cylindrical_coordinates_and_sort(pts, vectors)
        vectors = vectors.unsqueeze(1).repeat(1, self.num_group, 1)
        neighborhood_ccd = self.extract_points(pts_ccd, self.num_group * self.group_size, self.group_size,
                                               int(N / self.num_group)).view(B, self.num_group, self.group_size, 3)
        neighborhood_yuan = self.extract_points(pts_yuan, self.num_group * self.group_size, self.group_size,
                                                int(N / self.num_group)).view(B, self.num_group, self.group_size, 3)
        center_ccd = torch.mean(neighborhood_ccd, dim=2)
        center_yuan = torch.mean(neighborhood_yuan, dim=2)
        center = torch.cat((vectors, center_ccd), dim=-1)
        neighborhood = neighborhood_yuan - center_yuan.unsqueeze(2)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)
        x = group_input_tokens
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) #1152*2 + 64
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center_yuan.transpose(-1, -2), pts.transpose(-1, -2), x)
        x = torch.cat((f_level_0,x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss