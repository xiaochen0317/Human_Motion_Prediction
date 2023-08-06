# 导入所需的PyTorch库和模块
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn.parameter import Parameter  # 导入PyTorch中的参数
from torch.autograd import Variable  # 导入PyTorch中的变量
import torch.nn.functional as F  # 导入PyTorch中的函数操作


# 定义一个名为HEATlayer的类，继承自nn.Module，用于执行图中节点特征的聚合操作。
class HEATlayer(nn.Module):
    # 构造函数，初始化类的各种属性和超参数
    def __init__(self, in_channels_node=32, in_channels_edge_attr=2, in_channels_edge_type=2, edge_attr_emb_size=32,
                 edge_type_emb_size=32, node_emb_size=32, out_channels=32, heads=3, concat=True):
        # 调用父类nn.Module的构造函数，初始化神经网络模块
        super(HEATlayer, self).__init__()

        # 设置类的各种属性，用于嵌入、线性变换等操作
        self.in_channels_node = in_channels_node  # 节点特征的输入通道数
        self.in_channels_edge_attr = in_channels_edge_attr  # 边属性的输入通道数
        self.in_channels_edge_type = in_channels_edge_type  # 边类型的输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.heads = heads  # 注意头数
        self.concat = concat  # 是否连接标志
        self.edge_attr_emb_size = edge_attr_emb_size  # 边属性嵌入大小
        self.edge_type_emb_size = edge_type_emb_size  # 边类型嵌入大小
        self.node_emb_size = node_emb_size  # 节点嵌入大小

        # 创建节点特征的嵌入层
        self.set_node_emb()
        # 创建边属性和边类型的嵌入层
        self.set_edge_emb()

        # 创建用于更新节点特征的线性变换层
        self.node_update_emb = nn.Linear(self.edge_attr_emb_size + self.node_emb_size, self.out_channels, bias=False)

        # 创建注意机制中的线性变换层
        self.attention_nn = nn.Linear(self.edge_attr_emb_size + self.edge_type_emb_size + 2 * self.node_emb_size,
                                      1 * self.heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)  # 创建带有负斜率的Leaky ReLU激活函数
        self.soft_max = nn.Softmax(dim=1)  # 创建softmax函数，按指定维度进行计算

    # 设置节点特征的嵌入层
    def set_node_emb(self):
        # 创建用于嵌入车辆节点特征的线性变换层
        self.veh_node_feat_emb = nn.Linear(self.in_channels_node, self.node_emb_size, bias=False)
        # 创建用于嵌入行人/自行车节点特征的线性变换层
        self.ped_node_feat_emb = nn.Linear(self.in_channels_node, self.node_emb_size, bias=False)

    # 设置边属性和边类型的嵌入层
    def set_edge_emb(self):
        # 创建用于嵌入边属性的线性变换层
        self.edge_attr_emb = nn.Linear(self.in_channels_edge_attr, self.edge_attr_emb_size, bias=False)
        # 创建用于嵌入边类型的线性变换层
        self.edge_type_emb = nn.Linear(self.in_channels_edge_type, self.edge_type_emb_size, bias=False)

    # 将节点特征嵌入到不同向量空间的函数
    def embed_nodes(self, node_features, veh_node_mask, ped_node_mask):
        # 创建一个用于存储嵌入后节点特征的张量，初始为全零
        emb_node_features = torch.zeros(node_features.shape[0], self.node_emb_size).to(ped_node_mask.device)
        # 使用车辆节点掩码嵌入车辆节点特征
        emb_node_features[veh_node_mask] = self.veh_node_feat_emb(node_features[veh_node_mask])
        # 使用行人/自行车节点掩码嵌入行人/自行车节点特征
        emb_node_features[ped_node_mask] = self.ped_node_feat_emb(node_features[ped_node_mask])
        return emb_node_features

    # 将边属性和边类型嵌入为边特征的函数
    def embed_edges(self, edge_attrs, edge_types):
        # 使用线性变换嵌入边属性，然后使用带有负斜率的Leaky ReLU激活函数
        emb_edge_attributes = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        # 使用线性变换嵌入边类型，然后使用带有负斜率的Leaky ReLU激活函数
        emb_edge_types = self.leaky_relu(self.edge_type_emb(edge_types))
        return emb_edge_attributes, emb_edge_types

    # 此函数执行模块的前向传播
    def forward(self, node_f, edge_index, edge_attr, edge_type, veh_node_mask, ped_node_mask):
        """
        Args:
            node_f ([num_node, in_channels_nodeeature])
            edge_index ([2, number_edge])
            edge_attr ([number_edge, len_edge_feature])
        """
        # 嵌入边属性和边类型
        emb_edge_attr, emb_edge_type = self.embed_edges(edge_attr, edge_type.float())

        # 将嵌入后的边属性和边类型拼接在一起，形成嵌入后的边特征
        emb_edge_f = torch.cat((emb_edge_attr, emb_edge_type), dim=1)

        # 构造一个用于存储嵌入后的边特征的稀疏张量，然后将其转换为稠密张量
        size = torch.Size([node_f.shape[0], node_f.shape[0]] + [self.edge_attr_emb_size + self.edge_type_emb_size])
        emb_edge_f = torch.sparse_coo_tensor(edge_index, emb_edge_f, size).to_dense()

        # 嵌入节点特征，并扩展为与边特征相同的形状
        emb_node_f = self.leaky_relu(self.embed_nodes(node_f, veh_node_mask, ped_node_mask))
        nbrs_exp_emb_node_f = emb_node_f.unsqueeze(dim=0).repeat(node_f.shape[0], 1, 1)

        # 将嵌入后的边特征和节点特征拼接在一起，形成边邻居特征
        cat_edge_nbr_f_Weh = torch.cat((emb_edge_f, nbrs_exp_emb_node_f), dim=2)

        # 更新节点特征
        # 1. 将目标节点特征与边特征和节点邻居特征拼接在一起，并计算分数
        tar_exp_emb_node_f = emb_node_f.unsqueeze(dim=1).repeat(1, node_f.shape[0], 1)
        cat_tar_edge_nbr_f_Wheh = torch.cat((tar_exp_emb_node_f, cat_edge_nbr_f_Weh), dim=2)
        scores = self.leaky_relu(self.attention_nn(cat_tar_edge_nbr_f_Wheh))

        # 2. 根据邻接矩阵选择分数，将没有边的位置的分数设置为'-inf'或-10000
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(ped_node_mask.device),
                                      torch.Size([node_f.shape[0], node_f.shape[0]])).to_dense().to(
            ped_node_mask.device)
        adj = adj.unsqueeze(dim=2).repeat(1, 1, self.heads)
        scores_nbrs = torch.where(adj == 0.0, torch.ones_like(scores) * -10000, scores)

        # 计算注意权重
        attention_nbrs = self.soft_max(scores_nbrs).unsqueeze(dim=3)
        attention_nbrs = attention_nbrs.repeat(1, 1, 1, self.out_channels)

        # 3. 使用注意权重更新节点特征，仅考虑边属性和节点特征，不考虑边类型
        size = torch.Size([node_f.shape[0], node_f.shape[0]] + [self.edge_attr_emb_size])
        emb_edge_attr = torch.sparse_coo_tensor(edge_index, emb_edge_attr, size).to_dense()
        cat_edge_attr_nbr_feat = torch.cat((emb_edge_attr, nbrs_exp_emb_node_f), dim=2)
        cat_edge_attr_nbr_feat = cat_edge_attr_nbr_feat.unsqueeze(dim=2).repeat(1, 1, self.heads, 1)
        cat_edge_attr_nbr_feat = self.leaky_relu(self.node_update_emb(cat_edge_attr_nbr_feat))
        next_node_f = torch.sum(torch.mul(attention_nbrs, cat_edge_attr_nbr_feat), dim=1)

        # 根据concat标志，将更新后的节点特征进行展平或取平均
        if self.concat:
            next_node_f = torch.flatten(next_node_f, start_dim=1)
        else:
            next_node_f = torch.mean(next_node_f, dim=1)

        return next_node_f


# 主程序块，创建HEATlayer类的实例并设置输入通道数
if __name__ == '__main__':
    heatlayer = HEATlayer(in_channels_node=31, in_channels_edge=2)
