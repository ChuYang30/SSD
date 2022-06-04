import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from .res50_backbone import resnet50
from .utils import dboxes300_coco, Encoder, PostProcess


class Backbone(nn.Module):   #构建backbone，在resnet50的基础上进行修改
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]  #对应每一个预测特征层的channel（共六个）

        if pretrain_path is not None:   #是否有传入预训练模型参数的路径
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])   #构造特征提取部分，conv-1到conv-4一系列层结构，提取前7个子模块，索引从0到6，保留到layer3

        conv4_block1 = self.feature_extractor[-1][0]   #对conv4_block1进行定位，并找到需要修改哪个参数，对步距进行修改

        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)    #将3x3卷积层的步距修改为1
        conv4_block1.downsample[0].stride = (1, 1)   #将捷径分支的步距修改为1

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:       #对backbone进行相关判断
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)   #利用该函数构建backbone后面一系列的添加层，参数为前面定义的6个预测特征层的channel数
        self.num_defaults = [4, 6, 6, 6, 4, 4]   #每个特征层上的每个sale当中生成的defaultbox的个数
        location_extractors = []           #定义预测器，1.box的回归预测参数，2.预测分数
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))   #位置回归参数（4个）
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))   #目标分数预测

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)    #损失
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)   #后处理

    def _build_additional_features(self, input_size):    #input_size——六个预测特征层的channel
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:
        :return:
        """
        additional_blocks = []   #定义一个空列表，最终会生成五个额外的层结构，用于获得后面的特征预测层
        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]    #对应5个额外添加层结构中第一个卷积层的channel
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []     #位置参数
        confs = []    #置信度参数
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))   #通过view方法调整数据顺序
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()   #在dimension=2的维度上进行拼接，contiguous（）方法可以将数据调整为连续存储的格式
        return locs, confs

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)   #生成特征矩阵，对应网络中conv-4结构中的特征层的输出（38x38x1024）

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]  存储每一个预测特征层的列表
        detection_features.append(x)    #一共有6层预测特征层，其中conv-4的输出为第一层预测特征层
        for layer in self.additional_blocks:    #遍历构建的5个额外的预测特征层结构
            x = layer(x)                #将上一层的输出输入到当前层层结构，得到当前层的输出预测特征层结构
            detection_features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)   #获取所有预测特征层上预测的location回归参数以及cofidence参数

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            # print(bboxes_out.is_contiguous())
            labels_out = targets['labels']
            # print(labels_out.is_contiguous())

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)   #如果训练模式，会进行损失的计算
            return {"total_losses": loss}

        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(locs, confs)  #如果是非训练模式，则会预测结果进行后处理得到最终的结果
        return results


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10  scale_xy初始值为0.1
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5   scale_wh初始值为0.2

        self.location_loss = nn.SmoothL1Loss(reduction='none')   #定位损失
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),    #将xywh转换为pytorch中的参数，transpose用于调整变量的存储方式
                                   requires_grad=False)                               #unspueeze用于增加一个新的维度

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')   #定义分类loss

    def _location_vec(self, loc):
        # type: (Tensor) -> Tensor
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc: anchor匹配到的对应GTBOX Nx4x8732
        :return:
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):    #定义计算损失的正向传播过程
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732    ploc——预测的location回归参数，plabel——预测的标签值
                predicted location and labels           gloc——预处理过程中所得到的每一个anchor所匹配到的gtbox对应的gt坐标，glabel——真实标签

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        # 获取正样本的mask  Tensor: [N, 8732]
        mask = torch.gt(glabel, 0)  # (gt: >)   找到所有匹配到gt box的default box，匹配到则说明为正样本
        # mask1 = torch.nonzero(glabel)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]   这一步当中既有正样本，又有负样本
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]              通过与mask相乘，使结果中只有正样本的定位损失

        # hard negative mining Tenosr: [N, 8732]
        con = self.confidence_loss(plabel, glabel)   #分类损失，预测标签和default box匹配到的gt box，若没有匹配到则为背景

        # positive mask will never selected   分别获取正负样本，正样本在初始阶段已经得到
        # 获取负样本（选择分类损失较大的default box，负样本与正样本的比例为3:1）
        con_neg = con.clone()
        con_neg[mask] = 0.0   #将对应正样本的损失全部设置为0.mask为正样本的位置，则剩下的全部为负样本的损失
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)   #降序（索引）
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙   升序排序（索引）

        # number of negative three times positive  用于计算负样本的个数
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)  #限制负样本的数量
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss，最终的分类损失
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss  #总损失，定位损失和分类损失相加得到
        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret

