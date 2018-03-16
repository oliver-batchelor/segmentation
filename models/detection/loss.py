# Originally by Alexander (Max) deGroot
# https://github.com/amdegroot/ssd.pytorch.git

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detection.box_utils import match, log_sum_exp

def check_shape(name, t, shape):
    assert t.size() == torch.Size(shape), "shape for '{}' is ({}) should be ({})".format(name, shape_str(t.size()), shape_str(shape))

def shape_str(shape):
    return "x".join(str(x) for x in shape)
    



class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_targethresh=0.5, neg_pos=3, 
                 neg_overlap=0.5, variance = [0.1, 0.2], cuda=True):
        super(MultiBoxLoss, self).__init__()
        self.cuda = cuda
        self.num_classes = num_classes
        self.threshold = overlap_targethresh

        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = variance
        



    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds (relative centre size), conf preds,
            and prior boxes (centre size) from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors, 4), point form
            ground_targetruth (tensor): Tuple containing boxes (in point form) and labels (LongTensor),
                boxes shape: [batch_size,num_objs, 4]
                labels shape: [batch_size,num_objs]
        """
        loc_pred, conf_pred, priors = predictions
        target_boxes, labels = targets
        
        batch_size = loc_pred.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        num_targets = target_boxes.size(1)
        
        check_shape ("loc_pred", loc_pred, [batch_size, num_priors, 4])
        check_shape ("conf_pred", conf_pred, [batch_size, num_priors, num_classes])
        check_shape ("priors", priors, [num_priors, 4])
        
        check_shape ("target_boxes", target_boxes, [batch_size, num_targets, 4])
        check_shape ("labels", labels, [batch_size, num_targets])
        
        assert labels.data.max() < num_classes, "provided labels must be between 0 and num_classes - 1"

        # match priors (default boxes) and ground truth boxes
        loc_target = loc_pred.data.new(loc_pred.size())
        conf_target = labels.data.new(batch_size, num_priors)
        
        for idx in range(batch_size):    
            loc_target[idx], conf_target[idx] = match(
                self.threshold, target_boxes[idx].data, priors.data, self.variance, labels[idx].data)
            
            
        # wrap targets
        loc_target = Variable(loc_target, requires_grad=False)
        conf_target = Variable(conf_target, requires_grad=False)
        
        is_pos = conf_target > 0
        num_pos = is_pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = is_pos.unsqueeze(is_pos.dim()).expand_as(loc_pred)
        loc_pred = loc_pred[pos_idx].view(-1, 4)
        loc_target = loc_target[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_pred, loc_target, size_average=False)
        
        # print(loc_target.size(), loc_pred.ne(loc_target).sum())

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_pred.contiguous().view(-1, self.num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_target.view(-1), reduce=False)
                
        # Hard Negative Mining
        neg_conf = loss_conf.view(batch_size, -1).clone()    
        neg_conf[is_pos] = 0  # filter out pos boxes for now
         
        _, loss_idx = neg_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
         
        num_pos = is_pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=is_pos.size(1)-1)
        is_neg = idx_rank < num_neg.expand_as(idx_rank)
         
        is_used = (is_pos | is_neg)
        # Confidence Loss Including Positive and Negative Examples
        loss_conf = loss_conf[is_used.view(-1)].sum()
         
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        return loss_loc / N, loss_conf / N
    
    