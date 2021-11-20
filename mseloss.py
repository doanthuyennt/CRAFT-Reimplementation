import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        # batch_size = pre_loss.shape[0]
        # sum_loss = torch.mean(pre_loss.view(-1))*0
        # pre_loss = pre_loss.view(batch_size, -1)
        # loss_label = loss_label.view(batch_size, -1)
        # internel = batch_size
        positive_pixel = (loss_label > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pre_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        negative_pixel = (loss_label <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)

        if negative_pixel_number < 3*positive_pixel_number:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
        else:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region.view(-1), int(3*positive_pixel_number))[0]) / (positive_pixel_number*3)

        total_loss = positive_loss + negative_loss
        return total_loss



    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]