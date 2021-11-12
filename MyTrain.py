import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F


def structure_loss(pred, mask):
    # print(pred.size())
    # print(mask.size())
    # print(pred.size())
    # print(mask.size())
    # print(torch.min(pred))
    # print(torch.min(mask))
    # print(torch.max(pred))
    # print(torch.max(mask))
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    # print(pred.size())
    # print(torch.min(pred))
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def iou(pred, mask):
    # print(pred.size())
    # print(mask.size())
    # print(torch.min(pred))
    # print(torch.min(mask))
    # print(torch.max(pred))
    # print(torch.max(mask))
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))

    return 100*(inter/union)


def dice_loss(pred, mask, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = torch.sigmoid(pred)

        #flatten label and prediction tensors
        pred = pred.view(-1)
        mask = mask.view(-1)

        intersection = (pred * mask).sum()
        dice = (2.*intersection + smooth)/(pred.sum() + mask.sum() + smooth)

        return 100*(1 - dice)

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            outputs = model(images)
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
            # ---- accuracy ----
            # dice score/coefficient
            # Iou
            # change the label: pink->white; red->black
        # total_train = 0
        # correct_train = 0
        # _, predicted = torch.max(loss.data, 0)
        # total_train += gts.nelement()
        # correct_train += predicted.eq(gts.data).sum().item()
        # train_accuracy = 100 * correct_train / total_train
        # avg_accuracy = train_accuracy / len(train_loader)
        #     mask = gts[0, 0, :, :]
        #     print(mask.size())
        #     print(lateral_map_2.size())
        #     print(gts.size())
        #     mask = F.upsample(mask, size=masks.shape, mode='bilinear', align_corners=False)
        #     mask = mask.sigmoid().data.cpu().numpy().squeeze()
        #     mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        #
            # res = lateral_map_2[0, 0, :, :]
            # res = F.upsample(res, size=res.shape, mode='bilinear', align_corners=False)
            # res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)




        dice = dice_loss(lateral_map_2, gts)

        iou2 = iou(lateral_map_2, gts)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()),"Dice loss: %d %%" % ((dice.mean())),"IOU: %d %%" % ((2.mean(iou))))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='PraNet_Res2Net')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PraNet().cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)

