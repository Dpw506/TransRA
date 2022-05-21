import torch
import torch.nn as nn
import time
import argparse
from model import ResDehaze,Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr,to_ssim_skimage
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from wavelet_ssim_loss import WSloss
import datetime
import logger

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=1000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
args = parser.parse_args()
args.best_ssim = 0
best_epoch = 0
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_dataset=os.path.join(args.data_dir, 'train')

# --- test --- #
test_dataset = os.path.join(args.data_dir, 'val')
val_dataset = os.path.join(args.data_dir, 'test')
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir=os.path.join(args.model_save_dir,'output_result')
logger = logger.create_logger(output_dir=args.model_save_dir, name='ResDehaze')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = ResDehaze()
logger.info(f"MyEnsembleNet parameters: {sum(param.numel() for param in MyEnsembleNet.parameters())}")

DNet = Discriminator()
logger.info(f"MyEnsembleNet parameters: {sum(param.numel() for param in DNet.parameters())}")

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[300,600,800], gamma=0.5)

D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[300,600,800], gamma=0.5)
# --- Load training data --- #
dataset = dehaze_train_dataset(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = dehaze_test_dataset(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet= torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)
DNet = DNet.to(device)
DNet= torch.nn.DataParallel(DNet, device_ids=device_ids)
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

msssim_loss = msssim
wsloss = WSloss()


# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(os.path.join(args.teacher_model , 'epoch100000.pkl')))
    logger.info('--- weight loaded ---')
except:
    logger.info('--- no weight loaded ---')

# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch):
    start_time = time.time()
    scheduler_G.step()
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    logger.info(f"-->Epoch[{epoch}]")
    for batch_idx, (hazy, clean) in enumerate(train_loader):
        # print(batch_idx)
        iteration +=1
        hazy = hazy.to(device)
        clean = clean.to(device)
        output= MyEnsembleNet(hazy)

        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        # no more forward
        D_loss.backward(retain_graph=True)
        MyEnsembleNet.zero_grad()
        adversarial_loss = torch.mean(1 - fake_out)
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        #msssim_loss_ = 1 - msssim_loss(output, clean, normalize=True)
        wloss = wsloss(output,clean)

        total_loss = smooth_loss_l1 + 0.001 * perceptual_loss + 0.0005 * adversarial_loss + 0.1 * wloss


        total_loss.backward()
        D_optim.step()
        G_optimizer.step()

#         if iteration % 2 == 0:
#             frame_debug = torch.cat(
#                 (hazy, output, clean), dim=0)
#             writer.add_images('train_debug_img', frame_debug, iteration)
#         if iteration % 1 == 0:
#             print("===> Epoch[{}]({}/{}): total_loss: {:.6f}: L2_loss: {:.6f}: per_loss: {:.6f}: adv_loss: {:.6f}: wloss: {:.6f}: time: {:.4f}".format(epoch, batch_idx, len(train_loader),
#                                                                                                                                              total_loss, L2_loss,
#                                                                                                                                              perceptual_loss, adversarial_loss,
#                                                                                                                                              wloss, time.time()-start_time))
        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)
        writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                            'perceptual': perceptual_loss.item(),
                                            'wloss':wloss.item()
                                               }, iteration)
        writer.add_scalars('GAN_training', {
            'd_loss':D_loss.item(),
            'd_score':real_out.item(),
            'g_score':fake_out.item()
            }, iteration
            )

    if epoch % 5 == 0:
        logger.info(f"testing on epoch: {epoch}")
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            recon_psnr_list = []
            recon_ssim_list = []
            MyEnsembleNet.eval()
            for batch_idx, (hazy_upl,hazy_upr,hazy_downl,hazy_downr,hazy, clean) in enumerate(test_loader):

                clean = clean.to(device)
                hazy = hazy.to(device)
                frame_out = MyEnsembleNet(hazy)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)

            frame_debug = torch.cat((frame_out,clean), dim =0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr':avr_psnr,
                'testing ssim': avr_ssim
                                    }, epoch)
            if args.best_ssim < avr_ssim:
                args.best_ssim = avr_ssim
                best_epoch = epoch
                torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))
            logger.info(
                f"Epoch: {epoch}\t"
                f"avg_psnr: {avr_psnr:.6f}\t"
                f"avg_ssim: {avr_ssim:.6f}\t"
                f"best_epoch: {best_epoch}\t"
                f"best_ssim: {args.best_ssim:.6f}\t"
                f"lr: {G_optimizer.param_groups[0]['lr']}\t"
                f"used_time: {time.time() - start_time:.6f}")

writer.close()
