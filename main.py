import os

from config import DefaultConfig
import torch as t
import torchvision as tv
from data.dataset import AnimeImgs
from torch.utils.data import DataLoader
from model.models import Generator, Discriminator
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

opt = DefaultConfig()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    t.device(opt.device)
    if opt.vis:
        from utils.visualizer import Visualizer
        vis = Visualizer(opt.env)

    dataset = AnimeImgs(opt.train_data_root, transforms=True, train=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                            drop_last=True)
    netD, netG = Discriminator(opt), Generator(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netD.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netG.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netD.to(device=opt.device)
    netG.to(device=opt.device)

    optimizer_g = t.optim.Adam(netG.parameters(), opt.G_lr, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netD.parameters(), opt.D_lr, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device=opt.device)

    true_labels = t.ones(opt.batch_size).to(device=opt.device)
    fake_labels = t.zeros(opt.batch_size).to(device=opt.device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device=opt.device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device=opt.device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epochs in iter(epochs):
        for ii, (img, _) in tqdm(enumerate(dataloader)):
            real_imgs = img.to(opt.device)

            if ii % opt.D_every == 0:
                optimizer_d.zero_grad()
                output = netD(real_imgs)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_imgs = netG(noises).detach()
                output = netG(fake_imgs)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.G_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_imgs = netG(noises)
                output = netD(fake_imgs)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                fix_fake_imgs = netG(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if (epochs+1) % opt.save_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epochs), normalize=True,
                                range=(-1, 1))
            t.save(netD.state_dict(), 'checkpoints/netd_%s.pth' % epochs)
            t.save(netG.state_dict(), 'checkpoints/netg_%s.pth' % epochs)
            errord_meter.reset()
            errorg_meter.reset()

@t.no_grad()
def generate(**kwargs):
    pass

if __name__ == '__main__':
    import fire
    fire.Fire()