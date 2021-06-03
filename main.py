import os

from config import DefaultConfig
import torch as t
import torchvision as tv
from data.dataset import AnimeImgs
from torch.utils.data import DataLoader
from model.models import Generator, Discriminator
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

opt = DefaultConfig()
Tensor = t.cuda.FloatTensor if opt.device == 'cuda' else t.FloatTensor

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
    # criterion = t.nn.BCELoss().to(device=opt.device)
    #
    # true_labels = t.ones(opt.batch_size).to(device=opt.device)
    # fake_labels = t.zeros(opt.batch_size).to(device=opt.device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device=opt.device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device=opt.device)

    # errord_meter = AverageValueMeter()
    # errorg_meter = AverageValueMeter()

    for epoch in range(opt.max_epoch):
        for ii, (img, _) in tqdm(enumerate(dataloader)):
            real_imgs = img.to(opt.device)
            one = t.tensor(1, dtype=t.float).to(opt.device)
            mone = one * -1
            mone.to(opt.device)
            error_d_real = t.tensor(0, dtype=t.float).to(opt.device)
            error_d_fake = t.tensor(0, dtype=t.float).to(opt.device)

            if ii % opt.D_every == 0:
                optimizer_d.zero_grad()
                output = netD(real_imgs)
                error_d_real = output
                error_d_real = error_d_real.mean()
                # error_d_real.backward(mone)

                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_imgs = netG(noises).detach()
                output = netD(fake_imgs)
                error_d_fake = output
                error_d_fake = error_d_fake.mean()
                # error_d_fake.backward(one)

                gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
                # gradient_penalty.backward()

                error_d = error_d_fake - error_d_real + gradient_penalty * opt.lambda_gp
                error_d.backward()
                optimizer_d.step()
                # errord_meter.add(error_d.item())

            optimizer_g.zero_grad()

            if ii % opt.G_every == 0:
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_imgs = netG(noises)
                output = netD(fake_imgs)
                error_g = output
                error_g = - error_g.mean()
                error_g.backward()
                optimizer_g.step()
                # errorg_meter.add(-error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                vis.plot('errord', error_d.item())
                vis.plot('errorg', -error_g.item())
            if opt.vis and ii % 80 == 80 - 1:
                fix_fake_imgs = netG(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')

        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
            t.save(netD.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netG.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            # errord_meter.reset()
            # errorg_meter.reset()


@t.no_grad()
def generate(**kwargs):
    pass


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates).reshape(-1, 1)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == '__main__':
    import fire

    fire.Fire()