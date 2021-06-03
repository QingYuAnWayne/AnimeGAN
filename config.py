import warnings


class DefaultConfig(object):
    env = 'GAN'
    model = 'AnimeGAN'

    train_data_root = '/content/data/data/data'

    load_label_path = None

    batch_size = 256
    num_workers = 4
    printfreq = 1
    save_every = 10

    save_path = '/content/drive/MyDrive/AnimeGAN/results'

    max_epoch = 15000
    G_lr = 1e-4
    D_lr = 1e-4
    lr_decay = 0.95
    weight_decay = 1e-4

    G_every = 5
    D_every = 1
    image_size = 96
    beta1 = 0.5
    # netd_path = 'checkpoints/netd_199.pth' #预训练模型
    # netg_path = 'checkpoints/netg_199.pth'
    netd_path = None
    netg_path = None

    device = 'cpu'
    nz = 100
    ngf = 64
    ndf = 64
    plot_every = 1
    vis = True


    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差
    lambda_gp = 10
