import warnings


class DefaultConfig(object):
    env = 'GAN'
    model = 'AnimeGAN'

    train_data_root = './data/train'

    load_label_path = None

    batch_size = 256
    num_workers = 4
    printfreq = 1
    save_every = 5

    save_path = './results'

    max_epoch = 200
    G_lr = 2e-4
    D_lr = 2e-4
    lr_decay = 0.95
    weight_decay = 1e-4

    G_every = 5
    D_every = 1
    image_size = 96
    beta1 = 0.5
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

    device = 'cuda'
    nz = 100
    ngf = 64
    ndf = 64
    plot_every = 100
    vis = True


    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差
