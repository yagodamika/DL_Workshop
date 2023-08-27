import sys

sys.path.append('./deep_image_prior')
from deep_image_prior.models import *
from deep_image_prior.utils.sr_utils import *
import numpy as np
import torch
import torch.optim
import time
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia.augmentation as K
from madgrad import MADGRAD
import random
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import kornia

global aug
global folder_name
global logit_idx
global RA_degrees
global RA_translate
global RandomGaussianBlur_kernelsize
global sharpness
global start_time
global noise_on
global cut_flag
global normalize_on
global input_noise
global combine_models


start_time = time.time()
sharpness = 0

#torch.use_deterministic_algorithms(True, warn_only=True)
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.append("/home/mika/.conda/envs/py310/lib/python3.10/site-packages/torchvision/transforms/transforms.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_grid(batch_size, height, width):
    """ Creates a grid with a batch size of batch_size,
        2 channels, with given height and width """
    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)

    # create torch tensors from numpy ndarrays
    xv_t = torch.from_numpy(xv)
    yv_t = torch.from_numpy(yv)

    T = torch.stack((xv_t, yv_t))
    grid = T.repeat(batch_size, 1, 1, 1)
    return grid

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [torch.cuda.device_count(), input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
    return net_input


def create_input_normal(input_depth, sideY, sideX, batch_size):
    # Create random input
    net_input = torch.zeros([batch_size, input_depth, sideY, sideX], device=device).normal_().div(10).detach()
    return net_input

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn):
        global cut_flag
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cutflag = cut_flag
        # Compose image augmentations
        kernel_size = int(RandomGaussianBlur_kernelsize), int(RandomGaussianBlur_kernelsize)
        self.augs = torch.nn.Sequential(
            K.RandomAffine(degrees=RA_degrees, translate=RA_translate, p=0.8, padding_mode='border',
                           resample='bilinear'),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomPerspective(0.45, p=0.8, resample='bilinear'),
            K.RandomGrayscale(p=0.15),
            K.RandomGaussianBlur(kernel_size, (0.1, 3)),
            K.RandomSharpness(sharpness=sharpness, p=0.5, keepdim=True),
            #K.RandomErasing(p=0.7)
        )

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        if sideY != sideX:
            input = K.RandomAffine(degrees=0, shear=10, p=0.5)(input)
            # if sizes are different, shear the image, so it fits into a square

        cutouts = []
        cn_size = [self.cut_size, self.cut_size]

        if self.cutflag:
            max_size = min(sideX, sideY)
            for cn in range(self.cutn):
                if cn > self.cutn - self.cutn // 4:
                    cutout = input
                else:
                    size = int(
                        max_size * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                    offsetx = torch.randint(0, sideX - size + 1, ())
                    offsety = torch.randint(0, sideY - size + 1, ())
                    cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                #cutout = F.interpolate(cutout, size=cn_size)
                #cutout = F.avg_pool2d(cutout, kernel_size=4, stride=2, padding=1)
                cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
                cutouts.append(cutout)

        else:
            for cn in range(self.cutn):
                cutout = input
                cutout = F.avg_pool2d(cutout, kernel_size=4, stride=2, padding=1)
                #cutout = F.interpolate(cutout, size=cn_size)
                cutouts.append(cutout)  # cutout is of size cut_size ^ 2

        cutouts = torch.cat(cutouts)  # Concatenate cutouts - the rows
        cutouts = self.augs(cutouts)
        return cutouts


def optimize_network(num_iterations, seed, input_depth, input_dims, optimizer_type, lr,
                     lower_lr, display_rate, cut_size, cutn, results_folder, model):
    global folder_name
    global logit_idx
    global normalize_on
    global noise_on
    global input_noise

    loss_list = []

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # create results folder
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # create results sub folder
    full_sub_folder = results_folder + "/" + folder_name
    if not os.path.isdir(full_sub_folder):
        os.makedirs(full_sub_folder)

    make_cutouts = MakeCutouts(cut_size, cutn)
    make_cutouts = nn.DataParallel(make_cutouts)

    # Initialize DIP skip network
    DIP_net = get_net(
        input_depth, 'skip',
        pad='reflection',
        skip_n33d=128, skip_n33u=128,
        skip_n11=4, num_scales=7,
        upsample_mode='bilinear',
    ).to(device)

    DIP_net = nn.DataParallel(DIP_net)

    # Initialize input noise
    input_depth = input_dims[0]
    sideY = input_dims[1]
    sideX = input_dims[2]
    # net_input = create_input_normal(input_depth, sideY, sideX, batch_size)

    if input_noise:
        net_input = get_noise(input_depth, 'noise', (sideY, sideX), noise_type='u', var=1. / 10)

    else: # input is positional encoding
        net_input = create_grid(batch_size= torch.cuda.device_count(), height = sideY, width = sideX)

    net_input = net_input.type(torch.cuda.FloatTensor)
    net_input.to(device)

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(DIP_net.parameters(), lr)
    elif optimizer_type == 'MADGRAD':
        optimizer = MADGRAD(DIP_net.parameters(), lr, weight_decay=0.01, momentum=0.9)

    # get model
    if combine_models:
        classifier1 = torchvision.models.vit_b_16(
            weights=torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)
        mean1 = [0.485, 0.456, 0.406]
        std1 = [0.229, 0.224, 0.225]
        classifier2 = torchvision.models.efficientnet_v2_l(
            weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        mean2 = [0.5, 0.5, 0.5]
        std2 = [0.5, 0.5, 0.5]

        classifier1.to(device)
        classifier1.eval()
        classifier1 = nn.DataParallel(classifier1)

        classifier2.to(device)
        classifier2.eval()
        classifier2 = nn.DataParallel(classifier2)
    else:
        if model == "VIT_B_16":
            classifier = torchvision.models.vit_b_16(
                weights=torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        if model == "EFFICIENTNET":
            classifier = torchvision.models.efficientnet_v2_l(
                weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        classifier.to(device)
        classifier.eval()
        classifier = nn.DataParallel(classifier)

    # training
    #    try:
    for i in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        if noise_on:
            if input_noise:
                noise = get_noise(input_depth, 'noise', (sideY, sideX), noise_type='n', var=0.01)
                # noise = noise.type(torch.cuda.FloatTensor)
            else: # positional encoding
                noise = get_noise(input_depth, 'noise', (sideY, sideX), noise_type='u', var=1.0)

            noise = noise.to(device=device)
            noisy_net_input = net_input + noise

            with torch.cuda.amp.autocast():  # ops run in an op-specific dtype chosen by autocast to improve performance
                dip_out = DIP_net(noisy_net_input).float()

        else:
            with torch.cuda.amp.autocast():  # ops run in an op-specific dtype chosen by autocast to improve performance
                dip_out = DIP_net(net_input).float()  # run net on input

        cutouts = make_cutouts(dip_out)

        if normalize_on:
            normalize_func = kornia.enhance.Normalize(mean, std)
            cutouts = normalize_func(cutouts)
            # cutouts = T.Normalize(mean, std, inplace=False)

        if combine_models:
            out1 = classifier1(cutouts)
            out1 = F.softmax(out1, dim=1)

            out2 = classifier2(cutouts)
            out2 = F.softmax(out2, dim=1)

            loss1 = 1 - out1[:, logit_idx].mean()
            loss2 = 1 - out2[:, logit_idx].mean()
            loss = 0.005 * loss1 + loss2
        else:
            out = classifier(cutouts)
            out = F.softmax(out, dim=1)
            loss = 1 - out[:, logit_idx].mean()

        loss_list.append(loss)
        loss.backward()
        optimizer.step()

        if (i % display_rate) == 0:
            # with torch.inference_mode():  # context manager to be used when operations will have no interactions with autograd
            image = TF.to_pil_image(dip_out[0])
            # path = "results/xl/13_cuts/"+folder_name+f'/res_{i}.png'
            path = full_sub_folder + "/" + f'res_{i}.png'
            image.save(path, quality=100)

        if lower_lr:  # lower the learning rate over time - multiply by 0.99
            optimizer.param_groups[0]['lr'] = max(0.00001, .99 * optimizer.param_groups[0]['lr'])

        print(f'Iteration {i} of {num_iterations}')

    # save DIP weights
    torch.save(DIP_net.state_dict(), full_sub_folder + "/net")

    return TF.to_pil_image(dip_out[0]), loss_list


def train_net(results_folder):
    global input_noise
    global combine_models

    num_iterations = 2500
    seed = 1  # random.randint(0, 2 ** 32)
    sideY = 512  # 512
    sideX = 512  # 512
    # logit_idx = 1 became global
    optimizer_type = 'Adam'
    lr = 1e-3
    lower_lr = False
    display_rate = 200
    cut_size = 224
    cutn = 11
    model = "EFFICIENTNET" #"VIT_H_14" # "EFFICIENTNET" #"VIT_B_16"
    input_noise = 1  # determines if input is noise or positional encoding
    combine_models = False

    if input_noise:
        input_depth = 32
    else: # positional encoding
        input_depth = 2

    input_dims = (input_depth, sideY, sideX)

    out, loss_list = optimize_network(num_iterations, seed, input_depth, input_dims, optimizer_type, lr,
                                      lower_lr, display_rate, cut_size, cutn, results_folder, model)

    create_config_file(num_iterations, seed, input_depth, sideY, sideX, optimizer_type, lr, lower_lr, cut_size, cutn,
                       results_folder)
    create_loss_file(loss_list, results_folder)
    create_loss_graph(loss_list, results_folder)
    create_smooth_loss_graph(loss_list,results_folder)
    create_log_loss_graph(loss_list, results_folder)

    return out
    # timestring = time.strftime('%Y%m%d%H%M%S')
    # out.save('final_result_{timestring}.png', quality=100)

"""
def train_net_only_aug():
    global aug
    num_iterations = 10000
    seed = 1  # random.randint(0, 2 ** 32)
    input_depth = 3
    sideY = 512
    sideX = 512
    input_dims = (input_depth, sideY, sideX)
    logit_idx = 1
    batch_size = 1
    optimizer_type = 'Adam'
    lr = 1e-3
    lower_lr = False
    display_rate = 100
    cut_size = 224
    cutn = 70
    augs = ["affine", "horizontal", "perspective", "gaussian_blur", "sharpness", "gaussian_noise"]
    for a in augs:
        aug = a
        folder_path = "only_" + a
        out = optimize_network(num_iterations, seed, input_depth, input_dims, logit_idx, batch_size,
                               optimizer_type, lr, lower_lr, display_rate, cut_size, cutn, folder_path)
    return out
    # timestring = time.strftime('%Y%m%d%H%M%S')
    # out.save('final_result_{timestring}.png', quality=100)
"""

def check_classifier():
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    with torch.no_grad():
        classifier = torchvision.models.efficientnet_v2_l(
            weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        classifier.to(device)
        classifier.eval()

        # enter a goldfish image and see what the net outputs
        # address = "goldfish/fish_0.jpg"
        address = "res_9900.png"
        create_tensor = transforms.ToTensor()

        img = Image.open(address)

        ten = create_tensor(img)
        ten = ten.unsqueeze(0)
        ten = ten.type(torch.cuda.FloatTensor)
        ten.to(device)

        output = classifier(ten)
        softmax = F.softmax(output)
        idx = torch.argmax(softmax)
        print(idx)


def get_config_xl(path, row, sharp_on):
    """
    gets a path for an excel file with
    configuration,assigns the configuration
    parameters to global variables
    """
    global folder_name
    global logit_idx
    global RA_degrees
    global RA_translate
    global RandomGaussianBlur_kernelsize
    global sharpness
    global noise_on
    global cut_flag
    global normalize_on

    # row starts from 0
    df = pd.read_excel(path)
    folder_name = df.iloc[row]["folder_name"]
    logit_idx = df.iloc[row]["logit_idx"]
    RA_degrees = df.iloc[row]["RA_degrees"]
    RA_translate = df.iloc[row]["RA_translate"]
    RandomGaussianBlur_kernelsize = df.iloc[row]["RandomGaussianBlur_kernelsize"]
    noise_on = df.iloc[row]["Noise"]
    cut_flag = df.iloc[row]["Cut_flag"]
    normalize_on = df.iloc[row]["Normalize"]
    if sharp_on:
        sharpness = df.iloc[row]["sharpness"]



def create_config_file(num_iterations, seed, input_depth, sideY, sideX, optimizer_type, lr, lower_lr, cut_size, cutn,
                       results_folder):
    global logit_idx
    global start_time
    path = results_folder + "/" + folder_name + "/Config.txt"

    text = "Config File\n"
    text += "Start Time: " + str(start_time) + "\n"
    text += "Running Time: " + str(time.time() - start_time) + "\n"
    text += "Num Iterations: " + str(num_iterations) + "\n"
    text += "Seed: " + str(seed) + "\n"
    text += "Input Depth: " + str(input_depth) + "\n"
    text += "Y width: " + str(sideY) + "\n"
    text += "X length: " + str(sideX) + "\n"
    text += "Optimizer Type: " + optimizer_type + "\n"
    text += "Learning Rate: " + str(lr) + "\n"
    text += "Lower Learning Rate: " + str(lower_lr) + "\n"
    text += "Cut size: " + str(cut_size) + "\n"
    text += "Cuts Number: " + str(cutn) + "\n"
    text += "Logit Index: " + str(logit_idx) + "\n"
    text += "GPUs number: " + str(torch.cuda.device_count()) + "\n"

    f = open(path, "a")
    f.write(text)
    f.close()


def create_loss_file(loss_list, results_folder):
    path = results_folder + "/" + folder_name + "/Loss.txt"
    file = open(path, "a")
    for i, loss in enumerate(loss_list):
        text = f'Iterarion {i}: loss = {loss}\n'
        file.write(text)

    file.close()


def create_loss_graph(loss_list, results_folder):
    path = results_folder + "/" + folder_name + "/Loss.png"
    iterations = np.array(range(1, len(loss_list) + 1))
    loss = np.array(loss_list)
    plt.plot(iterations, loss, color='r', label='Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("The Loss as a Function of the iteration number")
    plt.savefig(path)

def create_smooth_loss_graph(loss_list,results_folder):
    path = results_folder + "/" + folder_name + "/Loss_smoothed.png"
    loss = torch.mean(torch.Tensor(loss_list).view(-1, 50), dim=1)
    plt.plot(range(1, len(loss_list)//50 + 1), loss, color='r', label='Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Smooth Loss")
    plt.title("The 5 timestep mean Loss as a Function of the iteration number")
    plt.savefig(path)

def create_log_loss_graph(loss_list, results_folder):
    path = results_folder + "/" + folder_name + "/Log_Loss.png"
    iterations = np.array(range(1, len(loss_list) + 1))
    log = np.log(np.array(loss_list))
    plt.plot(iterations, log, color='r', label='log(Loss)')
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("The Logarithm of the Loss as a Function of the iteration number")
    plt.savefig(path)

def run_xl(xl_path, sharp_on, results_folder, start_idx, end_idx):
    """
    runs a few rows from a config excel
    """
    for row in range(start_idx, end_idx):
        get_config_xl(xl_path, row, sharp_on)
        train_net(results_folder)

def get_config(sharp_on, logit):
    """
    assigns configuration
    parameters to global variables
    """
    global folder_name
    global logit_idx
    global RA_degrees
    global RA_translate
    global RandomGaussianBlur_kernelsize
    global sharpness
    global noise_on
    global cut_flag
    global normalize_on

    folder_name = "results"
    logit_idx = logit
    RA_degrees = 30
    RA_translate = 0.3
    RandomGaussianBlur_kernelsize = 31
    noise_on = 0
    cut_flag = 1
    normalize_on = 0
    if sharp_on:
        sharpness = 0

def run_config(sharp_on, logit):
    """
    runs a config
    """
    get_config(sharp_on, logit)
    train_net(results_folder)

#xl_path = "configs/configurations.xlsx"
sharp_on = False
results_folder_base = "configuration_excel_RESULTS/best_config/no erasing/2500_iters/"
#start_idx = 0  # 0
#end_idx = 1    # 32
#run_xl(xl_path, sharp_on, results_folder, start_idx, end_idx)
#run_config(sharp_on)

# train_net()
# check_classifier()


classes = [(0, "tench"), (3, "tiger_shark"), (4, "hammerhead"),
           (5, "crampfish"), (7, "cock"), (10, "brambling"),
           (17, "jay"), (29, "axolotl"), (30, "bullfrog"),
           (76, "tarantula"), (99, "goose"), (153, "Maltese_dog")]

for class_tuple in classes:
    logit = class_tuple[0]
    class_name = class_tuple[1]
    results_folder = results_folder_base + class_name
    run_config(sharp_on, logit)
