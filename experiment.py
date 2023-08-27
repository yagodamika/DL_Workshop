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
import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def create_arg_parser():
	parser = argparse.ArgumentParser()
	"""
	num_iterations, seed, input_depth, input_dims, optimizer_type, lr,
	lower_lr, display_rate, cut_size, cutn, results_folder, model
	"""
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	parser.add_argument('--results_folder', type=str, default="", help='folder where to save the results') #TODO define
	parser.add_argument('--subfolder', type=str, default="", help='sub folder where to save the results')  # TODO define
	parser.add_argument('--input_depth', type=int, default=32, help='input depth for Deep Image Prior')
	parser.add_argument('--optimizer_type', type=str, default="Adam", help='optimizer type')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--model', type=str, default="EFFICIENTNET", help='name of classifier') # "VIT_B_16", "EFFICIENTNET"
	parser.add_argument('--num_iterations', type=int, default=1000, help='iterations number')
	parser.add_argument('--add_noise', type=bool, default=True, help='a flag that indicates whether to add noise to DIP input')
	parser.add_argument('--display_rate', type=int, default=100, help='on what iterations to save image')
	parser.add_argument('--logit_idx', type=int, default=90, help='wanted class logit')

	return parser.parse_args()

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def get_noise(input_depth, spatial_size, noise_type='u', var=1. / 10):
	"""
	Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
	Args:
		input_depth: number of channels in the tensor
		method: `noise` for filling tensor with noise
		spatial_size: spatial size of the tensor to initialize
		noise_type: 'u' for uniform; 'n' for normal
		var: a factor that the noise will be multiplied by.
	"""
	if isinstance(spatial_size, int):
		spatial_size = (spatial_size, spatial_size)

	shape = [1, input_depth, spatial_size[0], spatial_size[1]]
	net_input = torch.zeros(shape)

	if noise_type == 'u':
		net_input.uniform_()
	elif noise_type == 'n':
		net_input.normal_()

	net_input *= var
	return net_input

def optimize_network(args):
	loss_list = []

	if args.seed is not None:
		set_seed(args.seed)

	# create results folder
	if not os.path.isdir(args.results_folder):
		os.makedirs(args.results_folder)

	# create results sub folder
	full_sub_folder = args.results_folder + "/" + args.subfolder
	if not os.path.isdir(full_sub_folder):
		os.makedirs(full_sub_folder)

	# Initialize DIP network
	DIP_net = get_net(
		args.input_depth, 'skip',
		pad='reflection',
		skip_n33d=128, skip_n33u=128,
		skip_n11=4, num_scales=7,
		upsample_mode='bilinear',
	).to(device)

	# Initialize input noise
	input_depth = 3
	sideY = 512
	sideX = 512

	net_input = get_noise(input_depth, 'noise', (sideY, sideX), noise_type='u', var=1./10) #TODO var?

	net_input = net_input.type(torch.cuda.FloatTensor)
	net_input.to(device)

	if args.optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(DIP_net.parameters(), args.lr)
	elif args.optimizer_type == 'MADGRAD':
		optimizer = MADGRAD(DIP_net.parameters(), args.lr, weight_decay=0.01, momentum=0.9)

	# get model
	if args.model == "VIT_B_16":
		classifier = torchvision.models.vit_b_16(
			weights=torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
	if args.model == "EFFICIENTNET":
		classifier = torchvision.models.efficientnet_v2_l(
			weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
		mean = [0.5, 0.5, 0.5]
		std = [0.5, 0.5, 0.5]

		classifier.to(device)
		classifier.eval()

	for i in range(args.num_iterations):
		optimizer.zero_grad(set_to_none=True)

		if args.add_noise:
			noise = get_noise(input_depth, 'noise', (sideY, sideX), noise_type='n', var=0.01)
			noise = noise.to(device=device)

			noisy_net_input = net_input + noise

			with torch.cuda.amp.autocast():  # ops run in an op-specific dtype chosen by autocast to improve performance
				dip_out = DIP_net(noisy_net_input).float()

		else:
			with torch.cuda.amp.autocast():  # ops run in an op-specific dtype chosen by autocast to improve performance
				dip_out = DIP_net(net_input).float()  # run net on input

		# cutouts = make_cutouts(dip_out)

		out = classifier(dip_out)
		out = F.softmax(out, dim=1)
		loss = 1 - out[:, args.logit_idx].mean()

		loss_list.append(loss)
		loss.backward()
		optimizer.step()

		if (i % args.display_rate) == 0:
			image = TF.to_pil_image(dip_out[0])
			path = full_sub_folder + "/" + f'res_{i}.png'
			image.save(path, quality=100)


		print(f'Iteration {i} of {args.num_iterations}')

	# save DIP weights
	torch.save(DIP_net.state_dict(), full_sub_folder + "/net")

	return TF.to_pil_image(dip_out[0]), loss_list