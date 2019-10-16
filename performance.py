# -*- coding: utf-8 -*-
# @Author: luoling
# @Date:   2019-10-16 12:48:42
# @Last Modified by:   luoling
# @Last Modified time: 2019-10-16 13:24:30
import torch

from ptflops import get_model_complexity_info
from models.hybridnet import HybridNet


if __name__ == '__main__':
	# https://github.com/Lyken17/pytorch-OpCounter
	# https://github.com/rwightman/pytorch-image-models
	# https://github.com/sovrasov/flops-counter.pytorch

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = HybridNet(input_size=512).to(device)
	# inputs = torch.randn(1, 3, 512, 512).to(device)

	# flops, params = profile(model, inputs=(inputs, ), verbose=False)
	# print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))

	flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False)
	print('Flops:  ' + flops)
	print('Params: ' + params)
	# params: 0.71M | Flops: 3.52GMac