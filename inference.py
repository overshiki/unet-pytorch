from timeit import default_timer as timer
import numpy as np
import cupy as cp
import torch
from torch.autograd import Variable
from utils import groupby

def preprocess(img, pos, half_size=15, device=0):
	with cp.cuda.Device(device):
		pos_x_left, pos_x_right = pos[:,0]-half_size, pos[:,0]+half_size
		pos_y_left, pos_y_right = pos[:,1]-half_size, pos[:,1]+half_size

		#TODO: current using filtering option, in the near future, change to padding option
		SELECT_MAP = (pos_x_left>=0)*(pos_y_left>=0)*(pos_x_right<2048)*(pos_y_right<2048)
		SELECT_INDEX = cp.where(SELECT_MAP>0)[0]

		pos_x_left, pos_x_right, pos_y_left, pos_y_right = pos_x_left[SELECT_INDEX], pos_x_right[SELECT_INDEX], pos_y_left[SELECT_INDEX], pos_y_right[SELECT_INDEX]

		pos = pos[SELECT_INDEX]

		# shape should be dx * dy * dz
		pos_x = pos_x_left
		pos_x = cp.expand_dims(pos_x, axis=0)
		adding = cp.expand_dims(cp.arange(2*half_size), 1)
		pos_x = pos_x+adding

		pos_y = pos_y_left
		pos_y = cp.expand_dims(pos_y, axis=0)
		adding = cp.expand_dims(cp.arange(2*half_size), 1)
		pos_y = pos_y+adding

		# x * y * N
		_x = img[pos_x[:, cp.newaxis, :], pos_y]

	return _x.get(), pos.get() #, groups_start, groups_end



def patch_interface(pos_x, pos_y, half_size):
	pos_x_left = pos_x - half_size
	pos_x_left = np.expand_dims(pos_x_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size), 1)
	pos_x = pos_x_left + adding

	pos_y_left = pos_y - half_size
	pos_y_left = np.expand_dims(pos_y_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size), 1)
	pos_y = pos_y_left+adding

	# x * y * N
	return pos_x[:, np.newaxis, :], pos_y


from model_convention import unet
def inference(img, state_path, device=0):

	model = unet().cuda(device)
	state = torch.load(state_path)
	model.load_state_dict(state)

	binary = np.zeros(img.shape)

	stride, patch_size = 60, 60

	with cp.cuda.Device(device):
		img = cp.asarray(img.astype(np.uint16))

		pos_x = np.arange(0, 2048, stride)
		pos_y = np.arange(0, 2048, stride)
		vx, vy = np.meshgrid(pos_x, pos_y)
		pos = cp.asarray(np.stack([vx, vy]).reshape((2, -1)).transpose([1,0]))


		X, pos = preprocess(img, pos, half_size=patch_size//2, device=device)

		X = X.transpose([2, 0, 1])

		indices = np.arange(len(X)).tolist()
		groups = groupby(indices, 100, key='mini')


		out_list = []
		for index, group in enumerate(groups):
			out = X[group]
			_pos = pos[group]

			out = Variable(torch.from_numpy(np.expand_dims(out, 1).astype(np.float32)).cuda(device))
			R = model(out)
			R = R.max(1)[1]
			patch = R.cpu().numpy().transpose([1,2,0])
			binary[patch_interface(_pos[:,0], _pos[:,1], patch_size//2)] = patch

	binary = (binary*255).astype('uint8')
	return binary
