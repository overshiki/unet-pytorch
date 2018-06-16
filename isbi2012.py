from model import * 
from dataset.isbi.c2012.feed import load 
from utils import loader, train
import torch, os

def pixel_wise_train_direct(data, mask, device=0):
	ld = loader(data, mask, minibatch=5)
	_model = unet()

	def correct_fun(correct, _len):
		return correct.item()*1./(_len)

	savePath = "./save/"
	if not os.path.exists(savePath):
		os.mkdir(savePath)
	# savePath = None

	# loss = nn.SmoothL1Loss()
	loss = torch.nn.CrossEntropyLoss()

	train(_model, ld, correct_fun=correct_fun, num_epoch=1000, device=0, savePath=savePath, loss=loss)



if __name__ == '__main__':
	data, mask = load()
	pixel_wise_train_direct(data, mask, device=0)

