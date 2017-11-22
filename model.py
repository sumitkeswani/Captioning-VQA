import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import VQA_model
from caption_model import EncoderCNN, DecoderRNN

class VQA_Cap(nn.Module):
	def __init__(self):
		super(VQA_Cap, self).__init__()
		
		# load VQA model
		log = torch.load('VQA_model/2017-08-04_00.55.19.pth')
		tokens = len(log['vocab']['question']) + 1
		net = torch.nn.DataParallel(VQA_model.Net(tokens))
		net.load_state_dict(log['weights'])
			
		# set require_grad to false
		for param in net.parameters():	
			param.requires_grad = False

		# Remove last layer and add a different layer
		net.module.classifier._modules['lin2'] = nn.Linear(1024, 512)	#replaced last linear layer!
		self.VQAmodel = net

		# load image captioning model
		encoder = EncoderCNN(256)
		decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

		# Load the trained model parameters
		encoder.load_state_dict(torch.load(args.encoder_path)).cuda()
		decoder.load_state_dict(torch.load(args.decoder_path)).cuda()

	def forward():




