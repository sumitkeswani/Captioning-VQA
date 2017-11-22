import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable

batch_size = 20
input_size = 10
output_size = 1
hidden_size = 5

global_avg = 2

class Model(nn.Module):
	def __init__(self, global_avg, input_size, hidden_size, output_size):
		super(Model, self).__init__()
		self.net1 = Net1(input_size, hidden_size, output_size)
		self.net1 = nn.DataParallel(self.net1)
		self.net2 = Net2(input_size)
		self.layer = nn.Linear(4, global_avg)

	def forward(self, x):
		out1 = self.net1(x)
		out2 = self.net2(x)
		out = torch.cat((out1, out2), 1)
		out = F.relu(out)
		out = self.layer(out)
		out = F.softmax(out)
		return out

class Net1(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Net1, self).__init__()
		self.layer1 = nn.Linear(input_size, hidden_size)
		self.layer2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		out = self.layer1(x)
		out = F.relu(out)
		out = self.layer2(out)
		return out

class Net2(nn.Module):
	def __init__(self, input_size):
		super(Net2, self).__init__()
		self.layer1 = nn.Linear(input_size, 3)

	def forward(self, x):
		out = self.layer1(x)
		out = F.relu(out)
		return out

torch.manual_seed(0)
input = autograd.Variable(torch.rand(batch_size, input_size))
# print "input:", input.view(1,-1)

target = autograd.Variable((torch.rand(batch_size, global_avg)*2).long())
# print "target:", target.view(1,-1)

model = Model(global_avg, input_size, hidden_size, output_size).cuda()

output = model(input)
print output

