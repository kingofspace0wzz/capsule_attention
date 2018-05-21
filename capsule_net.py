import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod


def squash(s, dim=-1):
    '''
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
    '''
    sj_norm = torch.norm(s, 2, dim).unsqueeze(-1)
    square_norm = sj_norm**2
    
    return square_norm / (1 + square_norm) * s / (sj_norm + 1e-8)


class CapsuleNet(nn.Module):

    def __init__(self, img_shape, in_conv_channels, out_conv_channels, primary_channels, 
                primary_dim, num_classes, out_dim, num_routing, 
                device: torch.device, kernel_size=9):
        super(CapsuleNet, self).__init__()
        
        self.img_shape = img_shape
        self.in_conv_channels = in_conv_channels
        self.out_conv_channels = out_conv_channels
        self.primary_channels = primary_channels
        self.device = device
        self.num_classes = num_classes
        self.primary_dim = primary_dim
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(self.in_conv_channels, self.out_conv_channels, kernel_size=kernel_size, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.primary_caps = PrimaryCapsules(self.out_conv_channels, self.primary_channels)
        # 1152 for MNIST
        in_caps = int(out_conv_channels / primary_dim * (img_shape[1] - 2*(
            kernel_size-1)) * (img_shape[2] - 2*(kernel_size-1)) / 4)
        self.object_caps = RoutingCapsules(primary_dim, in_caps, num_classes, out_dim, num_routing, self.device)

        # decoder, 3 FC layers
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * num_classes, 512),
            nn.ReLU(inplace=True),
         	nn.Linear(512, 1024),
         	nn.ReLU(inplace=True),
         	nn.Linear(1024, int(prod(img_shape))),
         	nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.primary_caps(out)
        out = self.object_caps(out)
        # compute the norm of each capsule output
        preds = torch.norm(out, dim=-1)

		# Reconstruct the *predicted* image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder((out*y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)
        return preds, reconstructions

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps=8,
              kernel_size=9, stride=2, padding=0):
        """
		Initialize the layer.

		Args:
			in_channels: 	Number of input channels.
			out_channels: 	Number of output channels.
			dim_caps:		Dimensionality, i.e. length, of the output capsule vector.
		
		"""
        super(PrimaryCapsules, self).__init__()
        self.primary_dim = dim_caps
        # primary_dim=8 convolutional units by default
        self.conv_units = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 
                    kernel_size=kernel_size, stride=stride, padding=padding) for i in range(self.primary_dim)])

    def forward(self, x):
        # create primary_dim=8 convolutional units
        s = [self.conv_units[i](x) for i in range(self.primary_dim)]

        # unit => [batch_size, primary_dim=8, out_channels=32, 6, 6]
        s = torch.stack(s, dim=1)
        # unit => [batch_size, 1152, 8])
        s = s.view(x.size(0), -1, self.primary_dim)
        # compute v = squash(s)
        
        return squash(s)


class RoutingCapsules(nn.Module):
    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device: torch.device):
        """
		Initialize the layer.

		Args:
			in_dim: 		Dimensionality (i.e. length) of each capsule vector.
			in_caps: 		Number of input capsules if digits layer.
			num_caps: 		Number of capsules in the capsule layer
			dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
			num_routing:	Number of iterations during routing algorithm		
		"""
        super(RoutingCapsules, self).__init__()
        self.in_dim = in_dim    # 8
        self.in_caps = in_caps  #   1152
        self.num_caps = num_caps    # 10
        self.dim_caps = dim_caps    # 16
        self.num_routing = num_routing
        self.device = device
        # W => [1 x 10 x 1152 x 16 x 8]
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim))

    def forward(self, x):
        # [batch_size, 1, 1152, 8, 1]
        x = x.unsqueeze(1).unsqueeze(4)

        # u_hat => [batch_size, 10, 1152, 16, 1]
        u_hat = torch.matmul(self.W, x)
        # u_hat => [batch_size, 10, 1152, 16]
        u_hat = u_hat.squeeze()
        temp_u_hat = u_hat.detach()

        batch_size = x.size(0)
        # b => [128, 10, 1152, 1]
        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for i in range(self.num_routing):
            # c: coeff from in_caps to num_caps
            c = F.softmax(b, dim=1)
            
            # c => [batch, 10, 1152, 1]
            # c * u_hat => [batch, 10, 1152, 16]
            # s => [batch, 10, 16]
            s = (c * temp_u_hat).sum(dim=2)

            v = squash(s)
            # [batch_size, 10, 1152, 16]x[batch, 10, 16, 1] => [batch_size, 10, 1152, 1]
            b += torch.matmul(temp_u_hat, v.unsqueeze(-1))

        c = F.softmax(b, dim=1)
        s = (c * temp_u_hat).sum(dim=2)
        v = squash(s)

        return v

class MarginLoss(nn.Module):
	def __init__(self, size_average=False, loss_lambda=0.5):
		'''
		Margin loss for digit existence
		Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2
		
		Args:
			size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
			loss_lambda: parameter for down-weighting the loss for missing digits
		'''
		super(MarginLoss, self).__init__()
		self.size_average = size_average
		self.m_plus = 0.9
		self.m_minus = 0.1
		self.loss_lambda = loss_lambda

	def forward(self, inputs, labels):
		L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
		L_k = L_k.sum(dim=1)

		if self.size_average:
			return L_k.mean()
		else:
			return L_k.sum()

class CapsuleLoss(nn.Module):
	def __init__(self, loss_lambda=0.5, recon_loss_scale=5e-4, size_average=False):
		'''
		Combined margin loss and reconstruction loss. Margin loss see above.
		Sum squared error (SSE) was used as a reconstruction loss.
		
		Args:
			recon_loss_scale: 	param for scaling down the the reconstruction loss
			size_average:		if True, reconstruction loss becomes MSE instead of SSE
		'''
		super(CapsuleLoss, self).__init__()
		self.size_average = size_average
		self.margin_loss = MarginLoss(size_average=size_average, loss_lambda=loss_lambda)
		self.reconstruction_loss = nn.MSELoss(size_average=size_average)
		self.recon_loss_scale = recon_loss_scale

	def forward(self, inputs, labels, images, reconstructions):
		reconstruction_loss = self.reconstruction_loss(reconstructions, images)
		margin_loss = self.margin_loss(inputs, labels)
		caps_loss = (margin_loss + self.recon_loss_scale * reconstruction_loss)

		return caps_loss
