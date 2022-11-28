import torch
import torch.nn as nn


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, x0, t.float())[0]
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class RateDistortionLoss(nn.Module):
	def __init__(self, type="sigmoid", constant_lambda=True):
		"""
		Initialise Rate-Distortion Loss function
		constant_lambda:: whether to keep lambda as constant or dynamically optimise it
		type:: which of the simplified hyperlatent rate calcuations to use [normal, sigmoid]
			normal - assumes that hyperlatens are normally distributed N(0,1) convolved with U[-1/2, 1/2]
			sigmoid - treats sigmoid of hyperlatents as their CDF,  convolved with U[-1/2, 1/2]
		"""
		super(RateDistortionLoss, self).__init__()
		if type == "normal":
			self.hyper_cumulative = self.simple_cumulative
		elif type == "sigmoid":
			self.hyper_cumulative = self.sigmoid_cumulative
		
		if constant_lambda:
			self.assign_lambda = self.constant_lambda
		else:
			self.assign_lambda = self.lambda_update
			self.epsilon = 1e-2
	
	def cumulative(self, mu, sigma, x):
		"""
		Calculates CDF of Normal distribution with parameters mu and sigma at point x
		"""
		half = 0.5
		const = (2 ** 0.5)
		return half * (1 + torch.erf((x - mu) / (const * sigma)))

	def simple_cumulative(self, x):
		"""
		Calculates CDF of Normal distribution with mu = 0 and sigma = 1
		"""
		half = 0.5
		const = -(2 ** -0.5)
		return half * torch.erf(const * x)

	def sigmoid_cumulative(self, x):
		"""
		Calculates sigmoid of the tensor to use as a replacement of CDF
		"""
		return torch.sigmoid(x)
	
	def lambda_update(self, lam, distortion):
		"""
		Updates Lagrangian multiplier lambda at each step
		"""
		return self.epsilon * distortion + lam

	def constant_lambda(self, lam, distortion):
		"""
		Assigns Lambda the same in the case lambda is constant
		"""
		return 0.025
	
	def latent_rate(self, mu, sigma, y):
		"""
		Calculate latent rate
		
		Since we assume that each latent is modelled a Gaussian distribution convolved with Unit Uniform distribution we calculate latent rate
		as a difference of the CDF of Gaussian at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
		
		See apeendix 6.2
		J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
		“Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
		Available: https://openreview.net/forum?id=rkcQFMZRb.
		"""
		upper = self.cumulative(mu, sigma, (y + .5))
		lower = self.cumulative(mu, sigma, (y - .5))
		return -torch.sum(torch.log2(torch.abs(upper - lower)), dim=(1, 2, 3))
	
	def hyperlatent_rate(self, z):
		"""
		Calculate hyperlatent rate
		Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
		as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
		See apeendix 6.2
		J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
		“Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
		Available: https://openreview.net/forum?id=rkcQFMZRb.
		"""
		upper = self.hyper_cumulative(z + .5)
		lower = self.hyper_cumulative(z - .5)
		return -torch.sum(torch.log2(torch.abs(upper - lower)), dim=(1, 2, 3))
	
	def forward(self, model, x0, t, e, b, lam):
		"""
		Calculate Rate-Distortion Loss
		"""
		a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
		x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

		x_hat, sigma, mu, y_hat, z_hat = model(x, x0, t)
		distortion = (e - x_hat).square().sum(dim=(1, 2, 3)).mean(dim=0)
		latent_rate = torch.mean(self.latent_rate(mu, sigma, y_hat))
		hyperlatent_rate = torch.mean(self.hyperlatent_rate(z_hat))
		lam = self.assign_lambda(lam, distortion)
		loss = lam * distortion + latent_rate + hyperlatent_rate
		return loss


loss_registry = {
    'simple': noise_estimation_loss,
    'rd': RateDistortionLoss,
}
