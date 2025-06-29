import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ======= GMM Model Class and Visualization ======= #

class GMM(torch.nn.Module):
    def __init__(self, K: int, dim: int):
        super().__init__()
        self.K, self.dim = K, dim
        self.logits = torch.nn.Parameter(torch.log(torch.tensor(ratios)))  # unnormalized log(alpha)
        self.mu = torch.nn.Parameter(torch.tensor(means))  # means
        self.log_std = torch.nn.Parameter(torch.log(torch.tensor(stds)))  # log std (shared across dims)

    # helpers ------------------------------------------------------- #
    def std(self):  # positive σ
        return torch.exp(self.log_std)

    def _comp_log_prob(self, x):  # log p_k(x)  (N,K)
        x = x.unsqueeze(1)  # (N,1,D)
        mu = self.mu.unsqueeze(0)  # (1,K,D)
        std = self.std().unsqueeze(0).unsqueeze(2)
        log_gauss = -0.5 * (((x - mu) / std) ** 2).sum(-1)
        log_gauss -= self.dim * torch.log(std.squeeze(2))
        log_gauss -= 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi))
        log_pi = F.log_softmax(self.logits, dim=0)  # (K,)
        return log_gauss + log_pi  # (N,K)

    def log_prob(self, x):  # mixture log-prob  (N,)
        return torch.logsumexp(self._comp_log_prob(x), dim=1)

    @torch.no_grad()
    def sample(self, n: int):  # draw n points  (n,dim)
        pi = F.softmax(self.logits, dim=0)
        comps = torch.multinomial(pi, n, replacement=True)  # (n,)
        eps = torch.randn(n, self.dim, device=self.mu.device)
        return eps * self.std()[comps].unsqueeze(1) + self.mu[comps]


def plot_policy(policy, title_addon='', num_samples=10000):
    with torch.no_grad():
        plt.figure(figsize=(6, 6))
        x_samples = policy.sample(num_samples).cpu()
        plt.scatter(x_samples[:, 0], x_samples[:, 1], s=2, alpha=0.2)

        g_probs = F.softmax(policy.logits).detach().cpu()
        for i in range(policy.K):
            cur_color = f"C{i}"
            circle = plt.Circle(policy.mu[i].cpu(), 2 * policy.std()[i].cpu().item(), color=cur_color, fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            # set the intensity of the color to be the probability of the Gaussian
            plt.scatter(*policy.mu[i].cpu(), color=cur_color, s=100, marker='x')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        gaussians_probs = [float(np.round(x, 2)) for x in F.softmax(policy.logits, dim=0).cpu().numpy()]
        centric_gaussian_mean = np.round(policy.mu[0].cpu().numpy(), 2).tolist()
        centric_gaussian_std = np.round(policy.log_std.exp()[0].cpu().numpy(), 2).tolist()
        plt.title(f'{title_addon}Probs for Gaussians: {gaussians_probs}\n'
                  f'G_0 Mean: {centric_gaussian_mean}\nG_0 Std: {centric_gaussian_std}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.show()


# ======= Reward Function ======= #

def single_mode_reward(x):
    radius = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)  # Euclidean distance from origin
    reward = (radius <= 3).float()  # 1 if in radius 3, else 0
    reward = reward * 2 - 1.  # r=1 if x in radius 3 else r=-1
    return reward


# ======= Configurable parameters ======= #
num_samples = 50000
num_epochs = 0
means = [[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, -5.0]]
stds = [0.5, 1., 1., 1., 1.]
ratios = [0.002, 0.2495, 0.2495, 0.2495, 0.2495]
device = 'cpu'


# ======= Generate synthetic data ======= #
def generate_data(num_samples, means, stds, ratios):
    component_sizes = [int(r * num_samples) for r in ratios]
    data = []
    for i in range(len(means)):
        samples = torch.randn(component_sizes[i], 2) * stds[i] + torch.tensor(means[i])
        data.append(samples)
    return torch.cat(data).to(device)


if __name__ == '__main__':
    data = generate_data(num_samples, means, stds, ratios)

    # ======= Define the GMM Model ======= #
    model = GMM(K=len(means), dim=2).to(device)

    # ======= Save the model ======= #
    torch.save(model.state_dict(), 'gmm_model.pth')

    # ======= Plot the GMM components ======= #
    plot_policy(model, title_addon='Initial GMM Model\n')
