import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os


def plot_combined_metrics(metrics_5_3, update_indices_5_3, expected_rewards_5_3,
                         metrics_5_4, update_indices_5_4, expected_rewards_5_4,
                         save_prefix=""):
    
    if save_prefix:
        import os
        directory = os.path.dirname(save_prefix)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_5_3['epoch_indices'], metrics_5_3['central_probs_epochs'], 
             'b-', linewidth=2, alpha=0.8, label='Q_5_3 (Modified Rewards)')
    plt.plot(metrics_5_4['epoch_indices'], metrics_5_4['central_probs_epochs'], 
             'r-', linewidth=2, alpha=0.8, label='Q_5_4 (Standard Rewards)')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title('Central Gaussian Probability Along Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_central_probabilities.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_prefix}_central_probabilities.png")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_5_3['epoch_indices'], metrics_5_3['central_stds_epochs'], 
             'b-', linewidth=2, alpha=0.8, label='Q_5_3 (Modified Rewards)')
    plt.plot(metrics_5_4['epoch_indices'], metrics_5_4['central_stds_epochs'], 
             'r-', linewidth=2, alpha=0.8, label='Q_5_4 (Standard Rewards)')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.title('Central Gaussian Standard Deviation Along Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_central_std.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_prefix}_central_std.png")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(update_indices_5_3, expected_rewards_5_3, 
             'b-o', linewidth=2, markersize=6, alpha=0.8, label='Q_5_3 (Modified Rewards)')
    plt.plot(update_indices_5_4, expected_rewards_5_4, 
             'r-o', linewidth=2, markersize=6, alpha=0.8, label='Q_5_4 (Standard Rewards)')
    plt.xlabel('Policy Update')
    plt.ylabel('Expected Reward')
    plt.title('Expected Reward Along Policy Updates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_expected_reward.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_prefix}_expected_reward.png")
    plt.show()


def custom_plot_policy(policy, title=None, num_samples=2000, save_path=None):
    with torch.no_grad():
        plt.figure(figsize=(6, 6))
        x_samples = policy.sample(num_samples).cpu()
        plt.scatter(x_samples[:, 0], x_samples[:, 1], s=2, alpha=0.2)

        g_probs = F.softmax(policy.logits).detach().cpu()
        for i in range(policy.K):
            cur_color = f"C{i}"
            circle = Circle(policy.mu[i].cpu(), 2 * policy.std()[i].cpu().item(), color=cur_color, fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            
            plt.scatter(*policy.mu[i].cpu(), color=cur_color, s=100, marker='x')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        
        if title is not None:
            plt.title(title)
        else:
            gaussians_probs = [float(np.round(x, 2)) for x in F.softmax(policy.logits, dim=0).cpu().numpy()]
            centric_gaussian_mean = np.round(policy.mu[0].cpu().numpy(), 2).tolist()
            centric_gaussian_std = np.round(policy.log_std.exp()[0].cpu().numpy(), 2).tolist()
            plt.title(f'Probs for Gaussians: {gaussians_probs}\n'
                      f'G_0 Mean: {centric_gaussian_mean}\nG_0 Std: {centric_gaussian_std}')
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() 
            print(f"Plot saved to: {save_path}")
        else:
            plt.show() 