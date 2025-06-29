from gmm import GMM, single_mode_reward
import torch
import os
import copy
from typing import Callable, Optional
from rl_loss import RLLoss
from utils import custom_plot_policy, plot_combined_metrics
import torch.nn.functional as F



def rft(n_of_policy_updates: int, 
        update_points: list[int],
        device: torch.device,
        n_samples: int,
        n_epochs: int,
        reward_modification_f: Optional[Callable] = None,
        ):
    
    lr = 1e-2
    batch_size = 3
    eps = 0.2
    saved_models = {}
    
    metrics = {
        'epoch_indices': [],          
        'central_probs_epochs': [],    
        'central_stds_epochs': [],     
    }
    
    base_policy = GMM(K=5, dim=2).to(device)
    base_policy.load_state_dict(torch.load('gmm_model.pth'))

    loss_fn = RLLoss(clipping_coef=eps, is_ppo=True)
    
    epoch_counter = 0  
    
    for policy_update in range(n_of_policy_updates + 1):
        if policy_update in update_points:
            saved_models[policy_update] = copy.deepcopy(base_policy.state_dict())
        
        if policy_update == n_of_policy_updates:
            break

        with torch.no_grad():
            samples = base_policy.sample(n_samples)
            rewards = single_mode_reward(samples)
            if reward_modification_f is not None:
                rewards = reward_modification_f(rewards)
            log_base_policy_probs = base_policy.log_prob(samples)

        data_set = torch.utils.data.TensorDataset(samples.to(device), rewards.to(device), log_base_policy_probs.to(device))
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

        paramaterized_policy = GMM(K=5, dim=2).to(device)
        paramaterized_policy.load_state_dict(base_policy.state_dict())

        optimizer = torch.optim.Adam(paramaterized_policy.parameters(), lr=lr)

        paramaterized_policy.train()
        for epoch in range(n_epochs):
            for samples_batch, rewards_batch, log_base_policy_probs_batch in data_loader:
                optimizer.zero_grad()
                log_paramaterized_policy_probs = paramaterized_policy.log_prob(samples_batch)
                loss = loss_fn(rewards_batch, log_paramaterized_policy_probs, log_base_policy_probs_batch)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                central_prob = F.softmax(paramaterized_policy.logits)[0].item()
                central_std = paramaterized_policy.std()[0].item()
                
                metrics['central_probs_epochs'].append(central_prob)
                metrics['central_stds_epochs'].append(central_std)
                metrics['epoch_indices'].append(epoch_counter)
                
                epoch_counter += 1

        base_policy.load_state_dict(paramaterized_policy.state_dict())

    return saved_models, metrics
    

def Q_5_2(device):
    number_of_gussians = 5
    samples_dim = 2
    model = GMM(K=number_of_gussians, dim=samples_dim).to(device)
    model.load_state_dict(torch.load('gmm_model.pth'))
    samples = model.sample(100)
    rewards = single_mode_reward(samples)
    print("positive rewards: ", (rewards > 0).sum())

def Q_5_3(device):
    def reward_modification_f(rewards: torch.Tensor) -> torch.Tensor:
        return torch.where(rewards < 0, torch.tensor(-1.0), torch.tensor(0.0))
    
    saved_models, metrics = rft(n_of_policy_updates=20, update_points=[0, 4, 8, 10, 12, 16, 20], device=device, n_samples=10, n_epochs=100, reward_modification_f=reward_modification_f)
    plots_folder = "5_3_plots"
    
    print(f"Saving plots to {plots_folder}/ folder...")
    for policy_update, model_state_dict in saved_models.items():
        model = GMM(K=5, dim=2).to(device)
        model.load_state_dict(model_state_dict)
        
        filename = f"policy_update_{policy_update}.png"
        save_path = os.path.join(plots_folder, filename)
        
        custom_plot_policy(
            model, 
            title=f'Policy Update {policy_update}',
            num_samples=2000,
            save_path=save_path
        )
    
    print(f"All plots saved to {plots_folder}/ folder!")
    
    # Calculate expected rewards using 2000 samples
    update_indices, expected_rewards = calculate_expected_rewards(
        saved_models, device, reward_modification_f, n_eval_samples=2000
    )
    
    return metrics, update_indices, expected_rewards 

def Q_5_4(device):
    saved_models, metrics = rft(n_of_policy_updates=20, update_points=[0, 4, 8, 10, 12, 16, 20], device=device, n_samples=10, n_epochs=100)

    plots_folder = "5_4_plots"
    
    print(f"Saving plots to {plots_folder}/ folder...")
    for policy_update, model_state_dict in saved_models.items():
        model = GMM(K=5, dim=2).to(device)
        model.load_state_dict(model_state_dict)
        
        filename = f"policy_update_{policy_update}.png"
        save_path = os.path.join(plots_folder, filename)
        
        custom_plot_policy(
            model, 
            title=f'Policy Update {policy_update}',
            num_samples=2000,
            save_path=save_path
        )
    
    print(f"All plots saved to {plots_folder}/ folder!")
    
    update_indices, expected_rewards = calculate_expected_rewards(
        saved_models, device, reward_modification_f=None, n_eval_samples=2000
    )
    
    return metrics, update_indices, expected_rewards


def calculate_expected_rewards(saved_models, device, reward_modification_f=None, n_eval_samples=2000):
    update_indices = []
    expected_rewards = []
    
    for policy_update in sorted(saved_models.keys()):
        model = GMM(K=5, dim=2).to(device)
        model.load_state_dict(saved_models[policy_update])
        
        with torch.no_grad():
            samples = model.sample(n_eval_samples)
            rewards = single_mode_reward(samples)
            if reward_modification_f is not None:
                rewards = reward_modification_f(rewards)
            expected_reward = rewards.mean().item()
            
            update_indices.append(policy_update)
            expected_rewards.append(expected_reward)
    
    return update_indices, expected_rewards


def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Running Q_5_3...")
    metrics_5_3, update_indices_5_3, expected_rewards_5_3 = Q_5_3(device)
    
    print("Running Q_5_4...")
    metrics_5_4, update_indices_5_4, expected_rewards_5_4 = Q_5_4(device)
    
    plot_combined_metrics(
        metrics_5_3, update_indices_5_3, expected_rewards_5_3,
        metrics_5_4, update_indices_5_4, expected_rewards_5_4,
        save_prefix="Q_5_4_metrics/combined_metrics"
    )
    
        

if __name__ == "__main__":
    main()