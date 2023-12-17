import torch
import numpy as np
from torch.autograd import Variable

from model.utils import *
from memory.trpo_memory import TRPOMemory
from model.actor import Actor
from model.critic import Critic
from model.kl import KL


class Agent:
    def __init__(self, n_actions, input_dim, gamma=0.99, gae_lambda=0.9, policy_clip=0.1, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = init_model(Actor(input_dim, n_actions))
        self.critic = init_model(Critic(input_dim))
        self.memory = TRPOMemory(batch_size)
        self.kl_div = KL(0.01)

    def remember(self, state, action, probs_, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs_, probs, vals, reward, done)

    def save_models(self):
        print("Saving Models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("Loading Models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        self.actor.eval()
        self.critic.eval()
        state = torch.tensor(np.array([observation]), device=self.actor.device, dtype=torch.float32)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs_ = dist.probs
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, torch.squeeze(probs_), probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, probs_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.sample_batch()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage, device=self.actor.device)
            values = torch.tensor(values, device=self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32, device=self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], device=self.actor.device, requires_grad=True)
                probs_ = torch.tensor(probs_arr[batch], device=self.actor.device, requires_grad=True)
                actions = torch.tensor(action_arr[batch], device=self.actor.device)

                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * \
                                         advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                kl_loss = self.kl_div(dist.probs, probs_)
                parameters = list(self.actor.parameters())
                grad_ = flat_grad(actor_loss, parameters, retain_graph=True)
                tol_kl = flat_grad(kl_loss, parameters, create_graph=True)

                def hessian_vector_product(vector):
                    return flat_grad(tol_kl @ vector, parameters, retain_graph=True)

                search_dir = conjugate_gradient(hessian_vector_product, grad_)
                max_length = torch.sqrt_(2 * self.kl_div.max_d_kl / (search_dir @ hessian_vector_product(search_dir)))
                max_step = max_length * search_dir

                def criterion(step):
                    apply_update(step, self.actor)
                    with torch.no_grad():
                        dist_ = self.actor(states)
                        new_probs_ = dist_.log_prob(actions)
                        prob_ratio_ = (new_probs_ - old_probs).exp()
                        weighted_probs_ = advantage[batch] * prob_ratio_
                        weighted_clipped_probs_ = torch.clamp(prob_ratio_, 1 - self.policy_clip, 1 + self.policy_clip) * \
                                                 advantage[batch]
                        actor_loss_new = -torch.min(weighted_probs_, weighted_clipped_probs_).mean()
                        kl_loss_new = self.kl_div(dist.probs, probs_)

                    loss_improvement = actor_loss_new - actor_loss

                    if loss_improvement < 0 and kl_loss_new <= self.kl_div.max_d_kl:
                        return True

                    apply_update(-step, self.actor)
                    return False

                idx = 0
                while not criterion((0.9 ** idx) * max_step) and idx < 10:
                    idx += 1

        self.memory.clear_memory()
