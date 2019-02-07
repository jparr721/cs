import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from collections import namedtuple
import ast
import sys
from policy_net import PolicyNet
import shutil

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

no_update = sys.argv[1].startswith("n")
if no_update:
    print("Not updating parameters.")
    exit(1)

data_path = 'data/batch_data'
cuda = torch.device('cuda')

print("Using ", torch.cuda.get_device_name(0))

policy_net = PolicyNet()
if DEVICE == 'cuda':
    policy_net.cuda()
optimizer = optim.Adam(policy_net.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

# Load state
policy_net.load_state()

# Preprocessing
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

savedactions = []
savedrewards = []
turns = []
sids = []
episodes = []
scores = []
num_episodes = 0

# Read data
raw_data = []
f = open(data_path, "r")
for line in f.readlines():
    data = line.strip().split("|")
    if len(data) == 1:
        scores.append(int(data[0]))
        num_episodes += 1
    else:
        raw_data.append(data)

mean_score = sum(scores)/float(num_episodes)
# Save old state with mean score
shutil.copyfile('model/state', 'saved/state_{}'.format(mean_score))


# Sort by ships, turns
raw_data = sorted(raw_data, key=lambda x: (int(x[0]), int(x[2]), int(x[1])))

for data in raw_data:
    episode, t, sid, reward, sample, state = data

    # Format typing
    sid = int(sid)
    episode = int(episode)
    t = int(t)
    reward = float(reward)
    if DEVICE == 'cuda':
        sample = torch.tensor(float(sample)).cuda()
    else:
        sample = torch.tensor(float(sample))
    state = ast.literal_eval(state)

    # Calculating
    if DEVICE == 'cuda':
        state = torch.FloatTensor(state).cuda()
    else:
        state = torch.FloatTensor(state)
    state = Variable(state)
    probs = policy_net(state)
    m = Categorical(probs)
    log_prob = m.log_prob(sample)

    # Log episode
    episodes.append(episode)

    # Log turn t
    turns.append(t)

    # Log ship id sid
    sids.append(sid)

    # Save reward for turn t
    savedrewards.append(reward)

    # Save action for turn t
    action = SavedAction(log_prob, policy_net.get_state_value(state))
    savedactions.append(action)

R = 0
policy_losses = []
value_losses = []
rewards = []
gamma = 0.5

last_t = 600
last_sid = -69
for (e, t, sid, r) in zip(
        episodes[::-1], turns[::-1], sids[::-1], savedrewards[::-1]):
    # Check for new episode
    if t > last_t or sid != last_sid:
        R = 0
    last_t = t
    last_sid = sid
    # Running reward
    R = r + gamma * R
    rewards.insert(0, R)
if DEVICE == 'cuda':
    rewards = torch.tensor(rewards).cuda()
else:
    rewards = torch.tensor(rewards)
rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
for (log_prob, value), r in zip(savedactions, rewards):
    reward = r - value.item()
    policy_losses.append(-log_prob * reward)
    if DEVICE == 'cuda':
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).cuda()))
        optimizer.zero_grad()
        loss = torch.stack(
                policy_losses) \
            .sum().cuda() + torch.stack(value_losses).sum().cuda()
    else:
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        optimizer.zero_grad()
        loss = torch.stack(
                policy_losses).sum() + torch.stack(value_losses).sum()
loss = loss/len(turns)

loss.backward()
optimizer.step()

# Save the updated model
policy_net.save_state()

# Reset batch_data file
open(data_path, 'w').close()
