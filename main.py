import numpy as np

from models.singlepass import SinglePass
from models.singlepass_env import SinglePass as SinglePassEnv
from models.ppo_agent import PPO


# env module
# this module is for monitering the generation of data block
# NOTE: this is only for reinforcement learning module testing
# it should be replaced by the real data blocks in experiment projects
all_data = np.load("data/vec.npy")
all_label = np.load("data/label.npy")[20255:]
data_size = all_data.shape[0]
label_size = all_label.shape[0]
print("n samples:", data_size, label_size)
block_size = 256
block_num = int(data_size/block_size)
print("block number:", block_num)
all_label = list(all_label)

blocks_data = []
blocks_label = []
for i in range(block_num):
    blocks_data.append(all_data[0: block_size*(i+1)])
    blocks_label.append(all_label[0: block_size*(i+1)])
# env end

# SinglePass with PPO
agent = PPO(5, 1, action_std_init=0.6, continuous=True)
first_time = True
for idx, (data, label) in enumerate(zip(blocks_data, blocks_label)):
    print("blocks:", idx, data.shape, len(label))
    if first_time:
        first_time = False
        continue
    # sp_env = SinglePassEnv(0.6, data, 1, label, 256)
    sp_sim = SinglePass(0.6, data, 1, label, 256, agent, sim=True)
    sp = SinglePass(0.6, data, 1, label, 256, agent, sim=False)
    print("done once")
# cluster begin
# state_dim =
# agent = PPO()
# for block in blocks:
#     sp = SinglePass(0.6, all_data, 1, label, 256)
