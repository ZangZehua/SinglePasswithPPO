# coding:utf-8

import numpy as np
import os
import torch.nn.functional as F
import torch

from models.ppo_agent import PPO


class SinglePass:
    def __init__(self, sim_threshold, data):
        self.text_vec = None  #
        self.topic_serial = None
        self.topic_cnt = 0
        self.sim_threshold = sim_threshold
        self.centers = []  # the center of every cluster
        self.sums_per_cluster = []  # the number of vec of every cluster
        self.indexs_per_cluster = []  # the index of vec of every cluster
        self.agent = PPO(4, 1, action_std_init=0.6, continuous=True)
        self.warmup = 1000
        self.agent_learn_iter = 100
        self.cluster_result = self.run_cluster(data)  # clustering
        self.get_center()

    def clustering(self, sen_vec, idx):
        if self.topic_cnt == 0:
            self.text_vec = sen_vec
            self.topic_cnt += 1
            self.topic_serial = [self.topic_cnt]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial[np.argmax(sim_vec)]
            self.text_vec = np.vstack([self.text_vec, sen_vec])

            state = self.get_state()
            state = torch.Tensor(state).cuda().unsqueeze(0)

            action = self.agent.select_action(state)
            temp = True
            if max_value >= action:
                self.topic_serial.append(topic_ser)
            else:
                temp = False
                self.topic_cnt += 1
                self.topic_serial.append(self.topic_cnt)
            reward = self.get_reward(10)  # 10 is a test parm, need changing
            reward = torch.Tensor(reward).cuda().unsqueeze(0)
            self.agent.buffer.reward = torch.cat((self.agent.buffer.reward, reward), dim=0)
            done = torch.Tensor([False]).cuda().unsqueeze(0)
            self.agent.buffer.done = torch.cat((self.agent.buffer.done, done), dim=0)
            if idx / self.agent_learn_iter == 0:
                self.agent.learn()
            if idx < self.warmup:  # agent is warming up
                if temp:
                    del self.topic_serial[-1]
                else:
                    self.topic_cnt -= 1
                    del self.topic_serial[-1]
                # cluster by init sim_threshold
                if max_value >= self.sim_threshold:
                    self.topic_serial.append(topic_ser)
                else:
                    self.topic_cnt += 1
                    self.topic_serial.append(self.topic_cnt)

    def run_cluster(self, data):
        sums = data.shape[0]
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        i = 0
        for vec in data:
            if i % 1000 == 0:
                print(i)
            i = i + 1
            self.clustering(vec, i)
        return self.topic_serial[len(self.topic_serial) - sums:]

    def get_center(self):
        self.centers = []
        self.sums_per_cluster = []
        self.indexs_per_cluster = []
        for i in range(max(self.topic_serial)):
            tmp_topic = i + 1
            sums = 0
            tmp_indexs_text = []
            indexs = [False] * self.text_vec.shape[0]
            for j in range(len(indexs)):
                if self.topic_serial[j] == tmp_topic:
                    indexs[j] = True
                    sums = sums + 1
                    tmp_indexs_text.append(j)
            center = np.mean(self.text_vec[indexs], 0).tolist()
            self.centers.append(center)
            self.sums_per_cluster.append(sums)
            self.indexs_per_cluster.append(tmp_indexs_text)

    def get_info_cluster(self):  # Get detailed clustering results
        res = []
        for i in range(len(self.indexs_per_cluster)):
            tmp_vec = []
            for j in range(len(self.indexs_per_cluster[i])):
                tmp_vec.append(self.text_vec[self.indexs_per_cluster[i][j]])
            tmp_vec = np.array(tmp_vec)
            res.append(tmp_vec)
        return res

    def get_state(self):  # get state of RL
        state_dict = {}
        state_dict['num_of_clusters'] = max(self.topic_serial)
        centers = np.array(self.centers)
        # calculate the neighbor distance
        neighbor_dists = np.dot(centers, centers.T)
        # absolute value
        neighbor_dists = np.maximum(neighbor_dists, -neighbor_dists)
        # minimum neighbor distance
        state_dict['min_neighbor_dist'] = neighbor_dists.min()
        state_dict['aver_sep_dist'] = (neighbor_dists.mean() * max(self.topic_serial) - 1) / max(self.topic_serial)
        coh_dists = 0
        info_of_cluster = self.get_info_cluster()
        for cluster in info_of_cluster:
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2
            tmp_vec = np.array(cluster)
            cohdist = np.dot(tmp_vec, tmp_vec.T)
            cohdist = np.maximum(cohdist, -cohdist)
            coh_dists = coh_dists + (cohdist.sum() - cluster.shape[0]) / (2 * sums)
        state_dict['aver_coh_dist'] = coh_dists / max(self.topic_serial)
        state = []
        for key in state_dict:
            state.append(state_dict[key])
        return state

    def get_reward(self, bs):  # get reward of RL
        if max(self.topic_serial) > bs:
            return 0
        info_of_cluster = self.get_info_cluster()

        # calculate the overall within-cluster variance
        internal_var = 0
        for cluster in info_of_cluster:  # calculate the within-cluster variance of every cluster
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2  # the number of distance
            tmp_vec = np.array(cluster)
            internal_dists = np.dot(tmp_vec, tmp_vec.T)
            internal_dists = np.maximum(internal_dists, -internal_dists).tolist()
            quadratic_sum = 0
            for i in range(len(internal_dists)):
                for j in range(len(internal_dists) - i - 1):
                    quadratic_sum = quadratic_sum + (internal_dists[i][j + i + 1]) ** 2
            # print(sums)
            internal_var = internal_var + quadratic_sum / sums
        internal_var = internal_var / max(self.topic_serial)

        # calculate the overall between-cluster variance
        centers = np.array(self.centers)
        external_dists = np.dot(centers, centers.T).tolist()
        if isinstance(external_dists, float):
            return 0
        else:
            quadratic_sum = 0
            for i in range(len(external_dists)):
                for j in range(len(external_dists) - i - 1):
                    quadratic_sum = quadratic_sum + (external_dists[i][j + i + 1]) ** 2
        external_var = quadratic_sum / (centers.shape[0] * (centers.shape[0] - 1) / 2)

        # calculate reward
        Nu = external_var / (max(self.topic_serial) - 1)
        De = internal_var / (self.text_vec.shape[0] - max(self.topic_serial))
        return Nu / De
