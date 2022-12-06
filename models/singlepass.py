# coding:utf-8

import numpy as np
import os
import torch.nn.functional as F
import torch
import copy
from sklearn.metrics import silhouette_score


class SinglePass:
    def __init__(self, sim_threshold, data, flag, label, size, agent, sim=False):
        self.device = torch.device('cuda:0')
        self.text_vec = None  #
        self.topic_serial = None
        self.topic_cnt = 0
        self.sim_threshold = sim_threshold

        self.done_data = data[0:data.shape[0] - size]
        self.new_data = data[data.shape[0] - size:]
        self.done_label = label

        self.centers = []  # the center of every cluster
        self.sums_per_cluster = []  # the number of vec of every cluster
        self.indexs_per_cluster = []  # the index of vec of every cluster
        if flag == 1:
            self.pseudo_labels = self.run_cluster_init(0.6, size)
            self.pseudo_labels_tight = self.run_cluster_init(0.55, size)
            self.pseudo_labels_loose = self.run_cluster_init(0.65, size)

        self.agent = agent
        self.sim = sim
        if self.sim:
            self.cluster_result = self.run_cluster_sim(flag, size)  # clustering
        else:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
            self.get_center()
            state = self.get_state()
            state = torch.Tensor(state).cuda().unsqueeze(0)
            action = self.agent.select_action(state)
            self.sim_threshold = action.item()
            self.cluster_result = self.run_cluster(flag, size)  # clustering
            self.agent.buffer.clear_buffer()

    def clustering(self, sen_vec):
        if self.topic_cnt == 0:
            self.text_vec = sen_vec
            self.topic_cnt += 1
            self.topic_serial = [self.topic_cnt]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial[np.argmax(sim_vec)]
            self.text_vec = np.vstack([self.text_vec, sen_vec])

            if max_value >= self.sim_threshold:
                self.topic_serial.append(topic_ser)
            else:
                self.topic_cnt += 1
                self.topic_serial.append(self.topic_cnt)

    def run_cluster_sim(self, flag, size):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
            self.get_center()
        i = 0
        for vec in self.new_data:
            state = self.get_state()
            state = torch.Tensor(state).cuda().unsqueeze(0)

            if i % 1000 == 0:
                print(i)
            i = i + 1
            action = self.agent.select_action(state)
            self.sim_threshold = action.item()
            self.clustering(vec)
            reward = self.get_reward()
            self.agent.buffer.reward = torch.cat((self.agent.buffer.reward, torch.Tensor([reward]).cuda().unsqueeze(0)), dim=0)
            self.agent.buffer.done = torch.cat((self.agent.buffer.done, torch.Tensor([False]).cuda().unsqueeze(0)), dim=0)
        self.agent.buffer.done[-1] = torch.Tensor([True]).cuda()
        self.agent.learn()
        return self.topic_serial[len(self.topic_serial) - size:]

    def run_cluster(self, flag, size):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
            self.get_center()
        i = 0
        for vec in self.new_data:
            if i % 1000 == 0:
                print(i)
            i = i + 1
            self.clustering(vec)
        return self.topic_serial[len(self.topic_serial) - size:]

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

    def clustering_init(self, t, sen_vec):
        if self.topic_cnt_init == 0:
            self.text_vec_init = sen_vec
            self.topic_cnt_init += 1
            self.topic_serial_init = [self.topic_cnt_init]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec_init.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial_init[np.argmax(sim_vec)]
            self.text_vec_init = np.vstack([self.text_vec_init, sen_vec])

            if max_value >= t:
                self.topic_serial_init.append(topic_ser)
            else:
                self.topic_cnt_init += 1
                self.topic_serial_init.append(self.topic_cnt_init)

    def run_cluster_init(self, t, size):
        self.text_vec_init = []
        self.topic_serial_init = []
        self.topic_cnt_init = 0
        for vec in self.new_data:
            self.clustering_init(t, vec)
        return self.topic_serial_init

    def get_state(self):  # get state of RL
        state_dict = {}
        state_dict['num_of_clusters'] = max(self.topic_serial)
        centers = np.array(self.centers)
        # calculate the neighbor distance
        neighbor_dists = np.dot(centers, centers.T)
        # absolute value
        neighbor_dists = np.maximum(neighbor_dists, -neighbor_dists)
        # zzh: nan to 0
        neighbor_dists = np.nan_to_num(neighbor_dists, 0.0001)
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
            coh_dists = coh_dists + (cohdist.sum() - cluster.shape[0]) / (2 * sums + 0.0001)
        state_dict['aver_coh_dist'] = coh_dists / max(self.topic_serial)
        state = []
        for key in state_dict:
            state.append(state_dict[key])
        state.append(silhouette_score(self.new_data, self.pseudo_labels, metric='euclidean'))

        return state

    def get_reward(self):  # get reward of RL
        pseudo_labels = torch.tensor(self.pseudo_labels).to(self.device)
        pseudo_labels_tight = torch.tensor(self.pseudo_labels_tight).to(self.device)
        pseudo_labels_loose = torch.tensor(self.pseudo_labels_loose).to(self.device)
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float().to(self.device)
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float().to(
            self.device)
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float().to(
            self.device)
        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1).to(
            self.device)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1).to(
            self.device)
        del label_sim
        del label_sim_tight
        del label_sim_loose
        del pseudo_labels
        del pseudo_labels_tight
        del pseudo_labels_loose
        return int(R_indep.mean().item() + R_comp.mean().item()) + 0.0001
