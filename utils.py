import numpy as np


vec = np.load("data/vec.npy")
# label = np.load("data/label.npy")[20255:]
print(vec.shape)
# print(label.shape)



# import os
# import shutil
# import numpy as np
#
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# import yaml
#
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# def save_config_file(model_checkpoints_folder, args):
#     if not os.path.exists(model_checkpoints_folder):
#         os.makedirs(model_checkpoints_folder)
#         with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
#             yaml.dump(args, outfile, default_flow_style=False)
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#
# def info_nce(features):
#     labels = torch.cat([torch.arange(256) for i in range(2)], dim=0)
#     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#     labels = labels.cuda()
#     features = F.normalize(features, dim=1)
#     similarity_matrix = torch.matmul(features, features.T)
#     mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
#     labels = labels[~mask].view(labels.shape[0], -1)
#     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
#     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
#     logits = torch.cat([positives, negatives], dim=1) / 0.07
#     return logits
#
#
# def search_rc_policy(train_loader, model, rc_agent):
#     for idx, (inputs, _) in enumerate(train_loader):
#         if idx >= 50:  # train enough episodes
#             break
#         batch_size = 256
#         inputs = torch.cat(inputs, dim=0)
#         with torch.no_grad():
#             features = model(inputs)
#             states = features  # tensor [batch_size, feature_dim]
#             base_loss = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 base_loss = torch.cat((base_loss, info_nce(features)), dim=0)
#             base_loss = base_loss.unsqueeze(1)
#             print("check base loss", base_loss)
#         for step in range(5):
#             actions = rc_agent.select_action(states)  # tensor [batch_size, action_dim]
#
#             next_inputs = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 image = transforms.functional.resized_crop(inputs[i],
#                                                            actions[i][0].item(), actions[i][1].item(),
#                                                            actions[i][2].item(), actions[i][3].item(),
#                                                            [224, 224])
#                 next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)
#
#             with torch.no_grad():
#                 feat_l, feat_ab = model(next_inputs)
#                 next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
#                 out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
#                 next_loss = torch.Tensor([]).cuda()
#                 for i in range(batch_size):
#                     next_loss = torch.cat(
#                         (next_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
#                 next_loss = next_loss.unsqueeze(1)
#             # print("check loss", next_loss)
#             reward = 1 / (abs(next_loss - base_loss + args.max_range) + args.epsilon)
#             # print("check reward", reward)
#
#             if step == 5 - 1:
#                 done = torch.zeros((batch_size, 1)).cuda()
#             else:
#                 done = torch.ones((batch_size, 1)).cuda()
#             rc_agent.buffer.reward = torch.cat((rc_agent.buffer.reward, reward), dim=0)
#             rc_agent.buffer.done = torch.cat((rc_agent.buffer.done, done), dim=0)
#
#             states = next_states
#             # last_loss = next_loss
#         rc_agent.learn()
#
#
# def search_hf_policy(train_loader, model, hf_agent):
#     rewards = []
#     for idx, (inputs, _, indexes) in enumerate(train_loader):
#         if idx >= 50:  # train enough episodes
#             break
#         batch_size = args.batch_size
#         inputs = inputs.float()
#         if torch.cuda.is_available():
#             indexes = indexes.cuda()
#             inputs = inputs.cuda()
#
#         episode_reward = torch.zeros((batch_size, 1)).cuda()
#
#         # env.reset()
#         with torch.no_grad():
#             feat_l, feat_ab = model(inputs)
#             states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
#             out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
#             base_loss = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 base_loss = torch.cat(
#                     (base_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
#             base_loss = base_loss.unsqueeze(1)
#         for step in range(5):
#             actions = hf_agent.select_action(states)  # tensor [batch_size, action_dim]
#
#             next_inputs = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 image = transforms.transforms.RandomHorizontalFlip(actions[i].item())(inputs[i])
#                 next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)
#
#             with torch.no_grad():
#                 feat_l, feat_ab = model(next_inputs)
#                 next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
#                 out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
#                 next_loss = torch.Tensor([]).cuda()
#                 for i in range(batch_size):
#                     next_loss = torch.cat(
#                         (next_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
#                 next_loss = next_loss.unsqueeze(1)
#             # print(next_loss)
#             reward = 1 / (abs(next_loss - base_loss + args.max_range) + args.epsilon)
#             # print("check reward", reward)
#
#             if step == 5 - 1:
#                 done = torch.zeros((batch_size, 1)).cuda()
#             else:
#                 done = torch.ones((batch_size, 1)).cuda()
#             hf_agent.buffer.reward = torch.cat((hf_agent.buffer.reward, reward), dim=0)
#             hf_agent.buffer.done = torch.cat((hf_agent.buffer.done, done), dim=0)
#             episode_reward += reward
#
#             states = next_states
#             # last_loss = next_loss
#         hf_agent.learn()
#         rewards.append(episode_reward.cpu().mean(dim=0).numpy())
#     return np.mean(rewards)
#
#
# def search_main_policy(train_loader, model, policy_agent, rc_agent, hf_agent):
#     rewards = []
#     for idx, (inputs, _, indexes) in enumerate(train_loader):
#         if idx >= 50:  # train enough episodes
#             break
#         batch_size = 256
#         inputs = inputs.float()
#         if torch.cuda.is_available():
#             indexes = indexes.cuda()
#             inputs = inputs.cuda()
#
#         episode_reward = torch.zeros((batch_size, 1)).cuda()
#
#         # env.reset()
#         with torch.no_grad():
#             feat_l, feat_ab = model(inputs)
#             states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
#             out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
#             base_loss = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 base_loss = torch.cat(
#                     (base_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
#             print(base_loss[10])
#             base_loss = base_loss.unsqueeze(1)
#
#         for step in range(5):
#             print("-------step {}--------".format(step))
#             actions = policy_agent.select_action(states)  # tensor [batch_size, action_dim]
#             print(actions[:10])
#             next_inputs = torch.Tensor([]).cuda()
#             for i in range(batch_size):
#                 # print(actions[i].item())
#                 if actions[i].item() == 0:
#                     sub_policy = rc_agent.select_action(states[i].unsqueeze(0),
#                                                         False)  # tensor [batch_size, action_dim]
#                     if i <= 10:
#                         print(sub_policy.cpu().numpy())
#                     image = transforms.functional.resized_crop(inputs[i],
#                                                                sub_policy[0].item(), sub_policy[1].item(),
#                                                                sub_policy[2].item(), sub_policy[3].item(),
#                                                                [224, 224])
#                 elif actions[i].item() == 1:
#                     sub_policy = hf_agent.select_action(states[i].unsqueeze(0), False)  # tensor [batch_size, n_actions]
#                     if i <= 10:
#                         print(sub_policy.item())
#                     image = transforms.RandomHorizontalFlip(sub_policy.item())(inputs[i])
#
#                 next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)
#
#             with torch.no_grad():
#                 feat_l, feat_ab = model(next_inputs)
#                 next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
#                 out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
#                 next_loss = torch.Tensor([]).cuda()
#                 for i in range(batch_size):
#                     next_loss = torch.cat(
#                         (next_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
#                 next_loss = next_loss.unsqueeze(1)
#             reward = 1 / (abs(next_loss - base_loss + args.max_range) + args.epsilon)
#             print("check reward", reward[:10], next_loss[:10])
#
#             if step == 5 - 1:
#                 done = torch.zeros((batch_size, 1)).cuda()
#             else:
#                 done = torch.ones((batch_size, 1)).cuda()
#             policy_agent.buffer.reward = torch.cat((policy_agent.buffer.reward, reward), dim=0)
#             policy_agent.buffer.done = torch.cat((policy_agent.buffer.done, done), dim=0)
#             episode_reward += reward
#
#             states = next_states
#             # last_loss = next_loss
#
#         print(indexes)
#         policy_agent.learn()
#         rewards.append(episode_reward.cpu().mean(dim=0).numpy())
#     return np.mean(rewards)
#
#
# def generate_batch(batch, model, rc_agent, hf_agent, policy_agent):
#     outputs = torch.Tensor([]).cuda()
#     with torch.no_grad():
#         features = model(batch)
#         actions = policy_agent.select_action(features, False)
#         for i in range(batch.shape[0]):
#             if actions[i].item() == 0:
#                 sub_policy = rc_agent.select_action(features[i].unsqueeze(0), False)  # tensor [batch_size, action_dim]
#                 image = transforms.functional.resized_crop(batch[i],
#                                                            sub_policy[0].item(), sub_policy[1].item(),
#                                                            sub_policy[2].item(), sub_policy[3].item(),
#                                                            [224, 224])
#             elif actions[i].item() == 1:
#                 sub_policy = hf_agent.select_action(features[i].unsqueeze(0), False)  # tensor [batch_size, n_actions]
#                 image = transforms.RandomHorizontalFlip(sub_policy.item())(batch[i])
#             outputs = torch.cat((outputs, image.unsqueeze(0)), dim=0)
#     return outputs
#
#
# def search_policy(train_loader, model, policy_agent, rc_agent, hf_agent):
#     model.eval()
#     policy_agent.model.train()
#     rc_agent.model.train()
#     hf_agent.model.train()
#     search_rc_policy(train_loader, model, rc_agent)
#     search_hf_policy(train_loader, model, hf_agent)
#     search_main_policy(train_loader, model, policy_agent, rc_agent, hf_agent)
#     policy_agent.model.eval()
#     rc_agent.model.eval()
#     hf_agent.model.eval()
#     model.train()
