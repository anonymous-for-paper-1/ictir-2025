import torch
import itertools
import numpy as np
import scipy.sparse as sp

from torch import nn
from torch import optim

import networkx as nx

from util.loss import BPRLoss
from util.loss import EmbLoss
from util.util import xavier_uniform_initialization

from tqdm import tqdm
from loguru import logger

from trainer.Trainer import Trainer


class GPPR(nn.Module):

    def __init__(self, config, dataset):
        super().__init__()

        self.config = config

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.user_id_name = config["user_id_name"]
        self.item_id_name = config["item_id_name"]

        self.user_idx_name = config["user_idx_name"]
        self.item_idx_name = config["item_idx_name"]
        self.neg_item_idx_name = config["neg_item_idx_name"]

        self.early_stop_num = config["early_stop_num"]
        self.gpu = config["gpu"]

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.alpha_list = config["alpha_list"]
        self.alpha_list = list(map(lambda x: x / sum(self.alpha_list), self.alpha_list))  # 正規化

        # load dataset info
        self.interaction_matrix = self.dataset.get_train_coo_matrix()

        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]
        self.reg_weight = config["reg_weight"]

        assert len(self.alpha_list) == self.n_layers + 1

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        ########################################################
        # APPRの計算
        self.model_name = config["model"]
        self.dataset_name = config["dataset"]

        # weight
        self.beta = config["beta"]
        self.alpha_0 = config["alpha_0"]
        self.alpha_1 = config["alpha_1"]
        self.result_dir = f"./result/{self.model_name}/"

        self.norm_adj_matrix = self.norm_adj_matrix.to_dense()

        # 最後のA_pprが新規の項目
        # 隣接行列の計算後は gppr_1 と命名する
        self.gppr_1_norm_adj_matrix = torch.inverse(
            (torch.eye(self.norm_adj_matrix.shape[0]).to(self.device) - self.beta * self.norm_adj_matrix)
        ) @ (
            self.alpha_0 * torch.eye(self.norm_adj_matrix.shape[0]).to(self.device)
            + self.alpha_1 * self.norm_adj_matrix
        )

        self.gppr_1_norm_adj_matrix = self.gppr_1_norm_adj_matrix / self.gppr_1_norm_adj_matrix.sum(
            axis=0, keepdims=True
        )

        # A_hatとA_apprの差分のノルムの計算
        self.norm_adj_matrix_diff_norm = torch.norm(
            (self.norm_adj_matrix / self.norm_adj_matrix.sum())
            - (self.gppr_1_norm_adj_matrix / self.gppr_1_norm_adj_matrix.sum())
        ).item()

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # optimizer
        self.set_optimizer()

    def get_dense_A(self):
        ######################################################
        # 再度Aの計算
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))
        # torch 2対応
        # DOK行列に値をセット
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # torch 1 ダメ
        # A._update(data_dict)

        A = sp.coo_matrix(A)
        row = A.row
        col = A.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(A.data)
        SparseA = torch.sparse.FloatTensor(i, data, torch.Size(A.shape))

        return SparseA.to_dense().to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))

        # torch 2対応
        # DOK行列に値をセット
        for (row, col), value in data_dict.items():
            A[row, col] = value

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [self.alpha_list[0] * all_embeddings]

        for layer_idx, alpha in zip(range(self.n_layers), self.alpha_list[1:]):
            all_embeddings = torch.sparse.mm(self.gppr_1_norm_adj_matrix, all_embeddings)
            embeddings_list.append(alpha * all_embeddings)

        ################################################################
        # 20230416: dim=0にしてtorch.stackを利用してtensorにする
        temp = torch.stack(embeddings_list, dim=0)

        # alphaで加重平均を取っているので，ここでは和を取る
        lightgcn_all_embeddings = torch.sum(temp, dim=0)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def get_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.user_idx_name]
        pos_item = interaction[self.item_idx_name]
        neg_item = interaction[self.neg_item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.user_idx_name]
        item = interaction[self.item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def all_user_predict(self, interaction):
        user = interaction[self.user_idx_name]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        # self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_weight)

    def fit(self):
        trainer = Trainer(self.config, self, self.dataset)
        self.best_valid_result, self.test_result = trainer.train()

    def calculate_gcn_diversity(self):
        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        # 解析
        score = torch.matmul(user_embedding, item_embedding.T)
        sorted_list, idx = torch.sort(score, dim=1)

        idx = torch.fliplr(idx)
        sorted_list = torch.fliplr(sorted_list)

        cdist_top_5_list = []
        cdist_middle_5_list = []
        cdist_middle_1_5_list = []
        cdist_middle_2_5_list = []
        cdist_middle_3_5_list = []
        cdist_middle_4_5_list = []
        cdist_middle_5_5_list = []

        combinations_top_5_list = []
        combinations_middle_5_list = []
        combinations_middle_1_5_list = []
        combinations_middle_2_5_list = []
        combinations_middle_3_5_list = []
        combinations_middle_4_5_list = []
        combinations_middle_5_5_list = []

        for i, e in enumerate(user_embedding):
            top_5 = item_embedding[idx[i, 0:5]]
            middle_5 = item_embedding[idx[i, 25:30]]
            middle_1_5 = item_embedding[idx[i, 50:55]]
            middle_2_5 = item_embedding[idx[i, 75:80]]
            middle_3_5 = item_embedding[idx[i, 100:105]]
            middle_4_5 = item_embedding[idx[i, 150:155]]
            middle_5_5 = item_embedding[idx[i, 250:255]]

            top_5_mean = torch.mean(top_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_mean = torch.mean(middle_5, dim=0).reshape(1, -1).to(torch.float)
            middle_1_5_mean = torch.mean(middle_1_5, dim=0).reshape(1, -1).to(torch.float)
            middle_2_5_mean = torch.mean(middle_2_5, dim=0).reshape(1, -1).to(torch.float)
            middle_3_5_mean = torch.mean(middle_3_5, dim=0).reshape(1, -1).to(torch.float)
            middle_4_5_mean = torch.mean(middle_4_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_5_mean = torch.mean(middle_5_5, dim=0).reshape(1, -1).to(torch.float)

            _cdist_top_5 = torch.cdist(top_5, top_5_mean).mean()
            _cdist_middle_5 = torch.cdist(middle_5, middle_5_mean).mean()
            _cdist_middle_1_5 = torch.cdist(middle_1_5, middle_1_5_mean).mean()
            _cdist_middle_2_5 = torch.cdist(middle_2_5, middle_2_5_mean).mean()
            _cdist_middle_3_5 = torch.cdist(middle_3_5, middle_3_5_mean).mean()
            _cdist_middle_4_5 = torch.cdist(middle_4_5, middle_4_5_mean).mean()
            _cdist_middle_5_5 = torch.cdist(middle_5_5, middle_5_5_mean).mean()

            cdist_top_5_list.append(_cdist_top_5.item())
            cdist_middle_5_list.append(_cdist_middle_5.item())
            cdist_middle_1_5_list.append(_cdist_middle_1_5.item())
            cdist_middle_2_5_list.append(_cdist_middle_2_5.item())
            cdist_middle_3_5_list.append(_cdist_middle_3_5.item())
            cdist_middle_4_5_list.append(_cdist_middle_4_5.item())
            cdist_middle_5_5_list.append(_cdist_middle_5_5.item())

            ## combination
            def get_combinations_norm(mat):
                temp = list(itertools.combinations(mat, 2))
                temp_list = []
                for i in temp:
                    temp_list.append(torch.dist(i[0], i[1]).item())
                return np.mean(temp_list)

            combinations_top_5_list.append(get_combinations_norm(top_5))
            combinations_middle_5_list.append(get_combinations_norm(middle_5))
            combinations_middle_1_5_list.append(get_combinations_norm(middle_1_5))
            combinations_middle_2_5_list.append(get_combinations_norm(middle_2_5))
            combinations_middle_3_5_list.append(get_combinations_norm(middle_3_5))
            combinations_middle_4_5_list.append(get_combinations_norm(middle_4_5))
            combinations_middle_5_5_list.append(get_combinations_norm(middle_5_5))

        self.cdist_top_5_list = float(np.mean(cdist_top_5_list))
        self.cdist_middle_5_list = float(np.mean(cdist_middle_5_list))
        self.cdist_middle_1_5_list = float(np.mean(cdist_middle_1_5_list))
        self.cdist_middle_2_5_list = float(np.mean(cdist_middle_2_5_list))
        self.cdist_middle_3_5_list = float(np.mean(cdist_middle_3_5_list))
        self.cdist_middle_4_5_list = float(np.mean(cdist_middle_4_5_list))
        self.cdist_middle_5_5_list = float(np.mean(cdist_middle_5_5_list))
        self.combinations_top_5_list = float(np.mean(combinations_top_5_list))
        self.combinations_middle_5_list = float(np.mean(combinations_middle_5_list))
        self.combinations_middle_1_5_list = float(np.mean(combinations_middle_1_5_list))
        self.combinations_middle_2_5_list = float(np.mean(combinations_middle_2_5_list))
        self.combinations_middle_3_5_list = float(np.mean(combinations_middle_3_5_list))
        self.combinations_middle_4_5_list = float(np.mean(combinations_middle_4_5_list))
        self.combinations_middle_5_5_list = float(np.mean(combinations_middle_5_5_list))
