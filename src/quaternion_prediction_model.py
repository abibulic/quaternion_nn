import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from dataset import QuaternionData
from graph_network import modules, graphs, utils_torch
from graph_network.blocks import broadcast_receiver_nodes_to_edges, broadcast_sender_nodes_to_edges, NodesToGlobalsAggregator

def list_files(root_dir, ext='.xyz'):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file).replace("\\","/"))
    return file_list


class CollectNeighboursAndEdgesToNodes(torch.nn.Module):
    def __init__(self):
        super(CollectNeighboursAndEdgesToNodes, self).__init__()
    def forward(self, graph):
        neighbours_final = []
        edges_final = []
        for n in range(graph.nodes.shape[0]):
            neighbours = []
            edges = []
            idx_n = graph.receivers[(graph.senders==n).nonzero().view(-1).tolist()].tolist() + graph.senders[(graph.receivers==n).nonzero().view(-1).tolist()].tolist()
            idx_e = (graph.senders==n).nonzero().view(-1).tolist() +(graph.receivers==n).nonzero().view(-1).tolist()
            for i in range(4): # 4 je max broj susjeda i veza
                if(i < len(idx_n)):
                    neighbours.append(graph.nodes[idx_n[i]])
                else:
                    neighbours.append(torch.zeros(graph.nodes[0].shape).to(torch.device('cuda')))
                if(i < len(idx_e)):
                    edges.append(graph.edges[idx_e[i]])
                else:
                    edges.append(torch.zeros(graph.edges[0].shape).to(torch.device('cuda')))

            neighbours_final.append(torch.cat(neighbours, axis=-1))
            edges_final.append(torch.cat(edges, axis=-1))
        return torch.cat(neighbours_final, axis=-1).view(graph.nodes.shape[0], -1), torch.cat(edges_final, axis=-1).view(graph.nodes.shape[0], -1)


class Permutator(nn.Module):
    def __init__(self, inpit_size):
        super(Permutator, self).__init__()

        self.model = torch.nn.Sequential(
                    torch.nn.Linear(inpit_size, 64),
                    #torch.nn.Dropout(),
                    torch.nn.SELU(),
                    torch.nn.Linear(64, 64),
                    #torch.nn.Dropout(),
                    torch.nn.SELU(),
                    torch.nn.Linear(64, inpit_size),
                    )

    def forward(self, data):
        return self.model(data)


class QuaternionPredictor(pl.LightningModule):
    def __init__(self, args):
        super(QuaternionPredictor, self).__init__()

        self.hparams = args
        
        # self.unit_vector = torch.cuda.FloatTensor([[0, 1, 1, 1]])
        # self.unit_vector.requires_grad = True

        # self.invert_vector = torch.cuda.FloatTensor([[1, -1, -1, -1]])
        # self.invert_vector.requires_grad = True

        # self.mask_real = torch.cuda.FloatTensor([[1, 0, 0, 0],
        #                                         [0, -1, 0, 0],
        #                                         [0, 0, -1, 0],
        #                                         [0, 0, 0, -1]])
        # self.mask_real.requires_grad = True

        # self.mask_i = torch.cuda.FloatTensor([[0, 1, 0, 0],
        #                                     [1, 0, 0, 0],
        #                                     [0, 0, 0, 1],
        #                                     [0, 0, -1, 0]])
        # self.mask_i.requires_grad = True

        # self.mask_j = torch.cuda.FloatTensor([[0, 0, 1, 0],
        #                                     [0, 0, 0, -1],
        #                                     [1, 0, 0, 0],
        #                                     [0, 1, 0, 0]])
        # self.mask_j.requires_grad = True

        # self.mask_k = torch.cuda.FloatTensor([[0, 0, 0, 1],
        #                                     [0, 0, 1, 0],
        #                                     [0, -1, 0, 0],
        #                                     [1, 0, 0, 0]])
        # self.mask_k.requires_grad = True

        self.permutate_nodes = Permutator(50)
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(57, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 57),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(57, 7),
                                )

        self.edges_aggregator = CollectNeighboursAndEdgesToNodes()

        self.permutate_neighbours = torch.nn.Sequential(
                                torch.nn.Linear(100, 128),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(128, 128),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(128, 100),
                                )

        self.permutate_edges = Permutator(28)
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(153, 100),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(100, 50),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(50, 25),
                                )

        self.predict = torch.nn.Sequential(
                                torch.nn.Linear(153, 100),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(100, 80),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(80, 60),
                                torch.nn.SELU(),
                                torch.nn.Linear(60, 40),
                                torch.nn.SELU(),
                                torch.nn.Linear(40, 20),
                                torch.nn.SELU(),
                                torch.nn.Linear(20, 10),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 4),
                                )
        
        self.aggregate_all = NodesToGlobalsAggregator()

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        #losses
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.l1_loss = torch.nn.L1Loss()
        self.cross_loss = torch.nn.CrossEntropyLoss()

    #TODO: dati intuitivnije ime funkciji
    def pred_real_sum(self, graph, pred, target, node_or_edge):
        pred_sum = []
        real_sum = []
        if(node_or_edge=='edge'):
            index_cumsum = torch.cumsum(graph.n_edge, dim=0)
        else:
            index_cumsum = torch.cumsum(graph.n_node, dim=0)
        remember_sum_pred = 0
        remember_sum_real = 0
        for idx in index_cumsum:
            pred_sum.append(torch.sum(pred[:idx])-remember_sum_pred)
            remember_sum_pred = torch.sum(pred[:idx])

            real_sum.append(torch.sum(target[:idx])-remember_sum_real)
            remember_sum_real = torch.sum(target[:idx])

        pred_sum = torch.stack(pred_sum)
        real_sum = torch.stack(real_sum)
        return pred_sum, real_sum


    def forward(self, graph, optimizer_idx):
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_pair_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_pair_nodes = self.permutate_nodes(collected_pair_nodes)

        # if optimizer_idx == 0:
        #     return self.pred_real_sum(graph, h1_pair_nodes, collected_pair_nodes, 'edge')

        h1_edges = graph.edges
        for _ in range(3):
            temp = torch.cat([h1_edges, h1_pair_nodes], axis=-1)
            h1_edges = self.update_edges(temp)
        
        neighbours_of_nodes, edges_of_nodes = self.edges_aggregator(graph.replace(edges=h1_edges))
        
        h_neighbours = self.permutate_neighbours(neighbours_of_nodes)
        h2_edges = self.permutate_edges(edges_of_nodes)

        # if optimizer_idx == 1:
        #     return self.pred_real_sum(graph, h_neighbours, neighbours_of_nodes, 'node')
        
        # if optimizer_idx == 2:
        #     return self.pred_real_sum(graph, h2_edges, edges_of_nodes, 'node')

        h2_nodes = graph.nodes
        for _ in range(3):
            temp2 = torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        collect_nodes_for_edges2 = []
        collect_nodes_for_edges2.append(broadcast_receiver_nodes_to_edges(graph.replace(nodes=h2_nodes)))
        collect_nodes_for_edges2.append(broadcast_sender_nodes_to_edges(graph.replace(nodes=h2_nodes)))
        collected_nodes2 = torch.cat(collect_nodes_for_edges2, axis=-1)
        
        gloabl = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)))

        out = self.predict(gloabl.to(torch.device('cuda')))

        # out0 = out*self.unit_vector.T
        # out_real = torch.sum(self.mask_real*out0)
        # out_i = torch.sum(self.mask_i*out0)
        # out_j = torch.sum(self.mask_j*out0)
        # out_k = torch.sum(self.mask_k*out0)

        # out2 = []
        # out2.append(torch.unsqueeze(out_real,0))
        # out2.append(torch.unsqueeze(out_i,0))
        # out2.append(torch.unsqueeze(out_j,0))
        # out2.append(torch.unsqueeze(out_k,0))
        # out2 = torch.cat(out2, dim=0)

        # out2 = out2*self.unit_vector.T
        # out_real2 = torch.sum(self.mask_real*out2)
        # out_i2 = torch.sum(self.mask_i*out2)
        # out_j2 = torch.sum(self.mask_j*out2)
        # out_k2 = torch.sum(self.mask_k*out2)

        # out3 = []
        # out3.append(torch.unsqueeze(out_real2,0))
        # out3.append(torch.unsqueeze(out_i2,0))
        # out3.append(torch.unsqueeze(out_j2,0))
        # out3.append(torch.unsqueeze(out_k2,0))
        # out3 = torch.cat(out3, dim=0)

        return out
        
    def prepare_data(self):
        
        files = list_files(self.hparams.data_path, ext='.npy')
        files = files[:20]
        #standardize data
        all_data = []
        for file in files:
            graph = np.load(file, allow_pickle=True).item()
            all_data.append(np.around(graph['globals'][0][:4], decimals=0))
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)

        train_data, valid_data = train_test_split(files, test_size=self.hparams.valid_split, random_state=0, shuffle=True)
        self.data_train = QuaternionData(train_data, self.hparams, mean, std)
        self.data_valid = QuaternionData(valid_data, self.hparams, mean, std)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.data_train,
                                               batch_size=self.hparams.batch_size, shuffle=True,
                                               collate_fn=QuaternionData.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.data_valid,
                                               batch_size=1, shuffle=False,
                                               collate_fn=QuaternionData.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.data_valid,
                                               batch_size=1, shuffle=False,
                                               collate_fn=QuaternionData.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return test_loader

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, verbose=True, patience=30)

        optimizer2 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)

        optimizer3 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)

        optimizer4 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)
        
        return [optimizer1], [scheduler1]
    
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     # update optimizer every 8 steps (virtually increase of batch size)
    #     if optimizer_i == 0:
    #         if batch_nb % 8 == 0 :
    #             optimizer.step()
    #             optimizer.zero_grad()

    def Hamilton_product(self, quaternion1, quaternion2):
        res = []
        for i in range(quaternion1.shape[0]):
            a1, b1, c1, d1 = quaternion1[i]
            a2, b2, c2, d2 = quaternion2[i]
            res.append(torch.tensor([[a1*a2 - b1*b2 - c1*c2 - d1*d2,
                            a1*b2 + b1*a2 + c1*d2 - d1*c2,
                            a1*c2 - b1*d2 + c1*a2 + d1*b2,
                            a1*d2 + b1*c2 - c1*b2 + d1*a2]], dtype=torch.float, device='cuda', requires_grad=False))

        return torch.cat(res, dim=0)
                        
    def loss_function(self, pred, target):
        # unit_vector = torch.tensor(target.shape[0]*[[0, 1, 1, 1]], dtype=torch.float, device='cuda', requires_grad=False)
        # invert_vector = torch.tensor(target.shape[0]*[[1, -1, -1, -1]], dtype=torch.float, device='cuda', requires_grad=False)

        # ham_target = self.Hamilton_product(target, unit_vector)
        # target_ = invert_vector*target
        # target = self.Hamilton_product(ham_target, target_)
        # target = target[:,1:]

        #target = target[:,0]

        # inner_product = (pred * target).sum(dim=1)
        # a_norm = pred.pow(2).sum(dim=1).pow(0.5)
        # b_norm = target.pow(2).sum(dim=1).pow(0.5)
        # cos = inner_product / (a_norm * b_norm)
        # angle = torch.acos(cos)

        # loss = torch.zeros(1).cuda()
        # loss2 = torch.zeros(1).cuda()

        # for i in range(pred.shape[0]):
        #     temp = torch.abs(torch.dot(pred[i], target[i]))
        #     loss += torch.abs(torch.dot(pred[i], target[i]))
        #     loss2 += self.mse_loss(pred[i], target[i])
        target = target.long()
        sig = self.sigmoid(pred.float())
        loss =self.bce_loss(sig, target.float())

        return loss

    def permutation_loss(self, pred, target):
        return self.mse_loss(pred, target)
    
    def valid_function(self, pred, target):
       return self.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # if optimizer_idx != 3:
        #     pred_sum, real_sum = self.forward(x, optimizer_idx)
        #     permutation_loss = self.permutation_loss(pred_sum, real_sum)
        #     logs = {'permutation_loss': permutation_loss}
        #     return {'loss': permutation_loss, 'log': logs}

        outputs = self.forward(x, 3)
        loss = self.loss_function(outputs, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x, 3)
        loss = self.loss_function(outputs, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print(f'n/avg_loss: {avg_loss}')
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x, 3)
        loss = self.valid_function(outputs, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}