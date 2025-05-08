from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from dgl.dataloading import GraphDataLoader

from trainer import Trainer
from parser import parse_optimizer, parse_gnn_model, parse_loss
from utils_local import acc, metrics
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset, CancerSurvivalAnalysingDataset

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from datetime import datetime
import pandas as pd
import os



class CoxPHLoss(torch.nn.Module):
    def forward(self, risk_pred, duration, event):
        """
        Cox 部分似然损失函数
        risk_pred: 模型预测的风险得分 (N,)
        duration: 生存时间 (N,)
        event: 是否发生事件 (N,)
        """
        # 按生存时间从大到小排序
        sorted_indices = torch.argsort(duration, descending=True)
        risk_pred = risk_pred[sorted_indices]
        event = event[sorted_indices]

        # 计算部分似然
        hazard_ratio = torch.exp(risk_pred)
        log_cumsum_hazard_ratio = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_cumsum_hazard_ratio
        censored_likelihood = uncensored_likelihood * event

        # 取负的均值作为损失
        loss = -torch.mean(censored_likelihood)
        return loss

class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict, patient_data):
        super().__init__(config)
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)
        self.loss_fcn = CoxPHLoss()
        self.patient_data = patient_data
        train_path = self.config_data["train_path"]
        test_path  =self.config_data["eval_path"]
        self.train_paths = self.load_graph_paths(train_path)
        self.test_paths = self.load_graph_paths(test_path)

    def load_graph_paths(self, path_file):
        with open(path_file, 'r') as f:
            graph_paths = f.read().splitlines()
        return graph_paths
    def extract_patient_id(self, file_path):
        """
        从文件路径中提取病理号。
        假设病理号在文件名的前部分（例如 "B201509818-13.pt" -> "B201509818"）。
        """
        filename = os.path.basename(file_path)
        # 提取病理号部分（截取到第一个 "-" 之前的部分）
        patient_id = filename.split('-')[0]
        return patient_id

    def load_all_graphs_and_labels(self, graph_paths):
        all_graphs = []
        all_labels = []
        all_ids = []
        for graph_path in graph_paths:
            slide_id = os.path.basename(graph_path).split('.')[0]  # "B201701521-20.pt"
            survival_info = self.patient_data[self.patient_data['slide_id'] == slide_id]
            # 有些切片没有对应的 原发/非原发 病理数据
            if survival_info.empty:
                continue
            
            graph = torch.load(graph_path).to(self.device)
            all_ids.append(slide_id)
            
            survival_time = survival_info['生存时间'].iloc[0]
            event = int(survival_info['事件'])
            
            survival_label = torch.tensor([survival_time, event], dtype=torch.float32).to(self.device)
            all_graphs.append(graph)
            all_labels.append(survival_label)
        return all_graphs, all_labels, all_ids

    def train_one_epoch(self, graphs, survival_labels):
        self.optimizer.zero_grad()
        
        # 初始化风险预测、持续时间和事件信息的列表
        risk_preds = []
        durations = []
        events = []

        for graph, label in zip(graphs, survival_labels):
            risk_pred = self.gnn(graph).squeeze()
            risk_preds.append(risk_pred)
            durations.append(label[0])
            events.append(label[1])
        
        # 转换为张量
        risk_preds = torch.stack(risk_preds)
        durations = torch.stack(durations)
        events = torch.stack(events)

        # 计算损失
        loss = self.loss_fcn(risk_preds, durations, events)
        loss.backward()
        self.optimizer.step()

        # 计算C-index
        c_index = concordance_index(durations.cpu().numpy(), -risk_preds.detach().cpu().numpy(), events.cpu().numpy())
        
        return loss.item(), c_index



    def train(self) -> None:
        print("Start training with Cox regression analysis for survival prediction")
        
        best_c_index = 0
        
        for epoch in range(self.n_epoch):
            self.gnn.train()
            total_loss = 0
            c_index_list = []

            # 全批次训练
            train_graphs, train_labels, _ = self.load_all_graphs_and_labels(self.train_paths)
            
            loss, c_index = self.train_one_epoch(train_graphs, train_labels)
            total_loss += loss
            c_index_list.append(c_index)

            mean_c_index = np.mean(c_index_list)
            print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | C-index: {mean_c_index:.4f}")
            
            # 仅当C-index有提升时保存模型
            best_model_path = "./data/model_save/best_model_save.pt"
            if mean_c_index > best_c_index:
                best_c_index = mean_c_index
                torch.save(self.gnn.state_dict(), best_model_path)
                print(f"New best model saved with C-index: {best_c_index:.4f}")
                
        print(f"Train sample num: {len(train_graphs)}")

    def test(self, patient_data):
        print("Start testing with the best saved model")
        test_graphs, test_labels, all_ids = self.load_all_graphs_and_labels(self.test_paths)
        
        best_model_path = "./data/model_save/best_model_save.pt"
        self.gnn.load_state_dict(torch.load(best_model_path))
        self.gnn.eval()

        risk_pred_all = []
        durations_all = []
        events_all = []
        risk_pred_dict = {}

        with torch.no_grad():
            for graph, label, wsi_id in zip(test_graphs, test_labels, all_ids):
                risk_pred = self.gnn(graph).squeeze()
                risk_pred_all.append(risk_pred.item())
                durations_all.append(label[0].item())
                events_all.append(label[1].item())
                
                risk_pred_dict[wsi_id] = [risk_pred.item()]

        # 计算 C-index
        risk_pred_all = np.array(risk_pred_all)
        durations_all = np.array(durations_all)
        events_all = np.array(events_all)
        c_index = concordance_index(durations_all, -risk_pred_all, events_all)

        print(f"Test sample num: {len(test_graphs)}")
        print(f"Test C-index: {c_index:.4f}")

        # 为 patient_data 增加 "risk" 列
        patient_data["risk"] = patient_data["slide_id"].map(risk_pred_dict)

        # 只保留参与测试的病人
        patient_data_filtered = patient_data[patient_data["risk"].notnull()]

        # 将包含多个风险值的病人展开
        patient_data_expanded = patient_data_filtered.explode("risk", ignore_index=True)

        # 导出到 Excel 文件
        output_path = "./data/test_results.xlsx"
        patient_data_expanded.to_excel(output_path, index=False)
        print(f"Test results saved to {output_path}")