from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from dgl.dataloading import GraphDataLoader

from trainer import Trainer
from evaluator import HomoGraphEvaluator
from parser import parse_optimizer, parse_gnn_model, parse_loss
from utils_local import acc, metrics
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset, CancerSurvivalAnalysingDataset

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from datetime import datetime
import pandas as pd
import os

from sklearn.model_selection import KFold
import pickle
from collections import defaultdict

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

        # 部分似然
        hazard_ratio = torch.exp(risk_pred)
        log_cumsum_hazard_ratio = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_cumsum_hazard_ratio
        censored_likelihood = uncensored_likelihood * event

        loss = -torch.mean(censored_likelihood)
        return loss

# class CoxPHLoss(torch.nn.Module):
#     def __init__(self, sample_ratio=0.6):
#         super().__init__()
#         self.sample_ratio = sample_ratio

#     def forward(self, risk_pred, duration, event):
#         sorted_indices = torch.argsort(duration, descending=True)
#         risk_pred = risk_pred[sorted_indices]
#         event = event[sorted_indices]

#         hazard_ratio = torch.exp(risk_pred)
#         log_cumsum_hazard_ratio = torch.log(torch.cumsum(hazard_ratio, dim=0))

#         uncensored_likelihood = risk_pred - log_cumsum_hazard_ratio
#         censored_likelihood = uncensored_likelihood * event

#         if self.sample_ratio < 1.0:
#             mask = torch.rand(len(risk_pred)) < self.sample_ratio
#             censored_likelihood = censored_likelihood[mask]
        
#         loss = -torch.mean(censored_likelihood)
#         return loss


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict, patient_data, slide_num, level):
        super().__init__(config)

        self.loss_fcn = CoxPHLoss()
        self.patient_data = patient_data
        train_path = self.config_data["train_path"]
        test_path = self.config_data["eval_path"]
        all_path = self.config_data["all_path"]
        
        self.train_paths = self.load_graph_paths(train_path)
        self.test_paths = self.load_graph_paths(test_path)
        self.all_paths = self.load_graph_paths(all_path)
        self.slide_num = slide_num
        self.level = level
        

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
            slide_id, graph_format = os.path.splitext(os.path.basename(graph_path))
            graph_format = graph_format.lstrip('.') # "B201701521-20.pt"
            
            survival_info = self.patient_data[self.patient_data['slide_id'] == slide_id]
            # 有些切片没有对应的 原发/非原发 病理数据
            if survival_info.empty:
                print(f"slide {slide_id} is not in survival list")
                continue
            if graph_format == "pkl":
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f).to(self.device)
            elif graph_format == "pt":
                graph = torch.load(graph_path).to(self.device)
            
            all_ids.append(slide_id)
            survival_time = survival_info['生存时间'].iloc[0]
            event = int(survival_info['事件'])
            
            survival_label = torch.tensor([survival_time, event], dtype=torch.float32).to(self.device)
            all_graphs.append(graph)
            all_labels.append(survival_label)
        return all_graphs, all_labels, all_ids

    def train_one_epoch(self, graphs, survival_labels, fold_index, random_state, fold_num):
        
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
        
        mean_risk = sum(risk_preds) / len(risk_preds)
        train_re = pd.DataFrame({
            'risk_preds': risk_preds.detach().cpu().numpy(),
            'durations': durations.detach().cpu().numpy(),
            'events':events.detach().cpu().numpy()
        })
        output_dir = f"./data/train_results/level{self.level}/slide_num{self.slide_num}/fold_num_{fold_num}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"train_results_seed{random_state}_fold_{fold_index + 1}.csv")
        train_re.to_csv(output_path, index=False)
        print(f"seed{random_state} fold{fold_index + 1} mean risk: {mean_risk}")
        
        return loss.item(), c_index



    def train(self, train_paths, fold_index, random_state, fold_num) -> None:
        """
        训练当前折的模型，并保存最佳模型到与折号相关的文件。
        """
        print(f"Start training for fold {fold_index + 1}")
        
        # 每折重新初始化 GNN 模型
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)
        best_c_index = 0
        best_model_path = self.best_model_path
        
        for epoch in range(self.n_epoch):
            self.gnn.train()
            total_loss = 0
            c_index_list = []

            # 全批次训练
            train_graphs, train_labels, _ = self.load_all_graphs_and_labels(train_paths)
            
            loss, c_index = self.train_one_epoch(train_graphs, train_labels, fold_index, random_state, fold_num)
            total_loss += loss
            c_index_list.append(c_index)

            mean_c_index = np.mean(c_index_list)
            print(f"Fold {fold_index + 1} | Epoch {epoch + 1} | Loss: {total_loss:.4f} | C-index: {mean_c_index:.4f}")
            
            # 仅当C-index有提升时保存模型
            if mean_c_index > best_c_index:
                best_c_index = mean_c_index
                torch.save(self.gnn.state_dict(), best_model_path)
                print(f"Fold {fold_index + 1}: New best model saved with C-index: {best_c_index:.4f}")
        
        print(f"Fold {fold_index + 1} Train sample num: {len(train_graphs)}")


    def test(self, patient_data, test_paths, fold_index, random_state, fold_num) -> float:
        """
        测试当前折的模型，确保使用当前折的最佳模型。
        """
        print(f"Start testing for fold {fold_index + 1}")
        
        test_graphs, test_labels, all_ids = self.load_all_graphs_and_labels(test_paths)
        
        # 加载对应折的最佳模型
        best_model_path = self.best_model_path
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

        # # 计算 C-index
        # risk_pred_all = np.array(risk_pred_all)
        # durations_all = np.array(durations_all)
        # events_all = np.array(events_all)

        # c_index = concordance_index(durations_all, -risk_pred_all, events_all)

        print(f"Fold {fold_index + 1} | Test sample num: {len(test_graphs)}")
        # print(f"Fold {fold_index + 1} | Test C-index: {c_index:.4f}")

        # 保存测试结果到 Excel 文件
        patient_data["risk"] = patient_data["slide_id"].map(risk_pred_dict)
        patient_data_filtered = patient_data[patient_data["risk"].notnull()]
        patient_data_expanded = patient_data_filtered.explode("risk", ignore_index=True)

        output_dir = f"./data/test_results/level{self.level}/slide_num{self.slide_num}/fold_num_{fold_num}/test_fold"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_results_seed{random_state}_fold_{fold_index + 1}.csv")
        patient_data_expanded.to_csv(output_path, index=False)
        print(f"Test results for fold {fold_index + 1} saved to {output_path}")
        
        # return c_index

    def k_fold_patient(self, k=2, random_state=42):
        """
        使用 K 折交叉验证进行训练和评估。
        确保同一个病人的切片只能同时出现在训练集或测试集。
        """
        patient_data = self.patient_data
        fold_results = []
        print(f"开始 {k}-折交叉验证，用于 Cox 生存预测模型")

        
        # 构造病人ID到切片路径的映射
        patient_to_slices = defaultdict(list)
        for path in self.all_paths:
            filename = os.path.basename(path)  # 提取文件名
            base_name = filename.split(".")[0]  # 去掉扩展名
            if '-' in base_name:
                patient_id = base_name.rsplit("-", 1)[0]  # 提取最后一个-之前的部分
            else:
                patient_id = base_name  # 如果没有-，就用整个文件名作为病人ID
            patient_to_slices[patient_id].append(path)
        
        # 获取所有病人ID
        patient_ids = list(patient_to_slices.keys())
        print(f"总病人数: {len(patient_ids)}")

        # 使用病人ID进行 K 折划分
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (train_patient_idx, test_patient_idx) in enumerate(kfold.split(patient_ids)):
            print(f"--- 第 {fold + 1}/{k} 折 ---")
            
            # 获取训练集和测试集的病人ID
            train_patient_ids = [patient_ids[i] for i in train_patient_idx]
            test_patient_ids = [patient_ids[i] for i in test_patient_idx]
            
            # 根据病人ID获取对应的切片路径
            train_paths = [path for patient_id in train_patient_ids for path in patient_to_slices[patient_id]]
            test_paths = [path for patient_id in test_patient_ids for path in patient_to_slices[patient_id]]
            
            print(f"训练集病人数: {len(train_patient_ids)}, 测试集病人数: {len(test_patient_ids)}")
            print(f"训练集切片数: {len(train_paths)}, 测试集切片数: {len(test_paths)}")
            
            # 训练模型
            model_dir = f"./data/model_save/level{self.level}"
            self.best_model_path = f"{model_dir}/best_model_fold_{fold + 1}_seed{random_state}.pt"
            os.makedirs(model_dir, exist_ok=True)
            self.train(train_paths, fold_index=fold, random_state=random_state, fold_num=k)
            
            # 测试模型
            # c_index = self.test(patient_data, test_paths, fold_index=fold, random_state=random_state, fold_num=k)
            self.test(patient_data, test_paths, fold_index=fold, random_state=random_state, fold_num=k)
            # fold_results.append(c_index)
        
        # 打印和保存交叉验证结果
        print(f"Cross-validation results: {fold_results}")
        print(f"Mean C-index across folds: {np.mean(fold_results):.4f}")

    def k_fold_slide(self, k=2, random_state=42):
            """
            使用 K 折交叉验证进行训练和评估。
            """
            patient_data = self.patient_data
            fold_results = []
            print(f"开始 {k}-折交叉验证，用于 Cox 生存预测模型")
            
            kfold = KFold(n_splits=k, shuffle=True, random_state = random_state)
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(self.all_paths)):
                print(f"--- 第 {fold + 1}/{k} 折 ---")
                
                train_paths = [self.all_paths[i] for i in train_idx]
                test_paths = [self.all_paths[i] for i in test_idx]
                
                # 训练模型
                self.train(train_paths, fold_index=fold, fold_num=k)
                
                # 测试模型
                c_index = self.test(patient_data, test_paths, fold_index=fold, fold_num=k)
                fold_results.append(c_index)
            
            # 打印和保存交叉验证结果
            print(f"Cross-validation results: {fold_results}")
            print(f"Mean C-index across folds: {np.mean(fold_results):.4f}") 

