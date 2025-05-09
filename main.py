from globals import *
import yaml
import argparse
import random
import torch
from utils_local import ordered_yaml
import pandas as pd
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split
import os
# from trainer.train_gnn_kfold import GNNTrainer
from trainer.train_batch import GNNTrainer
# from trainer.train_gnn import GNNTrainer
from trainer.train_cnn_kfold import ResNetTrainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=42)

args = parser.parse_args()

opt_path = args.config

model = "gnn"
if model == "gnn":
    default_config_path = "training_config/survival_analysis.yml"
elif model == "cnn":
    default_config_path = "SUR/Resnet_SUR.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

mode = "train"
####SUR#####
patient_data = pd.read_excel("data/进入分析OS.xlsx")
patient_data['病理号'] = patient_data['病理号'].astype(str)
patient_data = patient_data[(patient_data["死亡时间"] != "外省") & (patient_data["死亡时间"] != "无身份证")]
end_date = datetime.strptime("2024-08-03", "%Y-%m-%d")
patient_data['病理诊断时间'] = pd.to_datetime(patient_data['病理诊断时间'], errors='coerce')
patient_data['死亡时间'] = patient_data['死亡时间'].apply(
    lambda x: end_date if isinstance(x, str) and "截止存活" in x else (x if isinstance(x, datetime) else pd.to_datetime(x, errors='coerce'))
)
patient_data['生存时间'] = (patient_data['死亡时间'] - patient_data['病理诊断时间']).dt.days
patient_data['事件'] = patient_data['死亡时间'] != end_date
patient_data = patient_data.dropna(subset=['生存时间'])

mult = pd.read_csv("data/Multiclass.csv")
mult['label'] = mult['label'].replace({"wei":1, "zhichang":1, "yuanfa":0})

merged = patient_data.merge(mult, left_on='病理号', right_on='case_id', how='left')

patient_data = merged[merged['label']==0]

patient_data.to_csv("patient_data.csv", index=False)

print(f"data num: {patient_data.shape[0]}")

level = 2

if level == 2:
    all_paths = glob.glob("data/create_save/graph_files/*.pt")
elif level == 1:
    all_paths = glob.glob("data/create_save_level1/graph_files/*.pt")
elif level == 0:
    all_paths = glob.glob("data/create_save_level0/graph_files/*.pt")


def Cancer_Survival_train_val(all_paths, patient_data):
    patient_ids = set(patient_data['病理号'].astype(str))
    slide151_list = ['1326-2020-13', '2022-F180-9', '21178-2020-2', '22148448-1', '22148448-5', '22148448-6', 'B201600145-20', 'B201600145-21', 'B201601788-4', 'B201601788-6', 'B201603108-11', 'B201603108-15', 'B201603108-8', 'B201603229-13', 'B201603229-20', 'B201604631-25', 'B201604631-28', 'B201604631-32', 'B201605295-25', 'B201605295-32', 'B201605295-33', 'B201605295-34', 'B201607425-3', 'B201608356-35', 'B201608356-38', 'B201608356-39', 'B201608593-1', 'B201608593-5', 'B201611518-22', 'B201611518-23', 'B201613718-3', 'B201613718-8', 'B201701521-15', 'B201701521-18', 'B201701521-20', 'B201705293-10', 'B201705293-16', 'B201705293-2', 'B201705293-9', 'B201706297-14', 'B201706297-16', 'B201707543-24', 'B201707664-31', 'B201708172-29', 'B201708172-30', 'B201708172-31', 'B201708653-29', 'B201708653-32', 'B201709252-28', 'B201709252-30', 'B201710807-12', 'B201710807-15', 'B201710807-7', 'B201711396-5', 'B201711396-7', 'B201712475-22', 'B201712475-23', 'B201713711-28', 'B201713711-31', 'B201713711-35', 'B201801816-27', 'B201801938-10', 'B201801938-13', 'B201808476-1', 'B201808476-3', 'B201808476', 'B201812444-18', 'B201812710-17', 'B201812710-18', 'B201814797-17', 'B201814797-18', 'B201902839-21', 'B201902839-22', 'B201902839-25', 'B201908879-12', 'B201908971-19', 'B201908971-23', 'B201911126-10', 'B201911126-6', 'B201911811-33', 'B201911811-36', 'B201917282-15', 'B201917282-16', 'B202001995-33', 'B202003715-19', 'B202004892-18', 'B202004892-21', 'B202005084-37', 'B202005084-40', 'B202006585-26', 'B202006585-30', 'B202006724-31', 'B202006724-35', 'B202006724-36', 'B202007585-44', 'B202007585-45', 'B202007585-47', 'B202010645-15', 'B202010645-17', 'B202010645-20', 'B202013014-30', 'B202016125-21', 'B202016125-22', 'B202017212-10', 'B202017212-3', 'B202017212-9', 'B202018060-19', 'B202018060-21', 'B202018060-25', 'B202118038-18', 'B202118038-19', 'B202118038-20', 'B202119537-1', 'B202122327-22', 'B202122327-23', 'B202124943-26', 'B202124943-27', 'B202124943-28', 'B202213457-5', 'B202213457-6', 'B202213457-7', 'B202217367-14', 'B202217367-15', 'B202217367-16', 'B202218864-14', 'B202218864-15', 'B202218864-16', 'B202220430-4', 'B202226018-14', 'B202226018-15', 'B202226018-16', 'B202301862-26', 'B202301862-27', 'B202303063-32', 'B202303063-34', 'B202304460-11', 'B202304460-12', 'B202304460-13', 'B202307107-27', 'B202307107-28', 'B202309442-15', 'B202309442-16', 'B202309442-17', 'B202310233-46', 'B202310598-19', 'CS202201744-1', 'CS202201744-4', 'CS202201744', 'F20210343-2', 'F20210343-3', 'F20210343-4']
    slide160_list = ['B202016125-21','B202016125-22','B202013014-30','B202007585-44','B202007585-45','B202007585-47','B202006724-31','B202006724-35','B202006724-36','B202001995-33','B201911811-36','B201902839-21','B201902839-22','B201902839-25','B201812444-18','B201808476','B201801816-27','B201713711-28','B201713711-31','B201713711-35','B201712475-22','B201712475-23','B201711396-5','B201711396-7','B201710807-7','B201710807-12','B201710807-15','B201708172-29','B201708172-30','B201708172-31','B201707543-24','B201701521-18','B201701521-20','B201608356-35','B201608356-38','B201605295-32','B201605295-33','B201605295-34','B201604631-25','B201604631-28','B201604631-32','B201603229-13','B201603229-20','B201601788-4','B201601788-6','B201600145-21','B202124943-26','B202124943-27','B202124943-28','B202118038-18','B202118038-19','B202118038-20','B202218864-14','B202218864-15','B202218864-16','B202217367-14','B202217367-15','B202217367-16','B202213457-5','B202213457-6','B202213457-7','B202310598-19','B202310233-46','B202307107-27','B202307107-28','B202304460-11','B202304460-12','B202304460-13','F20210343-2','F20210343-3','F20210343-4','22148448-1','22148448-5','22148448-6','21178-2020-2','JCS202003259-6','CS202002731-39','CS201902112-08','CS201902112-10','CS201802929-01','CS201705779-01','CS201705779-02','B201713712-13','CS201700472-04','CS201701164-1','CS201601553-02','B202120285-21','B202120285-22','B202120285-23','B202018060-19','B202018060-21','B202018060-25','B202017212-3','B202017212-9','B202017212-10','B202010645-17','B202010645-20','B202006585-26','B202006585-30','B202005084-37','B202005084-40','B202004892-21','B202003715-19','B201917282-16','B201911126-10','B201908971-23','B201908879-12','B201814797-17','B201814797-18','B201812710-17','B201812710-18','B201808476-1','B201808476-3','B201801938-13','B201709252-28','B201709252-30','B201708653-29','B201708653-32','B201707664-31','B201706297-16','B201705293-10','B201705293-9','B201613718-8','B201611518-22','B201611518-23','B201607425-3','B201603108-8','B201603108-11','B201603108-15','B202122327-22','B202122327-23','B202119537-1','B202226018-14','B202226018-15','B202226018-16','B202220430-4','B202309442-15','B202309442-16','B202309442-17','B202303063-32','B202303063-34','B202301862-26','B202301862-27','CS202201744','CS202201744-1','CS202201744-4','2022-F180-9','1326-2020-13','JCS202002751-1','CS201904844-4','JCS201901835-2','B201814482-07','CS201803303-21','B201714220-20','B201714220-22','CS201703019-2','CS201703019-3','CS201701574-1','B201609600-22','B201609600-23']
    slide160_one_patient_one_slide_list = ['B202016125-21','B202013014-30','B202007585-44','B202006724-31','B202001995-33','B201911811-36','B201902839-21','B201812444-18','B201808476','B201801816-27','B201713711-28','B201712475-22','B201711396-5','B201710807-7','B201708172-29','B201707543-24','B201701521-18','B201608356-35','B201605295-32','B201604631-25','B201603229-13','B201601788-4','B201600145-21','B202124943-26','B202118038-18','B202218864-14','B202217367-14','B202213457-5','B202310598-19','B202310233-46','B202307107-27','B202304460-11','F20210343-2','22148448-1','21178-2020-2','JCS202003259-6','CS202002731-39','CS201902112-08','CS201802929-01','CS201705779-01','B201713712-13','CS201700472-04','CS201701164-1','CS201601553-02','B202120285-21','B202018060-19','B202017212-3','B202010645-17','B202006585-26','B202005084-37','B202004892-21','B202003715-19','B201917282-16','B201911126-10','B201908971-23','B201908879-12','B201814797-17','B201812710-17','B201801938-13','B201709252-28','B201708653-29','B201707664-31','B201706297-16','B201705293-10','B201613718-8','B201611518-22','B201607425-3','B201603108-8','B202122327-22','B202119537-1','B202226018-14','B202220430-4','B202309442-15','B202303063-32','B202301862-26','CS202201744','2022-F180-9','1326-2020-13','JCS202002751-1','CS201904844-4','JCS201901835-2','B201814482-07','CS201803303-21','B201714220-20','CS201703019-2','CS201701574-1','B201609600-22']
    slide148_list = ['B202013014-30','B202007585-44','B202007585-45','B202007585-47','B202006724-31','B202006724-35','B202006724-36','B202001995-33','B201911811-36','B201902839-21','B201902839-22','B201902839-25','B201812444-18','B201808476','B201801816-27','B201713711-28','B201713711-31','B201713711-35','B201712475-22','B201712475-23','B201711396-5','B201711396-7','B201710807-7','B201710807-12','B201710807-15','B201708172-29','B201708172-30','B201708172-31','B201707543-24','B201701521-18','B201701521-20','B201608356-35','B201608356-38','B201605295-32','B201605295-33','B201605295-34','B201604631-25','B201604631-28','B201604631-32','B201603229-13','B201603229-20','B201601788-4','B201601788-6','B201600145-21','B202124943-26','B202124943-27','B202124943-28','B202118038-18','B202118038-19','B202118038-20','B202218864-14','B202218864-15','B202218864-16','B202217367-14','B202217367-15','B202217367-16','B202213457-5','B202213457-6','B202213457-7','B202310598-19','B202310233-46','B202307107-27','B202307107-28','B202304460-11','B202304460-12','B202304460-13','F20210343-2','F20210343-3','F20210343-4','22148448-1','22148448-5','22148448-6','JCS202003259-6','CS202002731-39','CS201902112-08','CS201902112-10','CS201802929-01','CS201705779-01','CS201705779-02','B201713712-13','CS201700472-04','CS201701164-1','CS201601553-02','B202018060-19','B202018060-21','B202018060-25','B202017212-3','B202017212-9','B202017212-10','B202010645-17','B202010645-20','B202006585-26','B202006585-30','B202005084-37','B202005084-40','B202004892-21','B202003715-19','B201917282-16','B201911126-10','B201908971-23','B201908879-12','B201814797-17','B201814797-18','B201812710-17','B201812710-18','B201808476-1','B201808476-3','B201801938-13','B201709252-28','B201709252-30','B201707664-31','B201706297-16','B201705293-10','B201705293-9','B201613718-8','B201611518-22','B201611518-23','B201607425-3','B201603108-8','B201603108-11','B201603108-15','B202122327-22','B202122327-23','B202119537-1','B202226018-14','B202226018-15','B202226018-16','B202220430-4','B202309442-15','B202309442-16','B202309442-17','B202303063-32','B202303063-34','B202301862-26','B202301862-27','CS202201744','CS202201744-1','CS202201744-4','2022-F180-9','JCS202002751-1','CS201904844-4','JCS201901835-2','B201814482-07','CS201803303-21','B201714220-20','B201714220-22','CS201703019-2','CS201703019-3']
    filtered_paths = [p for p in all_paths if ((os.path.basename(p).rpartition('-')[0] in patient_ids) or (os.path.basename(p).split('.')[0] in patient_ids))
                    # and os.path.basename(p).split('.')[0] in slide151_list
                    # and os.path.basename(p).split('.')[0] in slide160_list
                    # and os.path.basename(p).split('.')[0] in slide160_one_patient_one_slide_list
                    and os.path.basename(p).split('.')[0] in slide148_list
                    ]
    print(f"filtered data num: {len(filtered_paths)}")
    
    train_set, test_set = train_test_split(filtered_paths, test_size=0.5, random_state=42)

    def save_to_txt(file_path, data_set):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for path in data_set:
                f.write(f"{path}\n")
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_train.txt", train_set)
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_test.txt", test_set)
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_all.txt", filtered_paths)
    print("Filtered data sets have been saved to yuedix5_train.txt, yuedix5_test.txt, and yuedix5_all.txt.")
    return len(filtered_paths)

train_val = True
if train_val:
    slide_num = Cancer_Survival_train_val(all_paths, patient_data)

def main():
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")
    if mode == "train":
        # seed_list = range(100) # 100 random experiments were conducted to examine the performance distribution of each model
        seed_list = [83] # Random seed for spliting dataset
        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config, patient_data, slide_num, level)
        elif config["train_type"] == "cnn":
            trainer = ResNetTrainer(config, patient_data)
        else:
            raise NotImplementedError("This type of model is not implemented")
        for random_state in tqdm(seed_list):
            trainer.k_fold_patient(k=4, random_state=random_state)
        # trainer.train()
        # trainer.test(patient_data)
    elif mode == "eval":
        if config["eval_type"] == "homo-graph":
            evaluator = HomoGraphEvaluator(config)
        else:
            raise NotImplementedError("This type of evaluator is not implemented")
        evaluator.eval()
    elif mode == "graph_explain":
        explainer = ExplainGraph(config)
        explainer.eval()
    elif mode == "construct_graph":
        pass


if __name__ == "__main__":
    main()


