import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

gen_length = 32
block_size = 16
exp_name = "llada-sft-gsm8k-0.7"
non_exp_name = "llada-sft-gsm8k-0.7-00"
device = 'cuda:4'
def main():

    local_interfere_history = torch.load(f"./SFT/exps/{non_exp_name}/attribute_targetnon_effect_gen{gen_length}_block{block_size}_another.pt", map_location='cpu')
    local_target_history = torch.load(f"./SFT/exps/{exp_name}/attribute_targetnon_effect_gen{gen_length}_block{block_size}.pt", map_location='cpu')
    interfere_history = torch.load(f"./SFT/exps/{non_exp_name}/attribute_target_effect_gen{gen_length}_block{block_size}_another.pt", map_location='cpu')
    target_history = torch.load(f"./SFT/exps/{exp_name}/attribute_target_effect_gen{gen_length}_block{block_size}.pt", map_location='cpu')

    interfere_tensor = torch.stack(interfere_history) 
    target_tensor = torch.stack(target_history) 
    local_interfere_tensor = torch.stack(local_interfere_history) 
    local_target_tensor = torch.stack(local_target_history) 

    target_tensor = target_tensor.float()
    interfere_tensor = interfere_tensor.float()
    local_target_tensor = local_target_tensor.float()
    local_interfere_tensor = local_interfere_tensor.float()
    # ------------------------------------------------------------
    # DBSCAN Clustering
    # ------------------------------------------------------------

    X_non = interfere_tensor.view(interfere_tensor.size(0), -1).cpu().numpy()
    X_mem = target_tensor.view(target_tensor.size(0), -1).cpu().numpy()
    X      = np.vstack([X_non, X_mem])           
    y_true = np.hstack([np.zeros(len(X_non)), np.ones(len(X_mem))])
    X_std = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.8, min_samples=20, metric='euclidean', n_jobs=-1) 
    labels = db.fit_predict(X_std)              
    unique_clusters = np.unique(labels)
    pred_scores = np.zeros(len(X))   
    pred_label  = np.zeros(len(X))  
    for k in unique_clusters:
        idx_k = np.where(labels == k)[0]           
        if k == -1:
            mu_k = X_std[idx_k].mean(axis=0, keepdims=True)
        else:
            mu_k = X_std[idx_k].mean(axis=0, keepdims=True)
        dist   = np.linalg.norm(X_std[idx_k] - mu_k, axis=1)
        order  = np.argsort(dist)                   
        pred_scores[idx_k] = dist                  
        half = len(order) // 2
        pred_label[idx_k[order[:half]]]  = 1       
        pred_label[idx_k[order[half:]]] = 0         
    precision, recall, thresholds = precision_recall_curve(y_true, pred_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    fpr, tpr, thresholds = roc_curve(y_true, pred_scores)
    f1 = np.max(f1_scores)
    acc = np.max(1-(fpr+(1-tpr))/2)
    auc = roc_auc_score(y_true, pred_scores)
    low_05 = tpr[np.where(fpr<.05)[0][-1]]
    low_01 = tpr[np.where(fpr<.01)[0][-1]]
    print(f"\nDBSCAN AUC = {auc:.4f}, TPR@5%FPR = {low_05:.4f}, TPR@1%FPR = {low_01:.4f}, f1 = {f1:.4f}, acc = {acc:.4f}\n")

    # ------------------------------------------------------------
    # Distance Based Attribution
    # ------------------------------------------------------------

    class1 = torch.mean(interfere_tensor.float(), dim=0) 
    class2 = torch.mean(target_tensor.float(), dim=0) 
    local_class1 = torch.mean(local_interfere_tensor.float(), dim=0) 
    local_class2 = torch.mean(local_target_tensor.float(), dim=0) 

    avg_non = class1.unsqueeze(0)    
    avg_mem = class2.unsqueeze(0)


    dist_non_non   = ((local_interfere_tensor - avg_non)**2).mean(dim=(1,2))
    dist_non_mem = ((local_interfere_tensor - avg_mem)**2).mean(dim=(1,2))
    score_non  = dist_non_non - dist_non_mem
    dist_mem_non = ((local_target_tensor - avg_non)**2).mean(dim=(1,2))
    dist_mem_mem   = ((local_target_tensor - avg_mem)**2).mean(dim=(1,2))
    score_mem  = dist_mem_non - dist_mem_mem

    y_true  = np.concatenate([np.zeros(len(score_non)), np.ones(len(score_mem))])
    y_score = torch.cat([score_non, score_mem]).cpu().numpy()   # 转 numpy
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    f1 = np.max(f1_scores)
    acc = np.max(1-(fpr+(1-tpr))/2)
    auc = roc_auc_score(y_true, y_score)
    low_05 = tpr[np.where(fpr<.05)[0][-1]]
    low_01 = tpr[np.where(fpr<.01)[0][-1]]
    print(f"\nDistance AUC = {auc:.4f}, TPR@5%FPR = {low_05:.4f}, TPR@1%FPR = {low_01:.4f}, f1 = {f1:.4f}, acc = {acc:.4f}\n")

    # ------------------------------------------------------------
    # Gaussian Trajectory Attribution
    # ------------------------------------------------------------

    def fit_gaussian(tensor_group):
        tensor_group = tensor_group.float()
        mean = tensor_group.mean(dim=0) 
        std  = tensor_group.std(dim=0) 
        return mean, std

    def log_likelihood(sample, mean, std, eps=1e-6):
        z = (sample - mean) / (std + eps)
        log_prob = -0.5 * (z**2 + torch.log(2 * torch.pi * (std + eps)**2))
        return log_prob.sum()  

    mean_mem, std_mem = fit_gaussian(local_target_tensor)
    mean_non, std_non = fit_gaussian(local_interfere_tensor)
    test_tensor = torch.cat([target_tensor, interfere_tensor], dim=0)
    true_labels = np.concatenate([
        np.ones(target_tensor.shape[0]),
        np.zeros(interfere_tensor.shape[0])
    ])

    scores = []
    for sample in test_tensor:
        logp_mem = log_likelihood(sample, mean_mem, std_mem)
        logp_non = log_likelihood(sample, mean_non, std_non)
        scores.append((logp_mem - logp_non).item())

    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    f1 = np.max(f1_scores)
    acc = np.max(1-(fpr+(1-tpr))/2)
    auc = roc_auc_score(true_labels, scores)
    low_05 = tpr[np.where(fpr<.05)[0][-1]]
    low_01 = tpr[np.where(fpr<.01)[0][-1]]
    print(f"\nGTA AUC = {auc:.4f}, TPR@5%FPR = {low_05:.4f}, TPR@1%FPR = {low_01:.4f}, f1 = {f1:.4f}, acc = {acc:.4f}\n")

if __name__ == '__main__':
    main()
