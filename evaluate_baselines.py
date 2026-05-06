import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, fbeta_score, confusion_matrix, accuracy_score

def load_jsonl(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[obj['id']] = obj
    return data

def evaluate(truth_path, pred_path):
    if not os.path.exists(pred_path):
        print(f"File not found: {pred_path}. Skipping...\n")
        return

    truth_data = load_jsonl(truth_path)
    pred_data = load_jsonl(pred_path)
    
    y_true = []
    y_prob = []
    y_pred = []
    
    for doc_id, truth_obj in truth_data.items():
        if doc_id in pred_data:
            y_true.append(truth_obj['label'])
            prob = pred_data[doc_id]['label']
            y_prob.append(prob)
            
            # Threshold probabilities at 0.5 for binary classification
            y_pred.append(1 if prob >= 0.5 else 0)
            
    if not y_true:
        print(f"No matching predictions found in {pred_path}\n")
        return
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    
    # 1. ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0 # Fallback in case of only one class present
        
    # 2. Brier Score
    # Note: PAN uses 1 - brier_score_loss so that higher is better (1.0 is perfect)
    brier = 1.0 - brier_score_loss(y_true, y_prob)
    
    # 3. C@1
    # When there are no unanswered predictions, C@1 is mathematically identical to accuracy
    c_at_1 = accuracy_score(y_true, y_pred)
    
    # 4. F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # 5. F0.5u Score
    # Weights precision twice as much as recall
    f05u = fbeta_score(y_true, y_pred, beta=0.5)
    
    # 6. Mean of the core 5 PAN metrics
    mean_score = np.mean([roc_auc, brier, c_at_1, f1, f05u])
    
    # FPR (False Positive Rate) and FNR (False Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"=== Evaluation Results for: {pred_path} ===")
    print(f"ROC-AUC : {roc_auc:.3f}")
    print(f"Brier   : {brier:.3f}")
    print(f"C@1     : {c_at_1:.3f}")
    print(f"F1      : {f1:.3f}")
    print(f"F0.5u   : {f05u:.3f}")
    print(f"Mean    : {mean_score:.3f}")
    print(f"----------------------")
    print(f"FPR     : {fpr:.3f}")
    print(f"FNR     : {fnr:.3f}")
    print("===========================================\n")

if __name__ == '__main__':
    truth_file = 'data/val.jsonl'
    
    if not os.path.exists(truth_file):
        print(f"Truth file {truth_file} not found!")
    else:
        evaluate(truth_file, 'tfidf.jsonl')
        evaluate(truth_file, 'ppmd.jsonl')
        evaluate(truth_file, 'custom_model.jsonl')
        evaluate(truth_file, 'binoculars.jsonl')
