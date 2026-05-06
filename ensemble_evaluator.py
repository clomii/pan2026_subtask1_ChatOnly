import json
import os
import numpy as np
from evaluate_baselines import evaluate

def load_predictions(filepath):
    """Loads a JSONL prediction file into a dictionary of {id: probability}."""
    if not os.path.exists(filepath):
        return None
    
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[obj['id']] = obj['label']
    return data

def main():
    print("=== Meta-Ensemble Blending System ===")
    
    # 1. Load predictions from our 3 diverse models
    # The structural detector
    custom_preds = load_predictions('custom_model.jsonl')
    
    # The statistical/perplexity detector
    binoc_preds = load_predictions('binoculars_fast.jsonl')
    
    # The semantic deep learning detector (if you trained it)
    deberta_preds = load_predictions('deberta_model.jsonl')
    
    if not custom_preds:
        print("Error: custom_model.jsonl not found!")
        return
        
    print(f"Loaded N-Gram Model: {len(custom_preds)} predictions")
    
    if binoc_preds:
        print(f"Loaded Binoculars Model: {len(binoc_preds)} predictions")
    if deberta_preds:
        print(f"Loaded DeBERTa Model: {len(deberta_preds)} predictions")

    # Combine them using Weighted Soft Voting
    print("\nBlending predictions...")
    ensemble_results = []
    
    for doc_id, custom_score in custom_preds.items():
        scores = []
        weights = []
        
        # We heavily trust our Custom N-Gram model because it scored 0.999
        scores.append(custom_score)
        weights.append(0.5) 
        
        if binoc_preds and doc_id in binoc_preds:
            scores.append(binoc_preds[doc_id])
            weights.append(0.2) # Binoculars gets 20% weight
            
        if deberta_preds and doc_id in deberta_preds:
            scores.append(deberta_preds[doc_id])
            weights.append(0.3) # DeBERTa gets 30% weight
            
        # Calculate weighted average
        weighted_avg = np.average(scores, weights=weights)
        ensemble_results.append({"id": doc_id, "label": float(weighted_avg)})
        
    # Save the ensemble results
    output_file = 'ensemble_model.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in ensemble_results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Ensemble saved to {output_file}!")
    print("\nEvaluating the Ultimate Ensemble Model...")
    
    # Reuse your evaluate_baselines logic!
    evaluate('data/val.jsonl', output_file)

if __name__ == '__main__':
    main()
