import json
import sys
from tqdm import tqdm
from pan25_genai_baselines.binoculars import Binoculars

def main():
    print("Initializing Binoculars Fast Runner...")
    print("This will use Llama-3.2-3B models to save VRAM and speed up processing.")
    
    detector = Binoculars()
    
    input_file = "data/val.jsonl"
    output_file = "binoculars_fast.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"\nLoaded {len(lines)} texts. Starting prediction...")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # PBar so you see exactly how many are left and the ETA!
        for line in tqdm(lines, desc="Predicting", total=len(lines), unit="texts"):
            if not line.strip():
                continue
                
            data = json.loads(line)
            # Detector returns a torch tensor, we grab the float
            score = detector.get_score(data['text'], normalize=True)
            
            # Save exactly like the baseline does
            res = {"id": data['id'], "label": float(score)}
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()

if __name__ == "__main__":
    main()
