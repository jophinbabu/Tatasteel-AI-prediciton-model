
import lightgbm as lgb
import os

model_path = r"D:\Tata\tata-trading-system\models\saved\TATASTEEL_NS_lgbm.txt"

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    bst = lgb.Booster(model_file=model_path)
    print("\nModel Parameters:")
    # dump_model() returns a dict with 'params' key
    dump = bst.dump_model()
    # However, sometimes params are not fully in dump if loaded from text
    # Let's try to print the 'params' attribute directly if accessible via Python API hacks or just inspect the dump
    print(bst.params) 
    
    # Or parsing from string dump
    print("\nDump 'params':")
    # Using .dump_model() is safer
    config = bst.dump_model()
    if 'params' in config:
        print(config['params'])
    else:
        print("Params not found in dump keys:", config.keys())
        
    print("\nNum Trees:", bst.num_trees())
else:
    print("Model file not found!")
