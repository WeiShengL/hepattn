# import numpy as np                                                                                                                                  
# import torch                                                                                                                                        
# import matplotlib.pyplot as plt                                                                                                                     
# from hepattn.experiments.cld.event_display import plot_cld_event                                                                                    
                                                                                                                                                    
# # Load the problematic file directly                                                                                                                
# file_path = "/global/cfs/cdirs/m2616/wlai/cld/prepped_with_mask/train/reco_p8_ee_tt_ecm365_7979928_274_condor/reco_p8_ee_tt_ecm365_7979928_274_condor_719.npz"  
                                                                                                                                                    
# with np.load(file_path, allow_pickle=True) as f:                                                                                                    
#     raw_event = {k: f[k] for k in f.files}                                                                                                          
                                                                                                                                                    
# # Check which subdetectors have zero hits                                                                                                           
# subdetectors = ["vtb", "vte", "itb", "ite", "otb", "ote", "ecb", "ece", "hcb", "hce", "hco", "msb", "mse"]                                          
# print("Subdetector hit counts:")                                                                                                                    
# for det in subdetectors:                                                                                                                            
#     key = f"{det}.pos.x"                                                                                                                            
#     if key in raw_event:                                                                                                                            
#         print(f"  {det}: {raw_event[key].shape[0]} hits")                                                                                           
#     else:                                                                                                                                           
#         print(f"  {det}: key not found")                                                                                                            
                                                                                                                                                    
# # Check merged subdetectors (vtxd = vtb + vte)                                                                                                      
# vtxd_hits = raw_event.get("vtb.pos.x", np.array([])).shape[0] + raw_event.get("vte.pos.x", np.array([])).shape[0]                                   
# print(f"\n  vtxd (vtb+vte): {vtxd_hits} hits")                                                                                                      
                                                                                                                                                    
# If you want to use the full plot_cld_event function, you'll need to process the event through the dataset first. Here's a more complete approach:   
                                                                                                                                                    
import numpy as np                                                                                                                                  
import torch                                                                                                                                        
import matplotlib.pyplot as plt                                                                                                                     
from hepattn.experiments.cld.data import CLDDataset                                                                                                 
from hepattn.experiments.cld.event_display import plot_cld_event                                                                                    
                                                                                                                                                    
# Create a minimal dataset pointing to just this file's directory                                                                                   
# You'll need to adjust inputs/targets based on your config                                                                                         
inputs = {                                                                                                                                          
    "vtxd": ["pos.x", "pos.y", "pos.z", "time", "type"],                                                                                            
    "trkr": ["pos.x", "pos.y", "pos.z", "time", "type"],                                                                                            
    # Add other inputs as needed from your config                                                                                                   
}                                                                                                                                                   
targets = {                                                                                                                                         
    "particle_vtxd": ["valid"],                                                                                                                     
    "particle_trkr": ["valid"],                                                                                                                     
    # Add other targets as needed                                                                                                                   
}                                                                                                                                                   
                                                                                                                                                    
# Load the single event file directly for inspection                                                                                                
file_path = "/global/cfs/cdirs/m2616/wlai/cld/prepped_with_mask/train/reco_p8_ee_tt_ecm365_7979928_274_condor/reco_p8_ee_tt_ecm365_7979928_274_condor_719.npz"  
                                                                                                                                                    
with np.load(file_path, allow_pickle=True) as f:                                                                                                    
    event = {k: f[k] for k in f.files}                                                                                                              
                                                                                                                                                    
# Print all keys containing "vtb" or "vte" to debug                                                                                                 
print("Keys related to vtxd (vtb/vte):")                                                                                                            
for k in sorted(event.keys()):                                                                                                                      
    if "vtb" in k or "vte" in k:                                                                                                                    
        v = event[k]                                                                                                                                
        if hasattr(v, 'shape'):                                                                                                                     
            print(f"  {k}: shape={v.shape}")                                                                                                        
        else:                                                                                                                                       
            print(f"  {k}: {v}")                                                                                                                    
                                                                                                                                                    
# Check particle count                                                                                                                              
print(f"\nNumber of particles: {event['particle.PDG'].shape[0]}") 