## GPU Memory Monitoring During Training

### **1. Real-Time Terminal Monitoring**

**Option A: nvidia-smi Watch (Windows)**
```cmd
# In a separate terminal, run this to update every 1 second:
nvidia-smi -l 1

# Or every 0.5 seconds for more granular monitoring:
nvidia-smi -l 0.5
```

**Option B: PowerShell Loop**
```powershell
# More readable format, updates every 2 seconds:
while($true) { 
    clear; 
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; 
    Start-Sleep 2 
}
```

### **2. Add Memory Logging to Training Code**

**Modify your trainer.py:**### **3. Quick Memory Check Commands**

**During Training (separate terminal):**
```cmd
# Simple memory check
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# More detailed with utilization
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
```

### **4. Add to Config for Memory Logging**
```yaml
logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  log_interval: 10
  memory_logging: true  # Add this line
```

### **5. Visual Memory Monitoring (Optional)**

**Install and use gpustat:**
```cmd
pip install gpustat
gpustat -i 1  # Updates every 1 second
```

**Or use built-in Task Manager:**
- Press `Ctrl + Shift + Esc`
- Go to "Performance" tab → "GPU 0"
- Watch "Dedicated GPU memory"

### **What to Watch For:**
- **Memory steadily increasing** = Memory leak
- **Sudden spikes** = Batch size too large
- **Memory near 12GB** = Reduce batch size immediately
- **Memory fragmentation** = Regular `torch.cuda.empty_cache()`

**Immediate Action:** Run the memory monitoring and see the pattern before the crash at epoch 12.