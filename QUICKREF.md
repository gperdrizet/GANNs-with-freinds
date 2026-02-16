# Quick Reference Guide

## For Students (Workers)

### Initial Setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds

# 2. Open in VS Code Dev Container
# Click "Reopen in Container" when prompted

# 3. Run setup script
bash scripts/setup.sh

# OR do it manually:
pip install -r requirements.txt
cp config/config.yaml.template config/config.yaml
# Edit config/config.yaml with database credentials
python scripts/download_celeba.py
```

### Starting Your Worker
```bash
cd src
python worker.py
```

### Stopping Your Worker
Press `Ctrl+C` in the terminal

### Troubleshooting
```bash
# Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test database connection
cd src/database
python init_db.py

# Test dataset
cd src/data
python dataset.py ../../data/celeba

# Check worker logs
# Look for "Processing work unit..." messages
```

## For Instructor (Main Coordinator)

### First Time Setup
```bash
# 1. Setup PostgreSQL database (containerized)
# Note connection details

# 2. Follow student setup steps

# 3. Initialize database
cd src
python database/init_db.py

# 4. Create initial model weights
python main.py --epochs 1 --sample-interval 1
# Stop it after first iteration (Ctrl+C)
```

### Starting Training
```bash
cd src
python main.py --epochs 50 --sample-interval 1
```

### Monitoring
```bash
# Watch outputs/samples/ directory for generated images
ls -ltr outputs/samples/

# Check active workers
# Query database: SELECT * FROM workers WHERE last_heartbeat > NOW() - INTERVAL '2 minutes';

# Check progress
# Query database: SELECT * FROM training_state;
```

### Stopping Training
Press `Ctrl+C` in the terminal. Workers will automatically stop when they detect training is inactive.

### Resetting Training
```bash
cd src
python database/init_db.py --reset
```

## Database Queries

### Check Training Progress
```sql
SELECT current_iteration, current_epoch, 
       g_loss, d_loss, 
       total_images_processed,
       training_active
FROM training_state;
```

### List Active Workers
```sql
SELECT worker_id, hostname, gpu_name, 
       total_work_units, total_images,
       last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes'
ORDER BY total_images DESC;
```

### Work Unit Status
```sql
SELECT iteration, status, COUNT(*) as count
FROM work_units
GROUP BY iteration, status
ORDER BY iteration DESC, status;
```

### Top Contributors
```sql
SELECT worker_id, gpu_name, 
       total_work_units, 
       total_batches,
       total_images
FROM workers
ORDER BY total_images DESC
LIMIT 10;
```

## File Structure

```
Important files:
├── config/config.yaml          # YOUR credentials (don't commit!)
├── src/
│   ├── worker.py               # Students run this
│   ├── main.py                 # Instructor runs this
│   └── database/init_db.py     # Database setup
├── data/celeba/                # Dataset location
└── outputs/samples/            # Generated images appear here
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Can't connect to DB | Check config.yaml credentials |
| Out of memory | Reduce batch_size in config.yaml |
| No work available | Wait for main process to create work units |
| Worker not updating | Check heartbeat in database |
| Slow training | Need more workers or larger batches |

## Performance Tuning

### If You Have Low VRAM (4-6GB)
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### If You Have High VRAM (12GB+)
```yaml
training:
  batch_size: 64  # Increase from 32
```

### If Database is Slow
```yaml
training:
  batches_per_work_unit: 20  # Increase from 10
  num_workers_per_update: 5  # Increase from 3
```

## Expected Timeline

- **Setup:** 15-30 minutes (dataset download takes longest)
- **Training:** 4-5 hours with 10 workers
- **Results:** Recognizable faces after ~2 hours

## Tips for Success

### Students
- Keep your worker running for the entire session
- Don't change config during training
- Close other GPU applications
- Check the samples directory to see progress

### Instructor
- Start training before students arrive
- Have database credentials ready to share
- Monitor for offline workers
- Save interesting sample generations
- Take screenshots for demonstration

## Emergency Procedures

### Worker Crashed
Just restart it - it will automatically pick up where it left off

### Main Process Crashed
Restart it - it will resume from last saved iteration

### Database Connection Lost
Workers will retry automatically. Check database availability.

### Need to Stop Everything
1. Stop main process (Ctrl+C)
2. Workers will stop automatically when they detect inactive training
3. OR manually stop database to force all connections closed
