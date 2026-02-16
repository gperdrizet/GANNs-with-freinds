# Distributed GAN Training with Students as Workers

An educational distributed deep learning system where students become part of a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces.

## 🎯 Concept

This project demonstrates distributed machine learning by:
- Using students' GPUs as a distributed compute cluster
- Coordinating training through a PostgreSQL database (no complex networking!)
- Training a DCGAN to generate realistic face images
- Teaching distributed systems, parallel training, and GANs simultaneously

## 🏗️ Architecture

**Main Process (Instructor):**
- Creates work units (batches of image indices)
- Aggregates gradients from workers
- Applies optimizer steps
- Tracks training progress

**Worker Process (Students):**
- Polls database for available work
- Computes gradients on assigned image batches
- Uploads gradients back to database
- Runs continuously until training completes

**PostgreSQL Database:**
- Stores model weights, gradients, work units
- Acts as communication hub (no port forwarding needed!)
- Tracks worker statistics for monitoring

## 📁 Project Structure

```
GANNs-with-freinds/
├── src/
│   ├── models/
│   │   └── dcgan.py              # Generator & Discriminator models
│   ├── data/
│   │   └── dataset.py            # CelebA dataset loader
│   ├── database/
│   │   ├── schema.py             # Database table definitions
│   │   ├── db_manager.py         # Database operations
│   │   └── init_db.py            # Database initialization
│   ├── worker.py                 # Worker process (students run this)
│   ├── main.py                   # Main coordinator (instructor runs this)
│   └── utils.py                  # Helper functions
├── scripts/
│   └── download_celeba.py        # Dataset download script
├── config/
│   └── config.yaml.template      # Configuration template
├── data/                         # CelebA dataset goes here
├── outputs/
│   ├── samples/                  # Generated image samples
│   └── checkpoints/              # Model checkpoints
├── DESIGN.md                     # Detailed design document
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### For Students (Workers)

1. **Fork and clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
   cd GANNs-with-freinds
   ```

2. **Open in Dev Container**
   - Open VS Code
   - Click "Reopen in Container" when prompted
   - Wait for container to build

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CelebA dataset**
   ```bash
   python scripts/download_celeba.py
   ```
   Or download manually from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and extract to `./data/celeba/`

5. **Configure database connection**
   ```bash
   cp config/config.yaml.template config/config.yaml
   ```
   Edit `config/config.yaml` with database credentials provided by instructor

6. **Start worker**
   ```bash
   cd src
   python worker.py
   ```

   Your GPU is now part of the training cluster! 🎉

### For Instructor (Main Coordinator)

1. **Setup PostgreSQL database**
   - Deploy containerized PostgreSQL (provided separately)
   - Note connection details

2. **Follow student setup steps 1-5**

3. **Initialize database**
   ```bash
   cd src
   python database/init_db.py
   ```

4. **Start main coordinator**
   ```bash
   python main.py --epochs 50 --sample-interval 1
   ```

   Arguments:
   - `--epochs`: Number of epochs to train (default: 50)
   - `--sample-interval`: Generate samples every N iterations (default: 1)
   - `--config`: Path to config file (default: config/config.yaml)

5. **Monitor progress**
   - Generated samples appear in `outputs/samples/`
   - Check database for worker statistics
   - Training stops automatically when complete

## 📋 Requirements

### Hardware
- **NVIDIA GPU** (any consumer GPU works, even older models)
- **Minimum 4GB VRAM** (larger batches need more)
- **10GB free disk space** (for CelebA dataset)

### Software
- **Docker** with GPU support
- **VS Code** with Dev Containers extension
- **NVIDIA drivers** (version ≥545)

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
database:
  host: YOUR_DATABASE_HOST  # Provided by instructor
  port: 5432
  database: distributed_gan
  user: YOUR_DATABASE_USER  # Provided by instructor
  password: YOUR_DATABASE_PASSWORD  # Provided by instructor

training:
  batch_size: 32
  batches_per_work_unit: 10
  num_workers_per_update: 3

worker:
  poll_interval: 5  # seconds
  heartbeat_interval: 30  # seconds
  work_unit_timeout: 300  # seconds

data:
  dataset_path: ./data/celeba
```

## 🎓 What Students Learn

### Distributed Systems
- Coordination without direct networking
- Fault tolerance and worker dropout
- Atomic operations and race conditions
- Work unit timeouts and reassignment

### Deep Learning
- GAN architecture (Generator & Discriminator)
- Data parallel training
- Gradient aggregation
- Optimizer state management

### Database as Message Queue
- Novel use of PostgreSQL for distributed computing
- BLOB storage for model weights/gradients
- Atomic work unit claiming with `FOR UPDATE SKIP LOCKED`

## 📊 Monitoring

### For Students
Check your contribution stats in the database or wait for the instructor's dashboard.

### For Instructor
Monitor training progress via SQL queries:

```sql
-- Check training state
SELECT * FROM training_state;

-- See active workers
SELECT worker_id, gpu_name, total_images, last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes';

-- Work unit progress
SELECT status, COUNT(*) 
FROM work_units 
WHERE iteration = (SELECT current_iteration FROM training_state)
GROUP BY status;
```

## 🔧 Troubleshooting

**Worker can't connect to database:**
- Verify config.yaml has correct credentials
- Check database is publicly accessible
- Test connection: `psql -h HOST -U USER -d DATABASE`

**Worker runs out of memory:**
- Reduce `batch_size` in config.yaml
- Reduce `num_workers_dataloader` to 2 or 0
- Close other GPU applications

**No work units available:**
- Training may not have started yet
- All work units may be claimed by other workers
- Check if training is still active in database

**Gradients not being aggregated:**
- Check that minimum number of workers have completed work units
- Verify `num_workers_per_update` setting
- Look for errors in main coordinator logs

## 🧪 Testing

Test individual components:

```bash
# Test dataset loader
cd src/data
python dataset.py ../../data/celeba

# Test models
cd src/models
python dcgan.py

# Test database connection
cd src/database
python init_db.py

# Test utilities
cd src
python utils.py
```

## 📈 Expected Results

**Training Time:** 4-5 hours with 10 workers (mixed consumer GPUs)

**Sample Progression:**
- Iteration 0-100: Random noise
- Iteration 100-500: Blob-like shapes and colors
- Iteration 500-1000: Face-like structures emerge
- Iteration 1000-3000: Recognizable faces with details
- Iteration 3000+: High-quality celebrity faces

## 🎯 Performance Tips

### For Students
- Close unnecessary applications
- Use dedicated GPU if you have multiple
- Reduce batch size if running low on VRAM
- Keep worker running continuously for best results

### For Instructor  
- Start with more workers than `num_workers_per_update`
- Monitor for stale workers and timed-out work units
- Generate samples frequently to track progress
- Save checkpoints periodically

## 📚 Further Reading

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) - Original architecture
- [Data Parallel Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - PyTorch guide
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Dataset details

## 🤝 Contributing

This is an educational project! Contributions welcome:
- Bug fixes and improvements
- Additional GAN architectures
- Web-based monitoring dashboard
- Gradient compression techniques
- Support for other datasets

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- CelebA dataset from MMLAB CUHK
- DCGAN architecture from Radford et al.
- Inspired by real distributed training systems
- Built for AI/ML bootcamp students

---

**Questions?** Open an issue or contact your instructor!

