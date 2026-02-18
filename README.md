# Distributed GAN training with students as workers

An educational distributed deep learning system where students become part of a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces.

## Concept

This project demonstrates distributed machine learning by:
- Using students' GPUs as a distributed compute cluster
- Coordinating training through a PostgreSQL database (no complex networking!)
- Training a DCGAN to generate realistic face images
- Teaching distributed systems, parallel training, and GANs simultaneously

## Architecture

**Main process (instructor):**
- Creates work units (batches of image indices)
- Aggregates gradients from workers
- Applies optimizer steps
- Tracks training progress

**Worker process (students):**
- Polls database for available work
- Computes gradients on assigned image batches
- Uploads gradients back to database
- Runs continuously until training completes

**PostgreSQL database:**
- Stores model weights, gradients, work units
- Acts as communication hub (no port forwarding needed!)
- Tracks worker statistics for monitoring

## Documentation

**[📚 Full documentation](https://gperdrizet.github.io/GANNs-with-freinds/)** (coming soon)

Quick links:
- [Getting Started](docs/getting-started/overview.md) - Introduction and concepts
- [Installation Guide](docs/getting-started/installation.md) - Choose your setup path
- [Student Guide](docs/guides/students.md) - How to participate as a worker
- [Instructor Guide](docs/guides/instructors.md) - Running the coordinator
- [Configuration Reference](docs/guides/configuration.md) - All config options
- [Architecture](docs/architecture/overview.md) - System design details
- [FAQ](docs/resources/faq.md) - Frequently asked questions

## Quick start

Choose your installation path:

| Setup Path | Best For | GPU Required | Documentation |
|------------|----------|--------------|---------------|
| **Google Colab** | Zero installation, free GPU | No (provided) | [Setup guide](docs/setup/google-colab.md) |
| **Dev Container** | Full development environment | Optional | [Setup guide](docs/setup/dev-container.md) |
| **Native Python** | Direct local control | Optional | [Setup guide](docs/setup/native-python.md) |
| **Conda** | Conda users | Optional | [Setup guide](docs/setup/conda.md) |
| **Local Training** | Single GPU, no database | Optional | [Setup guide](docs/setup/local-training.md) |

**New to the project?** Start with the [Getting Started Guide](docs/getting-started/overview.md).

**For students:** See the [Student Guide](docs/guides/students.md) for how to participate as a worker.

**For instructors:** See the [Instructor Guide](docs/guides/instructors.md) for running the coordinator and managing training.

## Features

- **Database-coordinated training**: No complex networking, works across firewalls
- **Fault tolerant**: Workers can disconnect/reconnect, automatic work reassignment
- **Flexible hardware**: CPU and GPU workers can participate together
- **Live monitoring**: Track progress via database queries or Hugging Face Hub
- **Educational**: Learn distributed systems, GANs, and parallel training

## What students learn

- **Distributed systems**: Coordination, fault tolerance, atomic operations
- **Deep learning**: GAN training, gradient aggregation, data parallelism
- **Practical skills**: PostgreSQL, PyTorch, collaborative computing

## Contributing

This is an educational project! Contributions welcome:
- Bug fixes and improvements
- Additional GAN architectures
- Web-based monitoring dashboard
- Gradient compression techniques
- Support for other datasets

## License

MIT License - See LICENSE file for details

