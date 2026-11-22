# Getting started

You can run GRAIL without generating a video or owning a robot. The two bundled recordings come with their camera, object mesh and initial scene geometry. Pick **stairs** for body-only locomotion or **kettlebell** for hand–object contact. Both finish by training and evaluating a policy in simulation.

## Before you download

| Requirement | Plan for |
|---|---|
| GPU host | Tested on Ubuntu 24.04 LTS with an RTX 6000 Ada (48 GB), driver 595.99.02, one video at a time. The L40S production rollout rendered on driver 580.95.05. See [GPU and renderer troubleshooting](troubleshooting.md). |
| Storage | The standard image is about 89 GB unpacked. The published input weights total about 20 GB before additional first-use inference downloads. In addition to the image, allow roughly 100 GB on a fast writable disk for the checkout, [isolated workers](configuration.md), weights, package caches and short-run outputs. Leave extra room for downloads and full training checkpoints. These are planning figures, not installation minima. |
| Tools and accounts | Install [Git LFS](https://git-lfs.com/), [Docker Engine](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Reconstruction and retargeting use [OpenAI Codex CLI](installation.md#check-codex-on-the-host). The supplied clips and released weights can be downloaded anonymously. |
| Simulation | The tracking examples use Isaac Sim inside the `sonic` environment. Review the [Isaac Sim terms](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/legal.html) before the tracking **installation**, which records acceptance. For the code and assets, see [GRAIL licensing](resources.md). |

**Host, a few seconds.** From the directory where you will keep the checkout, check the GPU, Docker, Git LFS and available space. Look for your GPU name and memory, two tool versions and room on the disk you plan to use. Docker's image store may be on a separate disk.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
docker --version
git lfs version
df -h .
```

## Budget your first run

Approximate timings below are from a single RTX 6000 Ada. Reconstruction and retargeting include source review and preview generation; agent decisions and scene complexity affect wall time.

**Shared cold setup for either example:** allow 45–90 minutes for the checkout, image, worker setup and public weights on a fast to shared connection. Network throttling can extend this. The first fit also downloads its depth model. The totals below exclude manual inspection, transfer retries and any extra first-use download.

| Step | Stairs | Kettlebell |
|---|---:|---:|
| Reconstruct and review the human fit | 33 min with a first-use depth download | 17 min with the depth checkpoint cached |
| Retarget and inspect the G1 motion | 9 min | 14 min |
| Short tracking run, including simulation startup and evaluation | 1 min 09 s | 2 min 20 s, metrics-only |
| First case, including shared setup and the short tracking run | About 1½–2¼ hours | About 1¼–2 hours plus first-use depth transfer |
| Full tracking on a single L40S, 2,048 environments × 5,000 updates | About 4 h 13 min | About 8 h 45 min |

The full-schedule timings are from separate single-L40S production runs. The local examples below exercise a shorter schedule in the standard release image.

### Will it fit on my GPU?

These are sampled **application** peaks from a separate run in the standard release image on a 48 GB RTX 6000 Ada. Tracking used 64 environments, 20 updates and metrics-only evaluation. Other workloads and driver allocations need additional memory.

| Stage | Stairs | Kettlebell |
|---|---:|---:|
| Human reconstruction | 16,228 MiB (15.8 GiB) | 29,438 MiB (28.7 GiB) |
| G1 retargeting and preview | 1,116 MiB observed (1.1 GiB) | 4,848 MiB (4.7 GiB) |
| Short tracking | 4,022 MiB (3.9 GiB) | 22,738 MiB (22.2 GiB) |

Stairs retargeting had a gap in sampling. Its observed value is a lower bound. **Inference for smaller cards:** the sampled stairs stages fit inside 24 GiB on paper. Kettlebell reconstruction exceeded 24 GiB, and tracking leaves little room on a 24 GiB card. On 32 GiB, kettlebell reconstruction leaves about 3 GiB before other allocations. Free the GPU for the run; reducing the tracking environment count cannot reduce reconstruction memory. Neither card nor full-schedule memory use was tested in this comparison.

Continue with [Installation](installation.md), then [Checkpoints and example data](downloads.md).
