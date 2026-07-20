---
title: "Installation"
lead: "For most users, no installation is needed"
weight: 1
---

## <a id="installation"></a>🛠️ + ⚙️ Installation 

**For XLA**, it will by default auto-install the required XLA PJRT plugins (for CPU, GPU and TPUs; Linux and Macs)
in the user's local lib directory (`$HOME/.local/lib/go-xla` in Linux; `$HOME/Library/Application Support/go-xla` in Mac;
`$HOME\AppData\Local\go-xla` in Windows).
It can be disabled by setting `GOMLX_NO_AUTO_INSTALL` or programmatically by calling `xla.EnableAutoInstall(false)`).

If you want to manually pre-install for building production dockers, a specific version, or such custom setups,
see [github.com/gomlx/go-xla](https://github.com/gomlx/go-xla) for details, 
there is a self-explanatory simple installer program.

If you want to use only a pure **Go backend**, simply do `import _ "github.com/gomlx/gomlx/backends/simplego"` and 
there is no need to install anything.

## 🐳  [Pre-built Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)

The easiest to start playing with it, it's just [pulling the docker image](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)
that includes **GoMLX** + [JupyterLab](https://jupyterlab.readthedocs.io/) + [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter) and 
[Nvidia's CUDA runtime](https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-runtime-ubuntu22.04/images/sha256-08aed54a213b52e9cb658760b6d985db2f4c5f7e8f11ac45ec66b5c746237823?context=explore)
(for optional support of GPU) pre-installed -- it is ~5Gb to download.

From a directory you want to make visible in Jupyter, do:
> For GPU support add the flag `--gpus all` to the `docker run` command bellow.

```bash
docker pull janpfeifer/gomlx_jupyterlab:latest
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jupyter/work janpfeifer/gomlx_jupyterlab:latest
```

It will display a URL starting with `127.0.0.1:8888` in the terminal (it will include a secret token needed) that you can open in your browser.

You can open and interact with the tutorial from there, it is included in the docker under the directory `Projects/gomlx/examples/tutorial`.

More details on the [docker here](docker/jupyterlab/README.md).

It runs on Windows as well: _Docker Desktop_ uses WSL2 under the hood.
