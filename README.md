# qwen3.5-deploy

Windows 在 wsl 使用 `llama.cpp` 部署 `qwen3.5`.

参考教程：[https://unsloth.ai/docs/models/qwen3.5](https://unsloth.ai/docs/models/qwen3.5)

## 一、环境准备

在有 GPU 的 Ubuntu 上安装 llama.cpp 时，从源码编译 llama.cpp

```bash
sudo apt-get update
sudo apt --fix-broken install -y
sudo apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j 2 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

> **Note:**
> 1）如果出现 `CUDA Toolkit not found` 或 `Unable to find cudart library`，先安装 CUDA Toolkit 并确认 WSL 可识别 GPU，再重新执行 CMake。
> 
> ```bash
> nvidia-smi
> sudo apt-get install nvidia-cuda-toolkit -y
> ls /usr/local/cuda/lib64/libcudart.so
> ```
> 2）在无 GPU 的 Ubuntu 上编译 llama.cpp 时，需要将 `GGML_CUDA` 设为 `OFF`

## 二、下载模型文件

```bash
# 创建用于存放模型的文件夹
mkdir model

# 安装 modelscope
pip install modelscope -U

# 下载模型文件
modelscope download --model unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf --local_dir ./model
```

## 三、模型推理

### 3.1 在命令行中对话

```bash
./llama.cpp/llama-cli --version

./llama.cpp/llama-cli \
    --model ./model/Qwen3.5-4B-UD-Q4_K_XL.gguf \
    --ctx-size 16384 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00
```

### 3.2 使用 API 服务

**1）启动 API 服务**

```bash
./llama.cpp/llama-server --version

./llama.cpp/llama-server \
    --model ./model/Qwen3.5-4B-UD-Q4_K_XL.gguf \
    --alias "unsloth/Qwen3.5-4B" \
    --ctx-size 16384 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

> To disable thinking / reasoning, use `--chat-template-kwargs '{"enable_thinking":false}'`

**2）测试 API 服务**

```bash
# 单次对话
python client.py

# 流式输出
python client.py --stream 你好，请介绍一下自己
```

**3）启动 Gradio Web APP**

```bash
cd app

uv run python app.py
```

在浏览器中打开 http://localhost:7860/ 即可与 Gradio Web APP 进行交互。
