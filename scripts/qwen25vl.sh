# ptq example
TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1  CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/python ./examples/v5/v5_example.py

# 旋转训练
TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1  CUDA_VISIBLE_DEVICES=4,5 ../vllm-omni/.venv/bin/python -m torch.distributed.run --nnode 1 --nproc_per_node 2 --rdzv_id 1176 --rdzv_backend c10d --rdzv_endpoint localhost:12353 /workspace/zhangl98@xiaopeng.com/code/llm-compressor/examples/v5/ostq.py --config examples/v5/train.yaml