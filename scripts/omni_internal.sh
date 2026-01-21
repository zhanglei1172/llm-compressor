# text ptq
CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/python ./examples/qwen3omni_example.py

# vit ptq
CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/python ./examples/qwen3omni_vit_example.py

# audio ptq
CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/python ./examples/qwen3omni_audio_example.py


# enable_modality 选择多模态
# vit 旋转训练
TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6,7 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/torchrun --nnode 1 --nproc_per_node 2 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12352 examples/qwen3omni_interal/internal_qwen3omni_example_spinquant_all_modalty.py --config examples/qwen3omni_interal/train_aut.yaml > logs/train_vit.log

# text 旋转训练
TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/torchrun --nnode 1 --nproc_per_node 4 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12352 examples/qwen3omni_interal/internal_qwen3omni_example_spinquant_all_modalty.py --config examples/qwen3omni_interal/train_text.yaml | tee logs/train_text.log

# audio 旋转训练
TORCHINDUCTOR_AUTOGRAD_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4,5 HF_ENDPOINT=https://alpha.hf-mirror.com HF_DATASETS_CACHE=/dataset/workspace/zhangl98/hf_cache/ ./.venv/bin/torchrun --nnode 1 --nproc_per_node 2 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12352 examples/qwen3omni_interal/internal_qwen3omni_example_spinquant_all_modalty.py --config examples/qwen3omni_interal/train_aut.yaml | tee logs/train_aut.log