uv python pin 3.12.11
uv sync --dev
uv pip install flash_attn  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
uv pip install -e .
