FROM vllm/vllm-openai:v0.21.0

RUN pip3 install --no-cache-dir "vllm[audio]==0.21.0"
