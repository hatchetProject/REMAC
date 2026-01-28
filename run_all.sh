export CUDA_VISIBLE_DEVICES=0,1


uv run src_lora/train_flow_lora.py --config.pretrained-model-path logs-bc/base_model/31/policies/ --config.seq lr=1e-4 --config.run-path ./logs-expert/ \
    --config.dir-name run_all --config.num-epochs 32 --config.learning-rate 0.0001 --config.json-file "config/config.json"


uv run src_lora/train_flow_lora.py --config.pretrained-model-path logs-bc/base_model/31/policies/ --config.seq lr=1e-5 --config.run-path ./logs-expert/ \
    --config.dir-name run_all --config.num-epochs 32 --config.learning-rate 0.00001 --config.json-file "config/config.json"


uv run src_lora/train_flow_lora.py --config.pretrained-model-path logs-bc/base_model/31/policies/ --config.seq lr=3e-4 --config.run-path ./logs-expert/ \
    --config.dir-name run_all --config.num-epochs 32 --config.learning-rate 0.0003 --config.json-file "config/config.json"


uv run src_lora/train_flow_lora.py --config.pretrained-model-path logs-bc/base_model/31/policies/ --config.seq lr=3e-5 --config.run-path ./logs-expert/ \
    --config.dir-name run_all --config.num-epochs 32 --config.learning-rate 0.00003 --config.json-file "config/config.json"


uv run src_lora/eval_flow.py --run-path None --config.seq all --config.model-path ./logs-bc/run_all \
    --config.output-dir run_all --config.save-path debug_hard.csv --config.use-ema

python src_lora/calc_metric.py --seq all --eval-dir "run_all" --output-dir "eval_output" --output-filename "run_all.png"
