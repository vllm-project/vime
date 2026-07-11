# Mooncake Store Rollout Transfer

Transfer rollout tensors (`tokens`, `loss_masks`) via [Mooncake Store](https://github.com/kvcache-ai/Mooncake) instead of Ray object refs when rollout and training run on separate GPUs.

```bash
pip install mooncake-transfer-engine
mooncake_master --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 --http_metadata_server_port=18080
python train_async.py --transfer-backend mooncake_store
```

Optional: `--mooncake-store-init-kwargs '{"master_server_addr":"127.0.0.1:50051"}'` or `MOONCAKE_*` env vars.

Ported from [slime PR #1709](https://github.com/THUDM/slime/pull/1709).
