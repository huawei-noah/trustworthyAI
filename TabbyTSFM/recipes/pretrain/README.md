# Tabby-Pretrain recipe

`train.py` is the reorganized mixed-data trainer. The 165K release entry point
is `train_165k.sh` and fixes the model width at `d_model=768`.

The recipe uses:

- BLAST training shards;
- pre-generated KernelSynth Arrow shards;
- online CauKer V2 samples.

The offline KernelSynth generator is not included, only its
Arrow reader.

```bash
export BLAST_DATA_ROOT=/path/to/BLAST/train
export SYNTHETIC_ARROW_ROOT=/path/to/kernel_synth_arrow
export BLAST_RATIO=<confirmed-fraction>
export CAUKER_V2_RATIO=<confirmed-fraction>

bash recipes/pretrain/train_165k.sh outputs/tabby-pretrain-165k
```

Set `RESUME=1` only when the output directory already contains a valid `latest`
checkpoint created by this trainer.
