# MuonVariants

Forked from the implementation of the `Muon` optimizer described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014) and [this writeup](https://kellerjordan.github.io/posts/muon/).

AdaMuon
-------
Muon + Novograd style second moment normalization

MomMuon
--------
Muon with NS formulation borrowed from [SWAN paper ](https://github.com/ethansmith2000/SWANOptimizer/tree/main) which produces a preconditioner, not the orthogonalized gradient.
From there, we can keep a running average of this preconditioner.
We can also opt to only NS every N steps.

OGSignMuon
-----------
After performing NS, replace the signs of matrix entries with that of the original pre-whitened grad

## Installation

```
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW instead.
Muon provides an internal AdamW for this so you don't have to use an extra optimizer.

```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

from muon import Muon
# Find ≥2D parameters in the body of the network -- these will be optimized by Muon
muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
# Find everything else -- these will be optimized by AdamW
adamw_params = [p for p in model.body.parameters() if p.ndim < 2]
adamw_params.extend(model.head.parameters())
adamw_params.extend(model.embed.parameters())
# Create the optimizer
optimizer = Muon(muon_params, lr=0.02, momentum=0.95,
                 adamw_params=adamw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)
```

You'll have to replace `model.body`, `model.head`, and `model.embed` with whatever subset is appropriate for your model.
E.g., for a ConvNet, `muon_params` should be all the convolutional filters, and `adamw_params` should be everything else.

## Hyperparameter tuning

If you're replacing an already-tuned AdamW with Muon, the only thing you should need to tune is Muon's learning rate.
The AdamW hyperparameters should be set to whatever you were already using.

## Benchmarks

For a comparison between AdamW, Shampoo, SOAP, and Muon for training a 124M-parameter transformer, see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

## Connection to Shampoo

See [this thread](https://x.com/kellerjordan0/status/1844782418676339059) for more info including the connection to Shampoo.

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.7 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)

## Citation

```
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cecista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}
```

