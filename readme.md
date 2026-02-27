# EnerGNN
A graph‑neural‑network based model for crystal‑structure energy/force/stress
prediction, built on [PyTorch](https://pytorch.org)/[PyG](https://pytorch-geometric.readthedocs.io/)
and the `matbench-discovery` datasets. The repository contains:

* `energnn.py`: model, dataset loaders and simple train/test helpers;
* `main.ipynb`: interactive notebook demonstrating data loading, training,
  evaluation, force calculation, etc.;
* `pixi.toml`: dependency list for creating a conda environment;
* helper functions for converting WBM/Alexandria JSON into PyG `Data`.

---

## Environment setup

We recommend using Pixi/Conda (see `EnerGNN/pixi.toml`):

```sh
# from the project root
pixi init       # creates an environment from [pixi.toml]
pixi install
pixi shell      # To enter the python envitonment, run this.
# Then we need to install matbench-discovery manually.
git clone https://github.com/janosh/matbench-discovery --depth 1
pip install -e ./matbench-discovery
```
> `matbench_discovery` is also needed but should be installed manually. [https://github.com/janosh/matbench-discovery](https://github.com/janosh/matbench-discovery)
> 
> 


## Quick start
Prepare data:
```Python
```
Training:
```Python
optimizer = tc.optim.Adam(model.parameters(), lr=1e-4)
train_ds = energnn.DatasetAlexandria(load_files=['000','001','002'], cutoff=8.0)

energnn.tcg_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=lambda: loss_fn,
    dataset=train_ds,
    device='cuda',
    batch_size=256,
    num_workers=4,
    epoch=10,
)
```

Testing:
```Python
test_ds = energnn.DatasetWBM()
energnn.tcg_tester(
    model=model,
    dataset=test_ds,
    device='cuda',
    batch_size=256,
    num_workers=4,
    loss_fn=lambda: loss_fn,
)

# compute forces for a small batch
for batch in tcg_loader.DataLoader(test_ds[:4], batch_size=1):
    f = model.force(
        batch.node_features,
        batch.edge_index,
        batch.edge_weight,
        batch.batch,
    )
    print(f)
```