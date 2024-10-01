# DCRNN for Dragonfly system application runtime forecasting

The folder scripts contains two ipynb files to preprocess the data.

- `data_preprocessing.ipynb` is used to create a 'merged df'
- `generate_graph_data.ipynb` is used to transform the previously created merged_df into training, validation, test datasets that are compatible with the DCRNN Pytorch implementation.

The datasets are stored in `data/processed`.

Training can be performed by setting the parameters in `DCRNN_PyTorch_config/dcrnn_network.yaml` and running `dcrnn_train_pytorch.py` as described in the ![original implementation](https://github.com/chnsh/DCRNN_PyTorch).
