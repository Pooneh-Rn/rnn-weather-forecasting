# RNN-1: Multivariate Weather Forecasting

Forecasting temperature from the Jena Climate dataset using recurrent neural networks. Built as
part of a Deep Learning course project (Spring 2026) — the base task compares LSTM, GRU, and a
simple sequence-to-sequence model, and the original contribution tests whether adding attention
over an LSTM's past hidden states improves multi-step forecasts.

## Summary

- **Input**: past 120 hours (5 days) of 6 weather variables (pressure, temperature, humidity,
  specific humidity, wind speed, wind direction), resampled to hourly resolution.
- **Output**: next 24 hours of temperature, forecast directly (LSTM/GRU/Attention) or
  autoregressively one step at a time (Seq2Seq).
- **Base task**: compare LSTM, GRU, and a simple seq2seq model.
- **Original contribution**: add Luong-style attention over the LSTM's past hidden states and test
  whether it improves forecasts, and at which horizons.

| Model | Mean MAE (°C) | Mean RMSE (°C) |
|---|---|---|
| LSTM | 1.831 | 2.336 |
| GRU | **1.765** | **2.264** |
| Seq2Seq | 1.819 | 2.320 |
| Attention LSTM | 1.819 | 2.305 |

GRU came out ahead of the other two base architectures across the whole forecast horizon.
Attention improved on the plain LSTM baseline at short horizons (1-10 hours ahead) but underperformed
it at longer horizons (11-24 hours ahead) — the opposite of the original hypothesis. Full
writeup, including the attention-weight analysis and discussion answers, is in
[`report.md`](./report.md).


## Running it

The notebook is self-contained and runs top to bottom in Google Colab with a GPU runtime
(tested on an L4). It downloads the Jena Climate dataset automatically — no manual data setup
needed.

1. Open `notebook.ipynb` in Colab.
2. Set **Runtime > Change runtime type > GPU**.
3. **Runtime > Run all**.


## Dependencies

Installed automatically by the first cell of the notebook: `torch`, `pandas`, `numpy`,
`matplotlib`, `tensorflow` (used only for its dataset download utility). No local setup needed if
running in Colab.

## Reproducibility

Random seed fixed at 42 throughout (data split, model init, training). All data splits are
chronological (train < val < test) to avoid leakage in the time series.
