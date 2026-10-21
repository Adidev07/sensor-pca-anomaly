# Sensor Anomaly Detection with PCA

Learn normal sensor relationships and detect deviations using reconstruction error with a validation-set threshold.

Actual model fitting with NumPy. Synthetic data; educational benchmark, not production performance.

## Quick start

Python 3.10+ required.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python project.py
python -m unittest discover -v
```

Open `results/report.html` after running. View [saved metrics](results/metrics.json), [PDF guide](PROJECT_GUIDE.pdf), or [editable guide](GUIDE.md).

## Experiment

Training contains no injected anomalies. Threshold is the 99th percentile of normal validation reconstruction errors, without reading test labels. Flag scores strictly above that threshold. Report test precision, recall, F1 and confusion counts, and compare with an always-normal baseline. The intended validation tail fraction is not a guaranteed false-positive rate on future data. With only 250 validation points, extreme quantile estimates are uncertain.

## What to explain in an interview

How is PCA different from a classifier? It learns a subspace rather than a direct class boundary. Why normalize? Large-scale sensors otherwise dominate. Can it miss anomalies? Yes: unusual points lying inside the learned subspace may reconstruct well. Why not fit on the full dataset? Test distribution and anomalies would influence preprocessing and components. Extend with real licensed sensor data, chronological evaluation, drift monitoring, residuals per sensor, rolling thresholds and comparisons with robust distance or isolation methods. Validate alarms with domain experts.

## Files

- `project.py`: training, inference and experiment.
- `common.py`: metrics and HTML/JSON report output.
- `test_project.py`: four behavioral tests.
- `results/`: model, data and measured experiment outputs.
- `BUILD_INFO.txt`: verification environment.

## Attribution and limitations

AI-assisted learning project. Review and customize before presenting your contributions. Uses synthetic data and deliberately limited models. No external model API or personal data. MIT license.
