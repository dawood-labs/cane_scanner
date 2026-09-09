# Wiring the guard into `Model_Execution_Pipeline_v3.0.ipynb`

Three changes, all in the static-model cells. The mask-building step stays exactly
as it is; only the classification call changes.

## What is wrong today

`classify_large_image` in cell 11 ends with:

```python
preds = model.predict(valid_pixels)
```

`.predict()` on an XGBoost binary classifier applies a fixed 0.5 cut. Training
selected a threshold by maximising F1 and never saved it, so that choice is lost
and inference silently uses 0.5 instead. Nothing anywhere compares the image
against the distribution the model was fitted on, which is why the 30 August run
returned confident labels for an image the model had no business classifying.

## Change 1: import the package

Add to the imports in cell 11:

```python
from static_training import inference as st_inference
```

## Change 2: replace the classification call

Keep `create_aligned_mask` as it is. It already writes an aligned mask raster, and
that path is exactly what the replacement takes. Where the notebook currently calls
`classify_large_image(...)`, call this instead:

```python
result = st_inference.classify_raster(
    raster_path=input_path,
    model_path=model_file,
    out_label_path=output_path,
    out_probability_path=str(Path(output_path).with_name(Path(output_path).stem + "_prob.tif")),
    mask_path=aligned_mask_path,      # from the existing create_aligned_mask
    positive_out=target_class_out,
    background_out=background_out,
    enforce_domain=False,             # set True once you trust the guard to stop a run
)
print(result.verdict)
print(f"{100 * result.positive_fraction:.1f}% of masked pixels classified as crop")
```

What this buys:

- the decision threshold comes from the model sidecar rather than being assumed
- a probability raster is written next to the labels, so a threshold can be
  changed later without re-running inference over the whole AOI
- the domain guard runs first and reports how far the image sits from the training
  distribution

## Change 3: check the verdict before trusting the output

`result.verdict` carries a level of `ok`, `warn` or `refuse`. Calibrated against
the deployed cane model on Al-Moiz Unit 1:

| Date | Shift | Verdict | Recall vs the RF mask |
|---|---|---|---|
| 10 Aug 2026 | 0.25 | ok | 85.0% |
| 06 Jul 2026 | 0.31 | ok | 80.5% |
| 30 Aug 2026 | 0.91 | warn | 61.8% |
| 04 Sep 2026 | 2.21 | refuse | 35.9% |

Leave `enforce_domain=False` at first so a run still completes and you can see the
verdict alongside the result. Switch it to `True` once the thresholds have been
checked against a few more AOIs.

## Picking a threshold afterwards

With the probability raster written, a threshold can be re-chosen without touching
the model:

```python
st_inference.probability_summary(prob_path, [0.5, 0.4, 0.3, 0.2])
```

On the 30 August image this returns 61.8% of masked pixels at 0.5 and 82.8% at 0.2,
which is the whole trade-off in one line.

## Sidecars

`classify_raster` looks for `<model>.sidecar.json` beside the model file. One has
already been written for the deployed model at
`model_files/fao_cane_xgb_model.sidecar.json`, reconstructed from the cleaned
training table. Models trained through `train_static_model.py` write their own.

Without a sidecar the call still works, falling back to a 0.5 threshold and no
guard, and logs a warning saying so.
