# Recognition Web App

Local browser UI for CAS(ME)^3 recognition-only inference.

## Run

```powershell
python scripts/app/run_recognition_web.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Inputs

- Onset image + apex image: the app computes TV-L1 optical flow and runs the 4-class model.
- `.npy` tensor: expects a saved motion tensor shaped `(4, H, W)`.

The default model source is `artifacts/runs_dev2/flow4_main_balanced_softmax_5fold`.
