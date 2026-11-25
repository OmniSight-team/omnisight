# Extra — Which clip encoder should the forecasting model read? CCD decides

## OMNI-X16 · CCD trained twice, one model per clip encoder

### Why this ticket exists

X14 (D-50) kept InternVideo2 for search, but Cosmos's clip vectors alone were better on actions, and the PFM reads
clip vectors alone. So the forecasting task itself decides which encoder the PFM reads, on CCD's validation split,
before UCF-Crime is encoded.

### What was done

2026-11-24, part 1: the encoding code (OMNI-902's runner, CCD part). The Colab run is next.

- `scripts/encode_pfm_windows.py`: per video, the grid frames of its windows (D-54: every 0.5 s on CCD, from the
  window's last frame back to the video's start), each grid frame's clip (Algorithm 2's, ai_workers' own window
  functions), any encoder with `encode_clips`. CCD videos have about 8 grid points, so clips are gathered across
  videos and encoded in full batches of 16. One file per video, written atomically: a file that exists is done, so a
  rerun skips it (no manifest rewrite per video: 4,500 small videos on Drive). The manifest keeps the pins and the
  grid and refuses a mix, and the failures; a failed read or batch is recorded and tried again on the next run.
  Decoding stops after the last frame a clip needs
- Cosmos runs through the file's command line (`cosmos`) in its own process on transformers 4.51.3, with X14's
  `CosmosEncoder`, as D-50 requires; InternVideo2 in the notebook's process
- `notebooks/pfm_windows_gpu.ipynb`: copies CCD's 4,500 videos from Drive to the Colab machine once (16 threads),
  then InternVideo2 → `embeddings/pfm/ccd/internvideo2/`, Cosmos → `.../cosmos/`, the check, and
  `results/x16_ccd_encoding.json`

### Verification performed

- `scripts/tests/test_encode_pfm_windows.py`, 6 tests: the grid of a positive and a normal window (and with a
  context limit), the clip is Algorithm 2's and cut at 0 s, every video's clips read back in order across chunks and
  batches, a rerun skips done videos and a folder refuses another encoder or step, a short video and a failed batch
  are recorded and done on the next run, and the real `ccd.csv` gives a grid for all 4,500 videos
- **Real model, real footage, this laptop's CPU:** InternVideo2 on two local CCD videos (`Crash-1500/000026`, window
  [0, 3.6 s); `Normal/000024`, [1.547, 4.847 s)): 8 and 10 clips on the expected grid frames, 512-D unit vectors,
  neighbouring vectors 0.97-1.00 alike. 10.8 s per clip on the CPU (why it runs on Colab)
- The whole CCD grid: **39,233 clips per encoder** (the ticket estimated about 36,000)

### Not verified

- Nothing on the GPU yet; the Cosmos command line runs only in the 4.51.3 process, not on the laptop

### The Cosmos run, first pass (2026-11-24, L4)

4,416 of 4,500 videos done, 84 failed, all `encode: ValueError: all input arrays must have the same shape`.
A few CCD videos have another resolution, and a batch holds clips of several videos (X14's batches held one
video each, so it never met this). Cosmos's encoder stacks a batch into one array; InternVideo2 resizes per frame
and was not affected. Fix: `by_frame_size` in `encode_virat_cosmos.py` encodes a mixed batch once per frame size,
the vectors back in order (the processor resizes every size to 336 x 336, so the vectors are the same). Test:
`test_a_batch_that_mixes_frame_sizes_is_encoded_per_size_in_order`. 397.5 ms per clip on the L4 for the 4,416.

### OMNI-903 · CCD's samples (2026-11-25)

**Why:** both encoders' models must train on exactly the same samples, so the only difference is the vectors.

**What was done**
- `scripts/pfm_samples.py`: a sample at every grid point inside a window (positives: every distance to the
  accident; negatives: every point of the normal window), its class, its split and, on a crash, `τ = event_start -
  t`. UCA-timed UCF-Crime windows measure `τ` to 8 s after their end (D-49), ready for UCF-Crime
- **Detour, before building:** the ticket says CCD's sequence is "all the footage before `t`". Counted first: every
  crash window starts at 0 s, normal windows later, so the lengths would give away the class (1 vector: 12.8% of
  crash samples, 3.1% of normal). The sequence starts at the window's start instead (D-55); the lengths then match.
  Nothing re-encoded: a window's grid points are a subset of its video's
- `datasets/samples/ccd.csv` committed (frames only, no vectors), as D-26 keeps the split and window files
- `Store`: reads each video's grid vectors once from an encoder's folder (about 120 MB for Cosmos's 39,233 x 768,
  not copied per sample) and pads a batch at the end with a mask

**Verification performed**
- `scripts/tests/test_pfm_samples.py`, 5 tests: a crash window's samples, lengths and `τ`; a normal window's sequence
  starts at its window, not at the video's grid; the UCA margin in `τ`; the store's sequences, padding, mask and
  its refusals; the committed file is the build, every sample in its video's split, the length shares within 0.5
  point between classes
- **Real vectors:** `check ccd` against InternVideo2's folder on Drive (read from this laptop through Google Drive
  for desktop, 59 min for 4,500 files): all 35,161 samples found as consecutive grid vectors, 39,233 vectors of
  512-D, all unit length

**Not verified:** the same check on Cosmos's folder (another hour through Drive here; it runs on Colab when the
models train, OMNI-904)

### OMNI-904 · The model, and the notebook for its proofs and X16 (2026-11-25)

**Why:** X16 needs the model before it can train it twice; OMNI-905 then trains the same model on both domains.

**What was done**
- `scripts/pfm_model.py`: Algorithm 4 as B.4 writes it (backbone, a head per domain, `κ`, the loss). What Phase A
  leaves open is D-56. In `scripts/` for now, as the encoding and sample code; OMNI-910 serves it from
  `services/pfm`
- `scripts/train_pfm.py`: training with one batch per domain per step, a checkpoint after every epoch, resume
  from it (refused for another config or seed), each epoch's order and dropout from `(seed, epoch)`; prediction;
  the X16 measures (AUC, top-1, `Δt` error, lead time at the 10% false-alarm threshold); the ticket's rule
- `notebooks/pfm_train_gpu.ipynb`: both encoders' folders copied from Drive, the overfit proof (32 real training
  samples, 300 epochs), the resume proof (4 epochs straight against 2 + restart + 2, on the whole train split),
  then 3 seeds per encoder → `results/x16_pfm_ccd.json` with the rule's verdict

**Verification performed**
- `scripts/tests/test_pfm_model.py`, 8 tests, on the CPU: each head's shapes and a non-negative `Δt`; padding with
  garbage changes nothing; `κ` 0 on a uniform and 1 on a certain prediction; the loss is `λ(CE + α·MSE)` with
  negatives left out of the MSE; **overfit:** 32 stand-in samples, the loss from above 0.5 to under 0.05, every one
  right; **resume:** 2 epochs, a new process from the checkpoint, 2 more: the same 4 losses as a straight run; the
  threshold and the lead time on hand-made predictions; the rule in its three cases
- **Real vectors, laptop CPU, a smoke run:** 20 videos per class from train and from validation (InternVideo2, from
  Drive). 15 epochs in 5 s, loss 2.72 to 0.29. Validation: AUC 0.993, top-1 0.946, `Δt` error 0.39 s; all 20 crash
  windows warned, mean lead 3.50 s
- **What the smoke run says (K-35):** near-perfect from 40 training videos, and the warning comes on almost the first
  sample. CCD's crashes and normals come from different sources (dashcam videos online, BDD100K), so the model may
  tell the source, not the lead-up. Both encoders may tie near 1.0 in X16, which keeps InternVideo2 under the rule

**Not verified:** the proofs and X16 on the GPU (the notebook, next)

### X16 on the GPU, and the decision (2026-11-25)

**What was done:** Ahmad ran `notebooks/pfm_train_gpu.ipynb` on an L4: both folders copied from Drive, the two
proofs, then 6 trainings of about 30 s each.

**Verification performed**
- Overfit proof: loss 2.697 to 0.0001 on 32 real samples, all right. Resume proof: identical losses (difference 0)
- X16: the numbers are in `results.md`. Checked that 3.49 s is the ceiling: the mean `τ` of the 225 validation
  crash windows' first samples is 3.489 s

**Not verified:** a K-35 control (for OMNI-906)

### Result

**InternVideo2 stays (D-53).** Cosmos's mean AUC 0.9998 against 1.0000, lead time equal: the rule keeps InternVideo2.
Both encoders are at CCD's ceiling, which confirms K-35: every crash is called in its first half-second. Cosmos is
not needed; UCF-Crime is encoded with InternVideo2 only.
