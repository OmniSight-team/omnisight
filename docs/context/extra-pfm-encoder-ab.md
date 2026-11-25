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

