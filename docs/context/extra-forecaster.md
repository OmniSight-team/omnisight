# Extra — The forecaster, trained to forecast: context

## OMNI-X17 · The decision, before any training

### Why this ticket exists

OMNI-914 to OMNI-918 trained the forecaster to say "Event" at full strength on every second of an event window, even
30 s before it, and measured it with a pooled AUC that compares different places. Ahmad replaced them with
`extra-forecaster.md` on Fri 27 Nov. Its rule has to be written before any training step, so the choice cannot be
fitted to the numbers.

### What was done

2026-11-28. **D-65** in `docs/decisions.md` (index row and section), holding every definition X18 to X22 use: the
eight measures and their ranges, Fix 1 (`τ0`, `s` per dataset) and Fix 2 (64 windows, the gaps, the margin 0.1,
`λ_rank = 1`), the total error, versions A, B, C, the epoch rule (UCF-Crime flat-line ≤ 0.60, then the highest
UCF-Crime within-window AUC), the served model, the Fix-2 and sharing comparisons, SM-2 and SM-3.

**Choices D-65 makes that the ticket leaves open:**
- Fix 2's extra batch: windows drawn from all train windows of the dataset, event and normal as they come; a window
  too short for the gap is not drawn; each second is an ordinary sample (sequence from the window's start, D-55)
- Lead time (mTTA) is the mean over warned event windows at `θ_d`; D-64's AP-averaged mTTA is dropped. Alarms per
  hour use the normal windows' summed spans
- Ties: the first epoch; for the served model the lower seed, then A. The patience counts only epochs that pass the
  flat-line check and improve. A seed with no passing epoch has no `best.pt`
- Run ids `pfm-v5{a,b,c}-s{seed}-e{epoch}`, folders `models/pfm/v5-{a,b,c}/`

**Said openly in D-65:** OMNI-916's event-only model rose more than v2 (+0.170 against +0.076), and the flat-line
check is the guard against a score that only climbs with time; v1's and v2's held-out numbers were already seen; the
UCF-Crime `close` range is [8, 13) s because train and validation windows end 8-9 s before the event.

### Verification performed

- D-65's numbers checked against the ticket line by line (ranges, `τ0`/`s`, 64 windows, gaps 5 s / 1 s, margin 0.1,
  `λ_rank`, model and optimiser settings, 0.60, 0.70, 1.80 s)
- The facts it cites are the ticket's (checked Fri 27 Nov) and earlier entries' (OMNI-916's rise, K-36); none
  re-measured here

### Not verified

- Committed before any training (bbefc88)

### Result

D-65 written. Next: OMNI-X18, the eight measures.

---

## OMNI-X18 · The eight measures

### Why this ticket exists

D-65's choice and SM-2 rest on measures that ask "does the score rise inside a window as the event nears, and stay
flat on normal footage". OMNI-917's `pfm_measures.py` had five of them with D-64's ranges; X18 brings it to D-65.

### What was done

2026-11-28. `scripts/pfm_measures.py` rewritten to D-65's eight, all on `score = 1 − p(Normal)`:
- `RANGES`: UCF-Crime `close` τ [8, 13) s (was [0, 10)), `far` [20, 30) s; CCD unchanged. `HELDOUT_CLOSE` [0, 8) s,
  added only with `measures(..., heldout=True)` as `within_window_auc_0_8`. `FLAT_LINE_MAX = 0.60`
- New: `threshold` (the 10% rule on `score` over normal windows' maxima), `at_threshold` (lead time over warned
  event windows, detection rate, share of normal windows warned, alarms per hour with each second standing for one
  grid step: 1 s UCF-Crime, 0.5 s CCD)
- `average_precision` alone; D-64's AP-averaged mTTA dropped. Per-window and pooled AUC kept
- `measures(data, p, theta=None, heldout=False)`: `theta=None` fits `θ_d` on that split (validation only)
- `kept`, and `uv run python scripts/pfm_measures.py kept validation` (or `held_out`): windows per range, no model
- `scripts/pfm_heldout.py` now passes validation's `θ` to `measures` (else it would have refit `θ_d` on held-out)
- `scripts/tests/test_pfm_model.py`: the epoch-choice test's expected score is 1.0, not 0.875 (with `close` from
  8 s, τ 8 is the window's only close second); what it tests (within-window beats pooled) is unchanged

**Said plainly:** the warning is now `score ≥ θ_d`, where `train_pfm.evaluate` warns when the top class is an
event and `max(p) ≥ θ`. For an Event/Normal model they agree whenever `θ_d > 0.5`. `train_pfm.validation` still
merges `evaluate`'s keys with these; X21 wires the epoch rule to the eight alone.

### Verification performed

- Tests on hand-made windows with answers worked out by hand: the ticket's example (10% → 30% → 70% gives 1.0, flat
  0.5); τ 2 s is outside UCF-Crime's `close` but inside the held-out [0, 8) s one; the flat-line check counts back
  from the last second (0.75 over two windows, one skipped); `θ_d` lets exactly 1 of 10 normal windows warn; lead
  time 5.0 s, detection 2/3, false alarms 0.5 of windows and 3 alarms in 8 s (1350 per hour), CCD's 0.5 s step; AP
  0.8333 and per-window AUC 0.75; a model with classes equals a merged one. 50 PFM tests pass
- **Windows each range keeps** (`kept`, from `datasets/samples/*.csv`, no model):

| Dataset · split | Event windows | with `close` | with `far` | both | Normal windows | both |
|---|---|---|---|---|---|---|
| CCD · validation | 225 | 225 | 225 | **225** | 450 | **445** |
| UCF-Crime · validation | 114 | 114 | 72 | **72** | 240 | **222** |
| CCD · held-out | 243 | 243 | 243 | **243** | 450 | **444** |
| UCF-Crime · held-out | 280 | 206 | 142 | **142** | 300 | **262** |

  UCF-Crime held-out with `close` [0, 8) s: 280 have it, 142 both. Validation's 42 skipped event windows (21 of 30 s,
  21 of 60 s) are videos whose event comes too early for 20 s of lead-up

### Not verified

- On a trained model: X21 runs them every epoch

### Result

The eight measures, D-65's ranges, tested. UCF-Crime's main measure rests on 72 validation and 142 held-out windows.
Next: OMNI-X19, Fix 1.

---

## OMNI-X19 · Fix 1: distance weights on the class error

### Why this ticket exists

Every second of an event window was labelled "Event" at full strength, even 30 s before the event, so training asked
the model to say "Event" where nothing is visible yet. Accident anticipation (DSA, DSTA) weights each second's error
by its distance to the event; D-65 sets the weights.

### What was done

2026-11-28. `scripts/pfm_model.py`:
- `Config.distance_weights`: per domain `[τ0, s]`; empty by default, so every older checkpoint loads and trains as
  before (a config saved before it has no such key and gets the default). `Config.weights_of(domain)`
- `DISTANCE_WEIGHTS = {"ucf_crime": [8, 10], "ccd": [0, 1]}`, D-65's settings, for X21's notebook to pass
- `distance_weight(τ, τ0, s) = exp(-max(0, τ - τ0) / s)` on event samples, 1 on normal ones (`τ` NaN)
- `domain_loss(..., weights)`: with weights, the CE is `Σ w · CE / Σ w`; without, the plain mean as before. The time
  error (MSE of `Δt` on event samples) is untouched; `loss` passes each domain's own weights

### Verification performed

- Three new tests: the weights match the ticket's table (τ 8 → 1.00, 15 → 0.50, 25 → 0.18, 35 → 0.07) and CCD's
  `exp(-τ)`; a badly wrong event sample raises the loss less when it is far (τ 28) than when it is close (τ 8), with
  the weighted means worked out by hand; a normal sample weighs 1; off by default and for a pre-D-65 config the loss
  is the old one; on, the CE is weighted and the MSE unchanged (4.0). 53 PFM tests pass
- A 3-epoch run on 32 hand-made CCD samples with the weights on: it trains, the loss falls (1.18, 1.07, 0.98)

### Not verified

- Its effect on a real model: X21's versions A to C

### Result

Fix 1 in the model, off by default, tested. Next: OMNI-X20, Fix 2.

---

## OMNI-X20 · Fix 2: ranking inside the same window

### Why this ticket exists

Fix 1 tells the model how much each second counts, not that the score should climb as the event nears. Early
detection work (Ma et al., CVPR 2016) adds a ranking error inside the same sequence: a later second of an event
window must score higher than an earlier one. On normal windows D-65 adds the opposite, no climbing for no reason,
so a score that only grows with time is penalised in training too, not only by the flat-line check.

### What was done

2026-11-28.
- `scripts/pfm_model.py`: `Config.rank_gaps` (per domain, empty = off, so older checkpoints load as before),
  `rank_windows = 64`, `rank_margin = 0.1`, `lambda_rank = 1`; `RANK_GAPS = {"ucf_crime": 5, "ccd": 1}` (D-65);
  `ranking_loss(early, late, event, margin)`: the mean of `max(0, margin − (late − early))` on event windows and
  `max(0, late − early)` on normal ones, with scores `1 − p(Normal)`
- `scripts/train_pfm.py`: `draw_pairs(data, rng, n, gap)`: `n` windows without repeats among those spanning at least
  the gap (event and normal as they come), one pair from each drawn uniformly among its pairs at least the gap
  apart; the windows are indexed once per split. In each step, every domain with a gap draws its pairs from the
  epoch's `(seed, epoch)` generator, scores both seconds in one forward pass (dropout on, as the usual batch), and
  adds `λ_d × λ_rank × ranking` to the total (`λ_d` is 1, so this is D-65's sum). The history gets `{domain}_rank`

### Verification performed

- Three new tests: `ranking_loss` on six hand-made pairs (a rise of 0.3 gives 0, of 0.05 gives 0.05, a fall of 0.2
  gives 0.3; a flat normal pair 0, a climbing one 0.4, a falling one 0); `draw_pairs` over 200 draws: both seconds
  from the same window, at least the gap apart, a window too short never drawn, every allowed pair drawn (9 of 9),
  the same generator gives the same pairs; training: off by default and for a pre-D-65 config, on it adds a
  non-negative `ccd_rank` and changes the loss, and a run stopped after epoch 1 and resumed gives the same losses as
  one that never stopped. 56 PFM tests pass

### Not verified

- Its effect on a real model, and its cost per step on Colab: X21's versions B and C
- The full suite (Docker was not running, so the Postgres tests could not run); the PFM tests do not need it

### Result

Fix 2 in training, off by default, tested. Next: OMNI-X21, training on Colab.

---

## OMNI-X21 · Training on Colab: three versions

### Why this ticket exists

D-65's three versions (A: Fix 1; B: Fix 1 + Fix 2; C: B on UCF-Crime alone) answer what each fix adds and whether
learning crashes helps forecast crimes. Each version has 3 seeds, and each seed's epoch is chosen by D-65's rule.

### What was done

2026-11-28.
- `scripts/train_pfm.validation`: per domain `pfm_measures.measures` (the eight, `θ_d` fit on validation) and `Δt`'s
  mean absolute error; `passes` is "UCF-Crime's within-window AUC and flat-line check both exist and the check is
  ≤ 0.60"; `score` is the within-window AUC if it passes, else `-inf`. `train` already keeps `best.pt` only on a
  higher score (the first epoch on a tie) and counts the patience from it, so an epoch that fails never becomes
  `best.pt`, and a seed with no passing epoch writes none and stops after 30 epochs. The X16 measures
  (`evaluate`) are no longer in the history
- `notebooks/pfm_train_forecast_gpu.ipynb` (new): the vectors as OMNI-905; versions in the order B, A, C, 3 seeds
  each, each config written out (`minus_first`, `labels="event"`, `DISTANCE_WEIGHTS`, `RANK_GAPS` for B and C,
  `α_d` per version's datasets, 120 epochs, patience 30, everything else the `Config` defaults D-65 names); folders
  `models/pfm/v5-{a,b,c}/seed{0,1,2}/`; resumable (Run all again); `RUN_ONLY` to train only some versions (C on
  Sunday). Cell 4 prints, per version, how many epochs each seed trained and passed, and the best epoch's eight
  measures for each dataset, checks `best.pt` holds that epoch (or that a failed seed has none), and writes
  `results/x21_v5{a,b,c}.json` (config, histories, best per seed, training time)
- Choosing the served model, the A/B and B/C comparisons and held-out are X22's; this notebook does not touch
  held-out data or `model_runs.json`

### Verification performed

- Two new tests and one updated: an epoch whose event window rises perfectly but whose normal window climbs (flat-line
  1.0) scores `-inf`, never becomes `best.pt`, and with patience 2 training stops two epochs after the last passing
  one; no passing epoch writes no `best.pt`; the epoch-choice test's normal window now has a `close` second (else it
  cannot pass), and the history holds the eight measures and `Δt`'s error. 58 PFM tests pass
- **Dry run of the notebook's own cells on this laptop's CPU** (cell 1 replaced by local paths), real vectors from
  Drive (3 videos per domain, split and kind; UCF-Crime validation event videos with 20 s of lead-up), 2 seeds × 2
  epochs, every version: all six seeds trained, the gate rejected epochs (B seed 0's first, both of A seed 0's, which
  then had no `best.pt` and was reported "failed"), cell 4's checks passed and the three files were written. A
  second Run all resumed every seed with no new epoch. The numbers mean nothing and are not recorded
- `draw_pairs` on the full train splits: UCF-Crime 66,863 samples, 1,858 windows drawable, indexed once in 4.4 s,
  then about 1 ms per draw; CCD 3,132 windows, 1.9 s, about 1 ms

### Not verified

- **The Colab runs themselves** (Ahmad): order B, A, C; C may finish on Sunday
- How long a version takes on the GPU, with the ranking batch's extra forward pass

### Result

Ready for Colab: `notebooks/pfm_train_forecast_gpu.ipynb`, after this is pushed. Next: OMNI-X22 once A and B end.

---

## OMNI-X22 · The choice, held-out once, the record

### Why this ticket exists

D-65 fixed how the served model is chosen and how it is judged before any of X21's numbers existed. This ticket
applies that rule to X21's nine seeds, measures the one chosen model on held-out once, and records SM-2 and SM-3.

### What was done

2026-11-28.
- **X21's runs ended** (Colab, NVIDIA A100, commit `cf78a39`). Every seed of every version had a passing epoch;
  the epochs that failed the flat-line check were B 3 of 141, A 16 of 117, C 2 of 111. GPU time per epoch: A about
  13 s, B about 21 s, C about 12 s (the ranking batch's extra forward pass costs about 60%)
- `scripts/pfm_choose.py` (new): reads `results/x21_v5{a,b,c}.json`, applies D-65's rule (the highest UCF-Crime
  validation within-window AUC among A's and B's seeds; on a tie the lower seed, then A), prints the validation
  table and the two comparisons, and writes `results/x22_choice.json`. With `--register` it copies the seed's
  `best.pt` (after checking it holds the chosen epoch) to `models/pfm/runs/pfm-v5a-s1-e14/model.pt`, writes its
  `run.json` (`θ_d` from the history, the rule `1 - p(Normal) >= theta`) and makes it `model_runs.json`'s chosen
  run, in place of `pfm-joint-v2-mf-s2-e55`. Nothing is refit
- **The served model: `pfm-v5a-s1-e14`** (version A, seed 1, epoch 14): UCF-Crime within-window AUC 0.8172,
  flat-line 0.568; `θ` CCD 0.0581, UCF-Crime 0.6281. Among A's seeds it is the weakest on the other UCF-Crime
  measures (detection 0.167, AP 0.450, 24.1 false alarms per hour), and its flat-line is close to the 0.60 limit,
  as are all of A's (0.568-0.588)
- **What Fix 2 adds:** A 0.7911 (spread 0.0522), B 0.7419 (spread 0.0331): B is 0.049 lower, less than the larger
  spread, so **no difference is claimed**. Beside it, not a claim: B's flat-line is lower on every seed
  (0.445-0.482 against 0.568-0.588)
- **Does sharing help:** B 0.7419, C 0.7521 (spread 0.0839): +0.010, **not claimed**; C is not clearly better, so
  no follow-up ticket for separate models
- `notebooks/pfm_heldout_gpu.ipynb`: a cell 5 that prints D-65's eight measures (and held-out UCF-Crime's τ [0, 8) s
  within-window AUC) for the model and the control, and SM-2's and SM-3's pass or fail. Cell 3's first line and the
  figure's CCD lead times keep D-59's warning rule; on a D-65 model cell 5 holds the numbers (K-37)
- K-37: OMNI-911's alert rule `max(p) ≥ θ_d` does not fit D-65's `θ_d`

### Verification performed

- Four new tests (`test_pfm_choose.py`): the highest seed of A or B is served, never C; a tie goes to the lower seed,
  then to A; a failed seed is skipped, and none passing serves nothing; a difference is claimed only above the
  larger spread. 44 tests in the four PFM test files pass
- The chosen `best.pt` reloaded **on the laptop's CPU** and run on the whole validation split of both datasets:
  `θ`, within-window AUC, flat-line, mTTA, detection, false alarms per hour and AP all equal the Colab history
  (to 1e-4)
- Cell 5 run on **validation** (not held-out) with the registered run: it runs, and gives the history's numbers.
  One CCD normal window sits exactly at `θ`, and the CPU's rounding tips it (false alarms 0.102 against 0.100)
- `pfm_runs.py list` shows `pfm-v5a-s1-e14` as the chosen run

### Not verified

- Held-out (Ahmad, on Colab): `notebooks/pfm_heldout_gpu.ipynb`, `RUN_ID` empty, after this is pushed
