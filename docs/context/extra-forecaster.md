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

- Nothing is committed yet: D-65 counts as "before training" once Ahmad commits it, before the first Colab step

### Result

D-65 written. Next: OMNI-X18, the eight measures.
