# BonsaiBench (iOS)

Paired A-B-B-A benchmarking of the Bonsai Metal research flags on iPhone, with the protocol of the Mac
studies in [`../../m5`](../../m5/README.md) adapted to a phone (see "Differences from the Mac studies").
Research tool; not an App Store app.

## What it measures

| Cell | What | Compare with |
|---|---|---|
| **tg128** | 128 single-token decodes from an empty context, random tokens, no sampling | llama-bench tg128 (the Mac studies' tg128) |
| **chat128** | greedy generation of 128 tokens after a chat prompt, including sampling; the generated tokens are compared between arms | llama-server generation speed; bit-exactness |
| **gen128** | greedy generation of 128 tokens after the chat prompt through llama.cpp's speculative-decoding loop, plain or with MTP per arm (see "MTP"); generation only, the prompt pass excluded | the M5's generation-only rate (45.4 -> 61.4, 1.35x), not the server rows, which include the prompt |
| **pp2 / pp4 / pp8** | k-token batches: the step shapes of MTP verification and concurrent requests | llama-bench ppK |
| **pp512** | a 512-token prompt batch | llama-bench pp512 |

Every observation creates a fresh llama context, so the research flags (environment variables, re-read
when a context is created) take effect per arm. The context uses n_ctx 1024.

- tg, chat and pp cells keep one output row (`n_outputs_max = 1`), which spares ~0.5 GB of logits buffer.
- gen cells use llama.cpp's own settings instead: two outputs for MTP verification, plus the MTP draft
  context.

For each cell, per cycle:

1. An A-B-B-A quartet.
2. Before every observation, the thermal gate. The app must be in front, then comes the cooldown (default
   8 s), and then the state is checked; waiting and cooldown repeat until the state is met, for up to 5
   minutes of thermal waiting. The state a run *starts* in is what counts:
   - the first run of a quartet needs nominal or fair (nominal with "Start quartets only when nominal");
   - each later run needs to be no hotter than the first run was, and never above fair.
3. The quartet is rejected if any of these hold:
   - the app left the foreground;
   - the gate was not met within 5 minutes;
   - any run reached serious/critical at any point (thermal-state notifications are watched during runs);
   - its runs started in different thermal states (a run may warm the phone while it runs, which is fine;
     the next run waits for the gate);
   - an observation failed (a Metal command buffer error, a context that could not be created);
   - the two A runs, or the two B runs, differ by more than the spread gate (default 1.20x).

   Once heat has decided a rejection, the rest of the quartet is skipped, so as not to heat the phone
   further.
4. A rejected quartet is kept and repeated, at most 3 times; the fastest is never chosen.
5. A cell without an accepted quartet for every cycle is marked incomplete.

The speedup is the geometric mean of the accepted quartets' mean(B)/mean(A).

ppK cells warm up for at least 1 s and measure for at least 2.5 s, since one small decode is ~0.1 s on a
phone. Each call's time is recorded, warmup included, even when a call fails.

A result JSON is written to Documents at the start and after every run, so a study that iOS kills or that
is stopped keeps what it measured (`unfinishedQuartet`). It records:

- **Build:**
  - the git revision the app was built from, and how many app files were uncommitted;
  - the revision the framework was built from (recorded by `build-xcframework.sh`);
  - the SHA-256 of the framework binary, before it is signed into the app.
- **Launch environment:** any `GGML_*`/`LLAMA_*` variables the app was launched with. They are cleared
  before every observation.
- **Device:** hardware, OS, GPU family, memory limits.
- **Model:** the model file, and whether it matched its published SHA-256.
- **Per observation:**
  - thermal state before and after;
  - the process footprint and the memory iOS still allows, while the context is alive;
  - any library warnings or errors, including Metal's reason for a failed command buffer.

Arms:

- **upstream:** no flags.
- **bit-exact (rows mode):** the in-place delta-net state only.
- **The recommended stack** for the model's weight type, read from the GGUF.
- **The stack plus batch-invariant mode.**
- **Per type:** the tensor path (PTQ1_0). For Q1_0, PrismML's own popcount option alone and with our
  stack, for context; it is their flag, not one of our changes.
- **"only X":** each flag of the stack on its own, to find which one causes a difference.

### MTP (multi-token prediction)

The app runs MTP with llama.cpp's own speculative-decoding code:

- `common/speculative.cpp` and the files it needs are compiled into the app and linked against the
  framework.
- `BonsaiBench/MTP/BonsaiMTP.cpp` drives it with `examples/speculative-simple`'s loop, at the Mac server
  studies' settings: greedy, 1 draft token, no minimum draft size or probability, and the persistent CPU
  threadpool.
- A gen cell runs both arms through this same loop, so they are timed identically:
  - an arm named "... + MTP" drafts one token per step (MTP arms run only gen cells);
  - other arms decode plainly.
- An MTP run fails instead of silently decoding plainly if the draft context stops producing drafts.
- The result file records, per run:
  - drafted and accepted tokens and the target decodes;
  - the split of the loop into drafting, verification and MTP processing.

  The summary also shows each arm's draft acceptance.
- **Tokens are compared within each arm and between the arms.**
  - Plain and MTP output can legitimately differ: two-token verification is not batch-invariant.
  - The "+ invariant + MTP" arm makes the comparison bit-for-bit.
  - A difference within an arm is always flagged.

**The model.** MTP needs a GGUF grafted with an MTP head, `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` (7.0 GB;
the base model plus a 1.07 GB head, made with the
[sudoingX graft recipe](https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft)).

- It is not published, so it is not in the download list. Copy it in over USB (see below).
- A file whose name contains `-mtp` is loaded with its MTP layers, and it gets the "upstream + MTP",
  "<stack> + MTP" and "<stack> + invariant + MTP" arms.
- Weights are memory-mapped. The check tool's footprint on the M5 was 0.8 GB with this file; the phone's
  footprint is recorded with every run.

**Checked on an M5 Max** (`../tools/check-mtp-bridge.cpp`, the same model, each configuration in a fresh
process):

- **Acceptance:** the bridge accepts 85.5% of drafts (59/69). `llama-speculative-simple` accepts 85.7%
  (60/70), and llama-server 86.8% (59/68).
- **Speed:** our flags + MTP run at 63-65 tok/s, like the example's 64.2-64.9, with each configuration in a
  fresh process. Upstream plain runs at about 45 in the check tool.
- **Tokens:** all four configurations (plain and MTP, upstream and flags) generate exactly the 128 tokens
  llama-server generates for the same prompt, both with and without MTP. The tool's `TOKENS_OUT` writes
  them out for this comparison.

### Differences from the Mac studies

- **Cooldowns and thermal gating.** A phone throttles in steps, so the thermal gate and the paused-in-
  background rule are additions.
- **Retries.** The Mac aborts a cell after 3 failed quartets; the app marks it incomplete and continues.
- **tg128 repetitions.** llama-bench averages 3 repetitions per observation; the app runs 1.

## Build and install

One-time setup:

1. **Sign in to Xcode.** Xcode › Settings › Accounts › + › your Apple Account. A paid team can sign both
   memory entitlements; the first build creates the development certificate.
2. **Enable Developer Mode on the phone.** Settings › Privacy & Security › Developer Mode, then restart.
   Connect by USB the first time and trust the Mac.
3. **Build the framework** from the repository root: `./build-xcframework.sh ios-device ios-sim`. This
   writes `build-apple/llama.xcframework`.
   - The script also records which revision the framework was built from.
   - The app build refuses a framework built from different library source (`ggml/`, `src/`, `include/`)
     than the checkout, because the app compiles llama.cpp's common code against the checkout's headers.
   - So rebuild the framework after pulling library changes. `BB_ALLOW_STALE_FRAMEWORK=1` overrides this
     for experiments.
4. **Create `Config/Local.xcconfig`** (gitignored) with your team:

   ```
   DEVELOPMENT_TEAM = ABCDE12345
   // PRODUCT_BUNDLE_IDENTIFIER = com.example.bonsaibench   (if dev.bonsaibench.app is taken)
   ```

   Do not pick a team in Xcode's Signing & Capabilities pane: that writes it into the committed
   `project.pbxproj`.
5. **Install xcodegen** (`brew install xcodegen`) and run `xcodegen generate` in this directory.

Build and install from the command line (`$DEVICE` from `xcrun devicectl list devices`; keep it out of
committed files):

```
xcodebuild -project BonsaiBench.xcodeproj -scheme BonsaiBench -configuration Release \
  -destination "id=$DEVICE" -allowProvisioningUpdates -allowProvisioningDeviceRegistration build
xcrun devicectl device install app --device "$DEVICE" <DerivedData>/Build/Products/Release-iphoneos/BonsaiBench.app
```

Then start the app from the home screen. Do not use Xcode's Run button for measurements: the debugger, Metal
API validation and frame capture change GPU timings. The shared scheme's Run action is Release with all
three off, but a home-screen launch is the reference.

The app requests `com.apple.developer.kernel.increased-memory-limit` and
`extended-virtual-addressing`. On a 12 GB iPhone 17 Pro Max the limit then reports about 6.3 GB at launch.
Weights are memory-mapped from flash and do not count toward the footprint: a 3.8 GB model ran at a
0.4 GB footprint.

## Getting models onto the phone

- **Download in the app.** "Download from Hugging Face" lists the four Bonsai 27B GGUFs at the PrismML
  revisions the Mac studies used.
  - Downloads use a background session, so they continue while the phone is locked.
  - They are Wi-Fi only unless "Use cellular data" is on.
  - A download is installed only if its size and SHA-256 match.
- **Copy from a Mac over USB.** This took about 400 MB/s here:

  ```
  xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer \
    --domain-identifier dev.bonsaibench.app --source Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --destination Documents/Ternary-Bonsai-2-27B-PTQ1_0.gguf
  ```

- **Finder or the Files app.** Put the file in the app's Documents, or use "Import a .gguf…".

Tap **Verify** on a model that was copied rather than downloaded, to check it against the published hash.
Every model in Documents is excluded from iCloud backup.

Each model row shows an estimate of what it needs (weights + compute buffer + recurrent state + KV cache +
app) against what iOS currently allows. If iOS kills the app while it loads a model, the next launch says
so: that model does not fit.

## Running a study

**Quick tests** (top of the app) are the simplest: each has one Run button that loads the right model and
sets the arms, cells, quartets and cooldown itself. The **Phone suite** runs several of them in a row.

**A custom study** (switch "Custom study" on):

1. Load a model.
2. Pick arm A (usually "upstream") and arm B.
3. Choose cells and tap "Run A-B-B-A".
4. Keep the app in the foreground, the phone on a hard surface (ideally on power), and Low Power Mode off.

Run is disabled while a download or hash check is in progress, because those would skew the timings.

**Unattended, from a Mac.** With the phone unlocked:

```
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables '{"BONSAIBENCH_AUTORUN":"{\"model\":\"Bonsai-27B-Q1_0.gguf\",\"a\":\"upstream\",\"b\":\"only SMALLM_MM\",\"cells\":[\"pp512\"],\"cycles\":1}"}' \
  dev.bonsaibench.app
```

- `--console` shows the library log.
- Arms are preset names as shown in the app.
- Cells may be any `tgN`, `chatN`, `genN` or `ppK`. An arm with MTP runs only gen cells.
- `cycles`, `cooldown`, `waitForNominal` and `ubatch` (512, 256 or 128) are optional.
- `{"suite":true}` runs the phone suite (below), resuming a suite the app died in, and `"from": N`
  starts it at study N.

Fetch the results with:

```
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
  --domain-identifier dev.bonsaibench.app --source Documents/<file>.json --destination .
```

**The phone suite.** "Run the suite" runs the studies still missing from the results table, in order of
value, including the MTP headline (upstream plain vs our flags + MTP) and MTP vs MTP on the MTP model. It:

- loads each model and sets each study's arms and protocol;
- saves one result file per study, and afterwards restores your manual settings;
- waits while a download or hash check is running.

Each study is marked done only if every cell got an accepted quartet for every cycle; otherwise it is
marked incomplete, failed or skipped, with a note.

If iOS stops the app:

- **While loading a model:** that model does not fit. A `bonsaibench-*-did-not-fit.json` record is written,
  and resuming skips every study of that model.
- **During a study:** resuming runs it again once, and a second stop skips it.

**Sending results from the phone.** Use "Share JSON" after a study, or Files › On My iPhone ›
BonsaiBench, and send or save the `bonsaibench-*.json` files (e.g. to iCloud Drive).

## Findings so far (iPhone 17 Pro Max, A19 Pro, iOS 27)

Bonsai 1 binary (Q1_0). tg128 is a complete study: 3 accepted quartets, 40 s cooldown, nominal throughout.
The other cells come from partial studies with one quartet each, so they are early signals. Raw JSON in
[`../results`](../results).

**Our changes** (the Q1_0 stack: in-place delta-net state, small-row routing), against default upstream:

| Cell | Upstream | Our Q1_0 stack | Speedup |
|---|---:|---:|---:|
| **tg128** (3 quartets, all accepted, all nominal) | **10.90 tok/s** | **12.81** | **1.18x** (1.157-1.188) |
| pp2 | 15.8 tok/s | 18.4 | 1.16x |
| pp4 | 17.5 | 18.8 | 1.07x |
| pp8 | 18.2 | 18.8 | 1.03x |
| pp512 | 78.9 (screen only, see below) | 81.7 / 82.3 (a rejected quartet) | not paired: no loss, see below |

**PrismML's popcount option, for context only.** `GGML_METAL_Q1_0_POPCNT` is PrismML's own bit-plane path,
off by default in their code. It is not one of our changes and is not counted in our speedups.

- With our stack it measured 22.8 tok/s at pp2 and 26.1 at pp4, against default upstream. That combination
  mixes their option with ours.
- Two comparisons still need measuring, with the "PrismML popcount (their option)" arm:
  - upstream against popcount alone;
  - popcount alone against our stack plus popcount, which is what our changes add when their option is on.
- Popcount is not bit-exact (M5: mean KLD 0.00040, 99.06% same top token), and greedy output can diverge
  under concurrency. On the phone, single-stream greedy output matched upstream in all 8 chat128 runs.
- At pp512 it has no effect (0.99x, from a quartet rejected for a thermal change): it only covers batches
  of up to 16 columns.

**Bonsai 2 PTQ1_0, our changes against default upstream.** Two studies with one accepted quartet per cell
each. The phone was mostly at fair (two runs reached serious, in rejected quartets), with a 50-60 s
cooldown:

| Cell | Upstream | Our PTQ1 stack | Speedup (study 1, study 2) | Geomean |
|---|---:|---:|---:|---:|
| pp2 | 2.9 tok/s | 11.9 | 4.27x, 4.10x | **4.19x** |
| pp4 | 4.8 | 11.7 | 2.35x, 2.53x | **2.44x** |
| pp8 | 5.0 | 11.7 | 2.30x, 2.41x | **2.35x** |
| chat128 (greedy decode) | 5.67 | 6.25 | 1.10x (study 2; identical tokens in all 4 runs) | |

- The tg128 quartet was rejected: the phone went from fair to serious during a stack run, which then ran
  at 3.3 tok/s.
- PTQ1_0 generation is compute-bound on the phone too: 5.7 tok/s upstream in chat128, against about 9.6
  for Q1_0 in chat128.

Upstream's PTQ1_0 path for 2-8-token batches is very slow on the A19: 2.9 tok/s at pp2, against 15.8
for Q1_0 upstream on the same phone. The multi-column kernels (the CUDA PR #218 port) remove that
bottleneck. These batch shapes are what MTP verification and concurrent requests run.

**Observations:**

- **The Q1_0 kernels are compute-bound on the phone GPU.** Throughput barely grows from 2 to 8 tokens
  per call, while on the M5 Max it grows strongly.
- **Heat biases a compute-bound comparison.** In the chat128 quartets the last upstream run was 16-18%
  slower than the first (nominal to fair), while the other arm held.
  - A drift like that is not linear, so A-B-B-A cannot cancel it, and it favours the arm that needs fewer
    ALU cycles.
  - The spread gate and the thermal-change rule rejected those quartets.
  - The app's thermal gate now requires every run of a quartet to start in the same state, checked after
    the cooldown (see "What it measures").
  - Use a 30-60 s cooldown for the generation cells on a phone.
- **Memory is not a limit for Q1_0.** The footprint stayed at 0.42-0.44 GB: weights are memory-mapped from
  flash and not counted, and iOS allowed 6.0 GB more.
- **PTQ1_0 pp512 is ~0.70x with the full stack on the phone** (70.9-71.5 vs 47.7-51.8 tok/s; per-call
  7.0-7.4 s vs 9.0-10.8 s). The M5 gives the opposite: stack 1.046x, rows mode alone 1.043x, `SMALLM_MM`
  alone 0.999x (llama-bench, 2 interleaved rounds).
  - On the phone, `SMALLM_MM` alone measured 0.975x (2 quartets), so it is not the main cause.
  - The PTQ1 multi-column, GLU and staging paths stop at 8 columns and do not act at 512. That leaves rows
    mode (`GDN_ROWS_PLAIN`) as the suspect; the first quick test checks it.
- **pp512 can fail on the phone: probably the GPU watchdog.** Both `llama_decode failed (-3)` stops (Q1_0
  first study, PTQ1_0 study) happened in pp512. The seeded cell order puts pp512 fourth in the first cycle,
  and both studies failed on their fourth cell.
  - Metal runs about 90% of a graph as one command buffer (`n_cb` = 1), so a 512-token ubatch is 5-9 s of
    GPU work on the phone. That is long enough to plausibly hit iOS's command-buffer timeout, more so when
    the phone is warm and slower.
  - This version records Metal's error text, and every call's time including warmup and a failing call,
    to confirm it. It also has a "prompt
    micro-batch" setting (512 / 256 / 128) to split the batch into shorter submissions.
  - Until then, leave pp512 out of phone studies.
- **pp512 is not slower with our stack.**
  - A first study showed 78.9 (upstream) and 54.0 (stack) tok/s on screen, then a failed command buffer.
    Its file has no pp512 data: that version discarded a quartet that stopped with an error.
  - It did not reproduce: a later study ran the stack at 81.7 and 82.3 tok/s with no error (in a quartet
    rejected for a thermal change).
  - The M5 profile agrees. Only two stack changes touch a 512-token batch: the in-place delta-net state
    (neutral) and `SMALLM_MM` (faster).
  - The failed run was probably a transient, for example iOS using the GPU for its own work while
    charging. Failures now record Metal's error text and each call's time.
