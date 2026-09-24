# BonsaiBench (iOS)

Paired A-B-B-A benchmarking of the Bonsai Metal research flags on iPhone, with the protocol of the Mac
studies in [`../../m5`](../../m5/README.md) adapted to a phone (see "Differences from the Mac studies").
Research tool; not an App Store app.

## What it measures

| Cell | What | Compare with |
|---|---|---|
| **tg128** | 128 single-token decodes from an empty context, random tokens, no sampling | llama-bench tg128 (the Mac studies' tg128) |
| **chat128** | greedy generation of 128 tokens after a chat prompt, including sampling; the generated tokens are compared between arms | llama-server generation speed; bit-exactness |
| **pp2 / pp4 / pp8** | k-token batches: the step shapes of MTP verification and concurrent requests | llama-bench ppK |
| **pp512** | a 512-token prompt batch | llama-bench pp512 |

Every observation creates a fresh llama context, so the research flags (environment variables, re-read
when a context is created) take effect per arm. The context keeps one output row (`n_outputs_max = 1`,
which spares ~0.5 GB of logits buffer) and n_ctx 1024.

For each cell, per cycle:

1. An A-B-B-A quartet.
2. Before every observation:
   - wait (up to 5 min) while the phone is `serious`/`critical`;
   - a cooldown (default 8 s);
   - pause while the app is not in the foreground.
3. The quartet is rejected if any of these hold:
   - the two A runs, or the two B runs, differ by more than the spread gate (default 1.20x);
   - the thermal state changed during the quartet;
   - the app left the foreground;
   - an observation failed (a Metal command buffer error, a context that could not be created).
4. A rejected quartet is kept and repeated, at most 3 times; the fastest is never chosen.
5. A cell without an accepted quartet for every cycle is marked incomplete.

The speedup is the geometric mean of the accepted quartets' mean(B)/mean(A).

ppK cells warm up for at least 1 s and measure for at least 2.5 s, since one small decode is ~0.1 s on a
phone. Each call's time is recorded.

A result JSON is written to Documents at the start and after every quartet, so a study that iOS kills
keeps what it measured. It records:

- **Build:** the git revision the app was built from, and the SHA-256 of the embedded llama.framework.
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

MTP is not included. The framework exposes the MTP context type, but speculative decoding (the draft and
verify loop) lives in llama.cpp's common library, which the iOS framework does not build, and the grafted
MTP GGUFs are not in the download list.

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
- Cells may be any `tgN`, `chatN` or `ppK`.
- `cycles` and `cooldown` are optional.

Fetch the results with:

```
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
  --domain-identifier dev.bonsaibench.app --source Documents/<file>.json --destination .
```

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
| pp512 | 78.9 | 81.7 / 82.3 | not paired (different studies): no loss, see below |

**PrismML's popcount option, for context only.** `GGML_METAL_Q1_0_POPCNT` is PrismML's own bit-plane path,
off by default in their code. It is not one of our changes and is not counted in our speedups.

- With our stack it measured 22.8 tok/s at pp2 and 26.1 at pp4, against default upstream. That combination
  mixes their option with ours.
- Two comparisons still need measuring, with the "PrismML popcount (their option)" arm:
  - upstream against popcount alone;
  - popcount alone against our stack plus popcount, which is what our changes add when their option is on.
- Popcount is not bit-exact (M5: mean KLD 0.00037, 99.07% same top token), and greedy output can diverge
  under concurrency. On the phone, single-stream greedy output matched upstream in all 8 chat128 runs.
- At pp512 it has no effect (0.99x): it only covers batches of up to 16 columns.

**Observations:**

- **The Q1_0 kernels are compute-bound on the phone GPU.** Throughput barely grows from 2 to 8 tokens
  per call, while on the M5 Max it grows strongly.
- **Heat biases a compute-bound comparison.** In the chat128 quartets the last upstream run was 16-18%
  slower than the first (nominal to fair), while the other arm held.
  - A drift like that is not linear, so A-B-B-A cannot cancel it, and it favours the arm that needs fewer
    ALU cycles.
  - The spread gate and the thermal-change rule rejected those quartets.
  - The app now waits for a nominal thermal state before every observation (toggle in Protocol).
  - Use a 30-60 s cooldown for the generation cells on a phone.
- **Memory is not a limit for Q1_0.** The footprint stayed at 0.42-0.44 GB: weights are memory-mapped from
  flash and not counted, and iOS allowed 6.0 GB more.
- **pp512 is not slower with our stack.**
  - A first study measured 54.0 tok/s and then a failed command buffer. That did not reproduce: a later
    study ran the stack at 81.7 and 82.3 tok/s, with no error.
  - The M5 profile agrees. Only two stack changes touch a 512-token batch: the in-place delta-net state
    (neutral) and `SMALLM_MM` (faster).
  - The failed run was probably a transient, for example iOS using the GPU for its own work while
    charging. Failures now record Metal's error text and each call's time.
