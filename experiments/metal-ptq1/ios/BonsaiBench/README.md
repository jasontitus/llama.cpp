# BonsaiBench (iOS)

Paired A-B-B-A benchmarking of the Bonsai Metal research flags on iPhone, using the same protocol as the
Mac studies in [`../../m5`](../../m5/README.md). Research tool; not an App Store app.

## What it measures

- **tg128**: greedy generation of 128 tokens after a short chat prompt (plain decoding speed), with the
  generated token IDs compared between arms.
- **pp2 / pp4 / pp8**: k-token batches, the step shapes of MTP verification and concurrent requests.
- **pp512**: prompt processing.

For each cell: A-B-B-A quartets (default 3), a cooldown before every observation (default 8 s), a fresh
llama context per observation (the research flags are re-read when a context is created), and the
spread gate (the two A runs, and the two B runs, within 1.20x of each other; failed quartets are kept and
repeated, at most 3 times). Every observation records thermal state, process footprint and the memory
the app may still allocate. Results are saved as JSON in the app's Documents and can be shared.

Arms: upstream (no flags), bit-exact (in-place delta-net state only), the recommended stack for the
model's weight type (read from the GGUF), the stack plus batch-invariant mode, and per type the tensor
path (PTQ1) or PrismML's popcount path (Q1_0).

MTP is not included: llama.cpp's speculative decoding lives in its server/common library, which the iOS
framework does not build.

## Build and install

1. Build the framework from the repository root: `./build-xcframework.sh ios-device ios-sim`
   (writes `build-apple/llama.xcframework`).
2. Generate the project: `cd experiments/metal-ptq1/ios/BonsaiBench && xcodegen generate`
   (or open the committed `BonsaiBench.xcodeproj`).
3. Open it in Xcode, select the BonsaiBench target, Signing & Capabilities, choose your team, and run on
   the phone.

The app requests `com.apple.developer.kernel.increased-memory-limit` and
`extended-virtual-addressing`, which the 4-8 GB Bonsai models need. If your account cannot sign with
them, remove the `entitlements` block from `project.yml`, regenerate, and expect only the smallest model
to load.

## Loading models

Connect the phone to a Mac, open it in Finder, Files tab, BonsaiBench, and drag a `.gguf` in (or use
"Import a .gguf…" in the app, which copies the file). Pull to refresh the list. Each model shows its file
size, an estimated total need (weights + ~0.5 GB compute buffer + recurrent state + KV cache), and
whether that is likely to fit in what iOS currently allows the app.

Expected on a 12 GB iPhone 17 Pro / Pro Max (estimates; the app reports the real limit):

| Model | File | Likely |
|---|---:|---|
| Bonsai 1 binary (Q1_0) | 3.8 GB | fits |
| Bonsai 2 PTQ1_0 | 5.9 GB | fits with the increased memory limit |
| Bonsai 2 PQ2_0 / Bonsai 1 ternary (PQ2_0) | 7.2 GB | borderline |

An out-of-memory termination is a failed configuration, not a result; record it.

## Running a study

Load a model, pick arm A (usually "upstream") and arm B, choose cells, and tap "Run A-B-B-A". Keep the app
in the foreground, the phone on power and on a hard surface, Low Power Mode off. Phones throttle much
sooner than laptops, so expect rejected quartets and wider ranges; the JSON keeps every observation.
