# M5, A19 and A20 tuning plan

## Verified starting information

Apple's feature table dated May 21, 2026 lists M1-series as Apple7 and M5-series/A19-series as Apple10. The version inspected does not list A20; its family, limits and performance remain unverified here. Query the actual device and current SDK before selecting a path. Shared family support does not establish equal occupancy, clocks, bandwidth or thermal behavior. [Apple Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

Apple recommends runtime feature detection with MTLDevice and supportsFamily. Use capability and OS checks, retaining the existing fallback, rather than product-name dispatch. [Detecting GPU features](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions)

Apple documents per-core neural accelerators and tensor operations for M5/A19 GPUs, with tiled operations and profiling as part of optimization. This is a later research avenue; it does not mean the packed PTQ1_0 representation or our scalar kernel automatically uses that hardware. [M5/A19 machine-learning talk](https://developer.apple.com/videos/play/tech-talks/111432/)

Metal's newer quantized tensor facilities deserve inspection for a future tiled implementation. Measure explicit trit unpacking, supported operand formats, scale handling and staging costs. No native PTQ1_0 tensor support or benefit at n=1..4 is assumed. [Custom operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)

## Metadata to retain on every actual device

The bundled device-info.mm queries the Metal device, supported Apple families known to its build SDK, maximum queried family, buffer/threadgroup limits, physical memory and thermal state; macOS also reports recommended working set and unified memory. Setup compiles it, and each measured run saves fresh output. If a device supports a newer family than the probe knows, update the probe/SDK and preserve that change in a new experiment; highest-family-queried is not a claim about the newest hardware present.

For each candidate pipeline additionally record threadExecutionWidth, maxTotalThreadsPerThreadgroup, static/dynamic threadgroup-memory use, the actual grid/group sizes, compiler options, and available occupancy/register/spill/stall/cache counters through Xcode GPU profiling tools. The generic device limits alone do not provide those pipeline-specific values. Preserve captures separately from timing runs because instrumentation can change performance.

Record exact hardware SKU/GPU core count, OS and Xcode/Metal compiler, RAM, power source/mode, thermal state and available frequency counters, and memory pressure. Collect cold load, warmed inference and sustained inference separately. Do not equate a laptop peak with phone sustained performance.

## Sequence

1. M5 Max: reproduce the pinned M1 candidate and controls. Establish local correctness, stable ABBA timings and native prefill/decode/MTP measurements.
2. M5 Max: use the editable development worktree and named snapshots. Screen portable geometries first; change one hypothesis at a time. Only then consider Apple10-specific implementation or dispatch.
3. A19: replay the portable survivors on an actual device using a signed iOS app/test target. Reuse fixtures, numerical logic and the same packed models where memory permits; record the real Metal family/limits.
4. A20: treat capabilities as unknown until actual Apple documentation/SDK and runtime queries establish them. Begin with the fallback plus portable candidates, then repeat the device study. Do not infer an Apple family number or supported tensor format from the name.
5. Return each candidate to M1 as a regression check before a cross-device default/PR. Preserve per-device alternatives if one geometry does not win everywhere.

## Phone harness work still required

The delivered launcher is macOS tooling. screen, sudo powermetrics, the process-memory probe and desktop process orchestration are not an iPhone runner. A signed Xcode harness must be built separately; no such app or phone measurement is claimed in this package.

That harness should embed the backend/fixtures, expose an A/B selector, use the same seeded tensors and prompts, and run whole ABBA quartets sequentially in the foreground. Record completed GPU work rather than just command submission. Export source/model/build identifiers, per-observation durations, rejection reasons, thermal state, output tokens/logits, memory high-water marks and app termination evidence. device-info.mm exposes bonsaiMetalDeviceInfo() for reuse without its CLI main.

Start with bounded projection tests. Attempt a full 27B model only after measuring the app's actual memory allowance and usable headroom. MTP adds a head plus runtime workspaces and cache state; macOS resident-memory results do not establish iOS admission or sustained operation. Test short-context plain and MTP at one request first, then contexts and concurrency that fit. An out-of-memory/jetsam event is a failed configuration, not zero throughput. Test charging and battery conditions separately; record Low Power Mode and thermal throttling.

The existing family study does not directly pair plain/MTP timing; add a distinct paired mode study for final claims about MTP benefit on each device. Keep kernel effects, speculative acceptance, memory and power separate in the report.
