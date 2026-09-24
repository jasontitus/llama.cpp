import Foundation
import Metal
import os

/// Device facts recorded with every result: hardware identifier, OS, GPU family and memory limits.
struct DeviceInfo: Codable {
    var machine: String            // e.g. "iPhone18,2"
    var systemVersion: String
    var gpuName: String
    var highestAppleFamily: Int
    var physicalMemoryBytes: UInt64
    var appAvailableMemoryBytes: UInt64   // os_proc_available_memory at capture time
    var thermalState: String
    var lowPowerMode: Bool

    static func capture() -> DeviceInfo {
        var u = utsname()
        uname(&u)
        let machine = withUnsafeBytes(of: &u.machine) { raw in
            String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
        }
        let dev = MTLCreateSystemDefaultDevice()
        var family = 0
        if let dev {
            for f in 1...12 {
                if let fam = MTLGPUFamily(rawValue: MTLGPUFamily.apple1.rawValue + f - 1), dev.supportsFamily(fam) { family = f }
            }
        }
        return DeviceInfo(machine: machine,
                          systemVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                          gpuName: dev?.name ?? "none",
                          highestAppleFamily: family,
                          physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                          appAvailableMemoryBytes: UInt64(availableMemory()),
                          thermalState: thermalStateName(),
                          lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}

func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal:  return "nominal"
    case .fair:     return "fair"
    case .serious:  return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// Memory the app may still allocate before iOS terminates it (0 on the simulator / macOS).
func availableMemory() -> Int {
    #if os(iOS) && !targetEnvironment(simulator)
    return Int(os_proc_available_memory())
    #else
    return 0
    #endif
}

/// Run one empty command buffer. Metal can hold freed residency-set memory until the next GPU work; this
/// makes sure a freed model's memory is released before the next one loads.
func flushGPU() {
    guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(), let cb = q.makeCommandBuffer() else { return }
    cb.commit()
    cb.waitUntilCompleted()
}

/// Physical footprint of this process (what jetsam compares against the limit).
func physicalFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.phys_footprint : 0
}

/// Rough app memory for running a Bonsai 27B GGUF: the weights are memory-mapped from flash and do not
/// count (a 5.9 GB model ran at a 0.43 GB footprint on an iPhone 17 Pro Max), so this is the Metal compute
/// buffer + recurrent state + KV cache at n_ctx 1024 + the app and its pipelines, plus the MTP draft context
/// for an -mtp file. Shown before loading; the real footprint is recorded with every observation.
func estimatedNeedBytes(modelFileBytes: UInt64, mtp: Bool = false) -> UInt64 {
    return 250_000_000 + 160_000_000 + 70_000_000 + 150_000_000 + (mtp ? 200_000_000 : 0)
}
