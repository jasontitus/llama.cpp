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

/// Rough whole-process estimate for loading a Bonsai 27B GGUF: weights + ~0.5 GB Metal compute buffer +
/// ~0.15 GB recurrent state + KV cache at 2K context. Shown before loading so an attempt that will not
/// fit can be skipped; the real limit is measured, not assumed.
func estimatedNeedBytes(modelFileBytes: UInt64) -> UInt64 {
    return modelFileBytes + 520_000_000 + 160_000_000 + 140_000_000
}
