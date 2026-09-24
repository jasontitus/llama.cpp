import CryptoKit
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// A Bonsai GGUF published by PrismML on Hugging Face, pinned to the revision the Mac studies used.
struct CatalogModel: Identifiable, Hashable {
    let file: String
    let title: String
    let repo: String
    let revision: String
    let bytes: UInt64
    let sha256: String
    var id: String { file }
    var url: URL { URL(string: "https://huggingface.co/prism-ml/\(repo)/resolve/\(revision)/\(file)")! }
}

enum Catalog {
    static let models: [CatalogModel] = [
        CatalogModel(file: "Bonsai-27B-Q1_0.gguf", title: "Bonsai 1 binary (Q1_0)",
                     repo: "Bonsai-27B-gguf", revision: "f10afb355f104535e3e3e98cf7ab7795c72bd292",
                     bytes: 3_803_452_480, sha256: "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"),
        CatalogModel(file: "Ternary-Bonsai-2-27B-PTQ1_0.gguf", title: "Bonsai 2 ternary (PTQ1_0)",
                     repo: "Ternary-Bonsai-2-27B-gguf", revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
                     bytes: 5_946_648_928, sha256: "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"),
        CatalogModel(file: "Ternary-Bonsai-27B-PQ2_0.gguf", title: "Bonsai 1 ternary (PQ2_0)",
                     repo: "Ternary-Bonsai-27B-gguf", revision: "86e89f34c93201c3dfd5e5880fedb0022fc7e34d",
                     bytes: 7_165_121_600, sha256: "e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6"),
        CatalogModel(file: "Ternary-Bonsai-2-27B-PQ2_0.gguf", title: "Bonsai 2 ternary (PQ2_0)",
                     repo: "Ternary-Bonsai-2-27B-gguf", revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
                     bytes: 7_206_168_928, sha256: "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"),
    ]
}

/// Downloads catalog models into Documents with a background URLSession (continues while the phone is
/// locked or the app is suspended), then checks size and SHA-256 before the file appears as a .gguf.
/// All state below is touched on the main queue only; delegate callbacks hop there.
final class Downloader: NSObject, ObservableObject, URLSessionDownloadDelegate {
    static let shared = Downloader()
    static let sessionID = "dev.bonsaibench.downloads"

    enum Phase: Equatable {
        case downloading(Double)       // fraction
        case verifying
        case failed(String)
    }

    @Published private(set) var phase: [String: Phase] = [:]   // file name -> phase; absent = idle
    @Published private(set) var freeSpace: UInt64 = 0
    @Published private(set) var message = ""                    // last completed download or check
    @Published var allowCellular = UserDefaults.standard.bool(forKey: "allowCellular") {
        didSet { UserDefaults.standard.set(allowCellular, forKey: "allowCellular") }
    }
    var onFinished: (() -> Void)?                               // refresh the model list
    var backgroundCompletion: (() -> Void)?                     // from the app delegate

    private var active: [String: Int] = [:]                     // file -> identifier of its live task
    private var cancelledTasks: Set<Int> = []
    private var resumeData: [String: (data: Data, at: Date)] = [:]
    private var pendingVerifications = 0
    private var eventsFinished = false
    private var scannedOrphans = false
    private lazy var session: URLSession = {
        let c = URLSessionConfiguration.background(withIdentifier: Self.sessionID)
        c.isDiscretionary = false
        c.sessionSendsLaunchEvents = true
        return URLSession(configuration: c, delegate: self, delegateQueue: nil)
    }()

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    var busy: Bool { phase.values.contains { if case .failed = $0 { return false }; return true } }

    /// Reattach to transfers that kept running while the app was not, and finish checking any download
    /// whose verification was interrupted. With `autostart`, then start the models named in
    /// BONSAIBENCH_DOWNLOAD (comma-separated file names, or "all") that are not present yet, e.g.
    /// `devicectl device process launch --environment-variables '{"BONSAIBENCH_DOWNLOAD":"all"}' …`.
    func restore(autostart: Bool = false) {
        session.getAllTasks { tasks in
            DispatchQueue.main.async {
                for t in tasks where t.state == .running || t.state == .suspended {
                    guard let file = t.taskDescription, self.phase[file] == nil else { continue }
                    let total = t.countOfBytesExpectedToReceive
                    self.active[file] = t.taskIdentifier
                    self.phase[file] = .downloading(total > 0 ? Double(t.countOfBytesReceived) / Double(total) : 0)
                }
                if !self.scannedOrphans {
                    self.scannedOrphans = true
                    self.recoverOrphans()
                }
                self.updateFreeSpace()
                guard autostart, let want = ProcessInfo.processInfo.environment["BONSAIBENCH_DOWNLOAD"] else { return }
                let names = Set(want.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
                for m in Catalog.models where (want == "all" || names.contains(m.file)) && self.phase[m.file] == nil
                    && !FileManager.default.fileExists(atPath: Self.documents.appendingPathComponent(m.file).path) {
                    self.start(m)
                }
            }
        }
    }

    /// `<catalog file>.<task id>.part` files are downloads whose verification never finished (the app was
    /// killed); check them now instead of leaving multi-GB files behind.
    private func recoverOrphans() {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: Self.documents.path)) ?? []
        for name in files where name.hasSuffix(".part") {
            let url = Self.documents.appendingPathComponent(name)
            guard let m = Catalog.models.first(where: { name.hasPrefix($0.file + ".") }),
                  name.dropFirst(m.file.count + 1).dropLast(5).allSatisfy(\.isNumber) else {
                try? FileManager.default.removeItem(at: url)   // an interrupted import
                continue
            }
            if phase[m.file] == nil { verify(m, part: url) } else { try? FileManager.default.removeItem(at: url) }
        }
    }

    func updateFreeSpace() {
        let v = try? Self.documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        freeSpace = UInt64(max(0, v?.volumeAvailableCapacityForImportantUsage ?? 0))
    }

    func start(_ m: CatalogModel) {
        // The finished file is moved (not copied) out of the download location, so each download needs its
        // own size plus headroom; transfers already running still need their remaining bytes.
        updateFreeSpace()
        let reserved = phase.reduce(UInt64(0)) { sum, e in
            guard case .downloading(let f) = e.value, let o = Catalog.models.first(where: { $0.file == e.key }) else { return sum }
            return sum + UInt64(Double(o.bytes) * max(0, 1 - f))
        }
        let need = m.bytes + 1_000_000_000 + reserved
        if freeSpace < need {
            phase[m.file] = .failed("needs \(gb(need)) free, have \(gb(freeSpace))")
            return
        }
        let task: URLSessionDownloadTask
        // Resume data holds the signed CDN URL, which expires after about an hour.
        if let r = resumeData.removeValue(forKey: m.file), Date().timeIntervalSince(r.at) < 50 * 60 {
            task = session.downloadTask(withResumeData: r.data)
        } else {
            var req = URLRequest(url: m.url)
            req.allowsCellularAccess = allowCellular
            req.allowsExpensiveNetworkAccess = allowCellular
            task = session.downloadTask(with: req)
        }
        task.taskDescription = m.file
        task.countOfBytesClientExpectsToReceive = Int64(m.bytes)
        active[m.file] = task.taskIdentifier
        phase[m.file] = .downloading(0)
        message = ""
        task.resume()
    }

    func cancel(_ file: String) {
        if let id = active.removeValue(forKey: file) { cancelledTasks.insert(id) }
        resumeData[file] = nil
        phase[file] = nil
        session.getAllTasks { tasks in
            let mine = tasks.filter { $0.taskDescription == file }
            DispatchQueue.main.async { mine.forEach { self.cancelledTasks.insert($0.taskIdentifier) } }
            mine.forEach { $0.cancel() }
        }
    }

    /// Callbacks from a cancelled or superseded task must not touch the row. After a relaunch the live
    /// task is not known until its first callback, which then claims the row.
    private func owns(_ file: String, _ id: Int) -> Bool {
        if cancelledTasks.contains(id) { return false }
        if let a = active[file] { return a == id }
        active[file] = id
        return true
    }

    // MARK: URLSessionDownloadDelegate (called on the session's background queue)

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let file = downloadTask.taskDescription,
              let m = Catalog.models.first(where: { $0.file == file }) else { return }
        let id = downloadTask.taskIdentifier
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : Int64(m.bytes)
        let f = Double(totalBytesWritten) / Double(total)
        DispatchQueue.main.async {
            guard self.owns(file, id) else { return }
            // Publishing every callback would redraw the list thousands of times; 0.1% steps are enough.
            if case .downloading(let old) = self.phase[file], f - old < 0.001, f < 1 { return }
            if case .verifying = self.phase[file] { return }
            self.phase[file] = .downloading(f)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The temporary file is deleted when this returns, so move it now (to a name unique to this task);
        // decide on the main queue whether it is still wanted, then verify.
        guard let file = downloadTask.taskDescription,
              let m = Catalog.models.first(where: { $0.file == file }) else { return }
        let id = downloadTask.taskIdentifier
        // A resumed download reports 206; size and SHA-256 decide whether the bytes are right.
        if let http = downloadTask.response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            DispatchQueue.main.async {
                guard self.owns(file, id) else { return }
                self.active[file] = nil
                self.phase[file] = .failed("HTTP \(http.statusCode)")
            }
            return
        }
        let part = Self.documents.appendingPathComponent("\(file).\(id).part")
        do {
            try FileManager.default.moveItem(at: location, to: part)
            var rv = URLResourceValues()
            rv.isExcludedFromBackup = true
            var p = part
            try? p.setResourceValues(rv)
        } catch {
            DispatchQueue.main.async {
                guard self.owns(file, id) else { return }
                self.active[file] = nil
                self.phase[file] = .failed("could not save: \(error.localizedDescription)")
            }
            return
        }
        DispatchQueue.main.async {
            guard self.owns(file, id) else { try? FileManager.default.removeItem(at: part); return }
            self.active[file] = nil
            self.verify(m, part: part)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, let file = task.taskDescription else { return }
        let id = task.taskIdentifier
        let ns = error as NSError
        DispatchQueue.main.async {
            guard self.owns(file, id) else { return }
            self.active[file] = nil
            if let data = ns.userInfo[NSURLSessionDownloadTaskResumeData] as? Data { self.resumeData[file] = (data, Date()) }
            self.phase[file] = .failed(error.localizedDescription)
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            self.eventsFinished = true
            self.finishBackgroundEventsIfIdle()
        }
    }

    /// iOS may suspend the app as soon as the completion handler runs, so hold it until checks finish.
    private func finishBackgroundEventsIfIdle() {
        guard eventsFinished, pendingVerifications == 0 else { return }
        eventsFinished = false
        backgroundCompletion?()
        backgroundCompletion = nil
    }

    // MARK: verification

    /// Check a finished download and install it as Documents/<file>.
    private func verify(_ m: CatalogModel, part: URL) {
        phase[m.file] = .verifying
        pendingVerifications += 1
        let time = BackgroundTime("verify \(m.file)")
        DispatchQueue.global(qos: .utility).async {
            let outcome = Self.check(m, at: part).map { digest -> Result<String, Error> in
                let dest = Self.documents.appendingPathComponent(m.file)
                // rename(2) replaces an existing file atomically: no moment without a model.
                guard rename(part.path, dest.path) == 0 else { return .failure(POSIXError(.init(rawValue: errno) ?? .EIO)) }
                var rv = URLResourceValues()
                rv.isExcludedFromBackup = true
                var d = dest
                try? d.setResourceValues(rv)
                VerifiedMark.set(dest, sha256: digest)
                return .success(digest)
            }
            DispatchQueue.main.async {
                switch outcome {
                case .success(.success):
                    self.phase[m.file] = nil
                    self.message = "Downloaded \(m.file); size and SHA-256 match."
                case .success(.failure(let e)):
                    try? FileManager.default.removeItem(at: part)
                    self.phase[m.file] = .failed("could not install: \(e.localizedDescription)")
                case .failure(let e):
                    try? FileManager.default.removeItem(at: part)
                    self.phase[m.file] = .failed(e.message)
                }
                self.pendingVerifications -= 1
                time.end()
                self.updateFreeSpace()
                self.onFinished?()
                self.finishBackgroundEventsIfIdle()
            }
        }
    }

    /// Check a model already in Documents (e.g. copied in with Finder or devicectl) against the catalog.
    /// A mismatch is reported, not deleted: the file is the user's.
    func verifyInstalled(_ m: CatalogModel) {
        let url = Self.documents.appendingPathComponent(m.file)
        phase[m.file] = .verifying
        pendingVerifications += 1
        let time = BackgroundTime("verify \(m.file)")
        DispatchQueue.global(qos: .utility).async {
            let outcome = Self.check(m, at: url)
            if case .success(let digest) = outcome { VerifiedMark.set(url, sha256: digest) }
            DispatchQueue.main.async {
                switch outcome {
                case .success:
                    self.phase[m.file] = nil
                    self.message = "\(m.file): size and SHA-256 match."
                case .failure(let e):
                    self.phase[m.file] = .failed(e.message + "; download it again")
                }
                self.pendingVerifications -= 1
                time.end()
                self.onFinished?()
                self.finishBackgroundEventsIfIdle()
            }
        }
    }

    struct CheckError: Error { let message: String }

    /// Size, then SHA-256; returns the digest.
    static func check(_ m: CatalogModel, at url: URL) -> Result<String, CheckError> {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { UInt64($0) } ?? 0
        guard size == m.bytes else { return .failure(CheckError(message: "size \(size) bytes, expected \(m.bytes)")) }
        guard let digest = sha256(of: url) else { return .failure(CheckError(message: "could not read the file")) }
        guard digest == m.sha256 else { return .failure(CheckError(message: "SHA-256 mismatch; the file is corrupt")) }
        return .success(digest)
    }

    static func sha256(of url: URL) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        var hasher = SHA256()
        // read(upToCount:) returns nil (or empty data) at end of file and throws on a read error.
        do {
            while let chunk = try autoreleasepool(invoking: { try h.read(upToCount: 16 << 20) }), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
        } catch {
            return nil
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Records on the file itself that it matched the catalog hash, with the size and modification time it had
/// then, so a later replacement of the file (Finder, devicectl) reads as unverified.
enum VerifiedMark {
    static let name = "dev.bonsaibench.sha256"

    static func set(_ url: URL, sha256: String) {
        guard let stamp = stamp(url) else { return }
        let value = "\(sha256) \(stamp)"
        _ = value.withCString { setxattr(url.path, name, $0, strlen($0), 0, 0) }
    }

    /// The verified SHA-256, if the file still has the size and modification time it was verified with.
    static func get(_ url: URL) -> String? {
        var buf = [CChar](repeating: 0, count: 256)
        let n = getxattr(url.path, name, &buf, buf.count - 1, 0, 0)
        guard n > 0, let stamp = stamp(url) else { return nil }
        let parts = String(cString: buf).split(separator: " ", maxSplits: 1)
        guard parts.count == 2, parts[1] == stamp else { return nil }
        return String(parts[0])
    }

    private static func stamp(_ url: URL) -> String? {
        guard let v = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]),
              let size = v.fileSize, let date = v.contentModificationDate else { return nil }
        return "\(size) \(date.timeIntervalSince1970)"
    }
}

/// Asks iOS for time to finish a multi-GB hash if the app goes to the background. Main queue only.
final class BackgroundTime {
    #if canImport(UIKit)
    private var id: UIBackgroundTaskIdentifier = .invalid
    #endif

    init(_ name: String) {
        #if canImport(UIKit)
        id = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in self?.end() }
        #endif
    }

    func end() {
        #if canImport(UIKit)
        if id != .invalid { UIApplication.shared.endBackgroundTask(id); id = .invalid }
        #endif
    }
}

#if canImport(UIKit)
/// Hands the background-session completion handler to the downloader when iOS relaunches the app to
/// deliver download events.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == Downloader.sessionID else { return completionHandler() }
        Downloader.shared.backgroundCompletion = completionHandler
        Downloader.shared.restore()
    }
}
#endif
