// LoopGuard — runtime degenerate-loop detector for token generation.
//
// Parity with the Odysseus Python engine's anti-loop (scripts/runner.py:
// _detect_loop / _detect_loop_large). Thinking models (e.g. Qwen3.6-35B-A3B)
// can enter a runaway where a large block repeats verbatim — observed on the
// TMB agentic bench as "bloc de 1878c repete 5x consecutifs" burning tens of
// thousands of tokens. A prompt-level reasoning-guard does not stop it once it
// starts; only a hard, token-id-level cut does.
//
// The guard ingests each emitted token id and reports true when the tail is a
// block of period P repeated R times consecutively:
//   - short period (P in 1...64):  R >= 12  (tight loops)
//   - large period (P in 65...1024): R >= 4  (thinking / paragraph runaways)
// A 65..1024-token block repeated 4x verbatim is never legitimate content, so
// the large-period rule is the safe, high-value case. Anchor trick: only test
// periods P where ids[n-1-P] == ids[n-1], which cuts the candidate set sharply.
//
// Enabled by default; set TELEMAK_ANTILOOP=0 to disable (bench / debugging).

import Foundation

public struct LoopGuard: Sendable {
    // Read once — this sits in the per-token hot path.
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment["TELEMAK_ANTILOOP"] != "0"

    private static let maxPeriod = 1024
    private static let shortPeriodMax = 64
    private static let shortRepeats = 12
    private static let largeRepeats = 4
    private static let checkEvery = 16
    // Keep enough tail to verify the largest loop: maxPeriod * (largeRepeats+1).
    private static let maxKeep = 1024 * 5

    private var ids: [Int] = []
    private var sinceCheck = 0

    public init() {}

    /// Feed the next emitted token id. Returns true when a runaway loop is
    /// detected and generation should stop.
    public mutating func ingest(_ id: Int) -> Bool {
        if !Self.enabled { return false }
        ids.append(id)
        if ids.count > Self.maxKeep {
            ids.removeFirst(ids.count - Self.maxKeep)
        }
        sinceCheck += 1
        if sinceCheck < Self.checkEvery { return false }
        sinceCheck = 0
        return detect()
    }

    private func detect() -> Bool {
        let n = ids.count
        if n < 4 { return false }
        let last = ids[n - 1]
        let maxP = min(Self.maxPeriod, n / 2)
        var p = 1
        while p <= maxP {
            // Anchor: a period-p tail loop requires the token p back to match.
            if ids[n - 1 - p] == last {
                let need = p <= Self.shortPeriodMax ? Self.shortRepeats : Self.largeRepeats
                if hasConsecutiveRepeat(period: p, repeats: need) {
                    return true
                }
            }
            p += 1
        }
        return false
    }

    /// True when the last `repeats` blocks of length `period` are identical.
    private func hasConsecutiveRepeat(period p: Int, repeats r: Int) -> Bool {
        let n = ids.count
        if n < p * r { return false }
        let base = n - p  // start of the final block
        var k = 1
        while k < r {
            let off = base - k * p
            var i = 0
            while i < p {
                if ids[base + i] != ids[off + i] { return false }
                i += 1
            }
            k += 1
        }
        return true
    }
}
