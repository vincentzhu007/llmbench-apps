import SwiftUI

/// Compact monospaced read-out of one reply's throughput, token counts, and
/// time-to-first-token, shown beneath an assistant bubble. Low-contrast on
/// purpose so it never competes with the message itself.
///
/// Only overall throughput is reliably measurable on the `LanguageModelSession`
/// path (CoreAI buffers streamed output), so that is what we surface here.
struct MetricsLabel: View {
    let metrics: ChatMessage.Metrics

    var body: some View {
        HStack(spacing: 12) {
            // Throughput (overall = prefill = decode for these S=1 exports)
            HStack(spacing: 3) {
                Image(systemName: "bolt.fill")
                    .foregroundStyle(.orange)
                    .font(.system(size: 7))
                Text("\(Int(metrics.throughput)) tok/s")
            }
            // Tokens: prompt → output
            HStack(spacing: 3) {
                Image(systemName: "text.alignleft")
                    .foregroundStyle(.blue)
                    .font(.system(size: 7))
                Text("\(formatTokens(metrics.promptTokens)) → \(formatTokens(metrics.outputTokens)) tok")
            }
            // Time to first token
            HStack(spacing: 3) {
                Image(systemName: "timer")
                    .foregroundStyle(.green)
                    .font(.system(size: 7))
                Text(formatTTFT(metrics.ttftMs))
            }
        }
        .font(.system(size: 10, weight: .medium, design: .monospaced))
        .foregroundStyle(.secondary)
    }

    private func formatTTFT(_ ms: Int) -> String {
        if ms >= 1000 {
            return String(format: "%.1fs TTFT", Double(ms) / 1000)
        }
        return "\(ms)ms TTFT"
    }

    private func formatTokens(_ n: Int) -> String {
        if n >= 1000 {
            return String(format: "%.1fk", Double(n) / 1000)
        }
        return "\(n)"
    }
}

/// Live streaming read-out shown inside the in-progress assistant bubble while
/// tokens arrive: `generating… · X tok/s` (or `warming up…` before the first
/// token).
struct LiveMetricsLabel: View {
    let phase: Phase
    let liveTokPerSec: Double?

    enum Phase { case warmingUp, generating }

    var body: some View {
        HStack(spacing: 4) {
            switch phase {
            case .warmingUp:
                Image(systemName: "circle.dotted").foregroundStyle(.secondary).font(.system(size: 8))
                Text("warming up…")
            case .generating:
                Image(systemName: "bolt.fill").foregroundStyle(.orange).font(.system(size: 7))
                if let rate = liveTokPerSec {
                    Text("generating · \(Int(rate)) tok/s")
                } else {
                    Text("generating…")
                }
            }
        }
        .font(.system(size: 10, weight: .medium, design: .monospaced))
        .foregroundStyle(.secondary)
    }
}
