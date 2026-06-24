import SwiftUI

struct ChatBubble: View {
    let message: ChatMessage
    var isStreaming: Bool = false
    var livePhase: LiveMetricsLabel.Phase? = nil
    var liveTokPerSec: Double? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .assistant {
                avatar
            } else {
                Spacer()
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : "Assistant")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)

                Text(message.content)
                    .font(.body)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .foregroundStyle(message.role == .user ? Color.white : Color.primary)
                    .background(
                        (message.role == .user ? Color.blue : Color.secondary.opacity(0.15)),
                        in: RoundedRectangle(cornerRadius: 18)
                    )
                    .overlay(alignment: .trailing) {
                        if isStreaming {
                            Rectangle()
                                .fill(.primary)
                                .frame(width: 2, height: 16)
                                .opacity(0.5)
                                .padding(.trailing, 6)
                        }
                    }

                // Metrics: final read-out once done, live read-out while streaming.
                if message.role == .assistant {
                    if let metrics = message.metrics {
                        MetricsLabel(metrics: metrics)
                            .padding(.horizontal, 6)
                    } else if let livePhase {
                        LiveMetricsLabel(phase: livePhase, liveTokPerSec: liveTokPerSec)
                            .padding(.horizontal, 6)
                    }
                }
            }

            if message.role == .user {
                avatar
            } else {
                Spacer()
            }
        }
    }

    private var avatar: some View {
        Group {
            if message.role == .assistant {
                Image(systemName: "brain.head.profile")
                    .font(.title3)
                    .foregroundStyle(.purple)
            } else {
                Image(systemName: "person.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.blue)
            }
        }
        .frame(width: 32, height: 32)
        .background((message.role == .assistant ? Color.purple : Color.blue).opacity(0.15))
        .clipShape(Circle())
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var metrics: Metrics?        // only assistant replies carry metrics
    let timestamp = Date()

    init(role: Role, content: String, metrics: Metrics? = nil) {
        self.role = role
        self.content = content
        self.metrics = metrics
    }

    enum Role { case user, assistant }

    struct Metrics {
        let promptTokens: Int
        let outputTokens: Int
        /// Overall tokens/sec over the whole generation. This is the only
        /// reliably measurable rate on the `LanguageModelSession` path: CoreAI
        /// buffers streamed output, so per-phase (prefill vs decode) split is
        /// not exposed. With the S=1 exports these models use, prefill and
        /// decode run at the same per-token rate, so this also equals both.
        let throughput: Double
        /// Time to first output token. Approximate — CoreAI flushes buffered
        /// snapshots, so this is the time to the first flush, not a clean
        /// prefill boundary.
        let ttftMs: Int
    }
}
