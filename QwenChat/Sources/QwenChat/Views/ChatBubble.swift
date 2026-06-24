import SwiftUI

struct ChatBubble: View {
    let message: ChatMessage
    let isStreaming: Bool

    init(message: ChatMessage, isStreaming: Bool = false) {
        self.message = message
        self.isStreaming = isStreaming
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .assistant {
                Image(systemName: "brain.head.profile")
                    .font(.title3)
                    .foregroundStyle(.purple)
                    .frame(width: 32, height: 32)
                    .background(.purple.opacity(0.15))
                    .clipShape(Circle())
            } else {
                Spacer()
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : "Qwen3-0.6B")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)

                HStack(alignment: .top, spacing: 0) {
                    Text(message.content)
                        .font(.body)
                    if isStreaming {
                        Rectangle()
                            .fill(.primary)
                            .frame(width: 2, height: 16)
                            .opacity(0.5)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .foregroundStyle(message.role == .user ? .white : .primary)
                .background(
                    (message.role == .user ? Color.blue : Color.secondary.opacity(0.15)),
                    in: RoundedRectangle(cornerRadius: 18)
                )
            }

            if message.role == .user {
                Image(systemName: "person.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.blue)
                    .frame(width: 32, height: 32)
                    .background(.blue.opacity(0.15))
                    .clipShape(Circle())
            } else {
                Spacer()
            }
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    let timestamp = Date()

    enum Role { case user, assistant }
}
