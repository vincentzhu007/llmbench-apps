import SwiftUI

struct ModelCard: View {
    @EnvironmentObject var vm: ChatViewModel

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .fill(.purple.opacity(0.15))
                    .frame(width: 48, height: 48)
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundStyle(.purple)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(vm.modelName)
                    .font(.headline)
                HStack(spacing: 4) {
                    Badge(text: vm.modelSize, color: .blue)
                    Badge(text: vm.modelFormat, color: .green)
                    Badge(text: vm.modelSpeed, color: .orange)
                }
            }

            Spacer()

            Button {
                Task { await vm.loadModel() }
            } label: {
                HStack(spacing: 4) {
                    Circle()
                        .fill(vm.modelLoadState == .ready ? .green : .orange)
                        .frame(width: 8, height: 8)
                    Text(vm.modelLoadState.rawValue)
                        .font(.caption.weight(.medium))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
            }
            .disabled(vm.modelLoadState == .loading || vm.modelLoadState == .ready)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct Badge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}
