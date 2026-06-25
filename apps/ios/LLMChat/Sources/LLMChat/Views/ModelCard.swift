import SwiftUI

/// Compact model header shown at the top of the chat screen. Reads identity
/// from the active `ChatViewModel`'s descriptor.
struct ModelCard: View {
    @EnvironmentObject var vm: ChatViewModel

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .fill(vm.model.accentColor.opacity(0.15))
                    .frame(width: 48, height: 48)
                Image(systemName: vm.model.systemIcon)
                    .font(.title2)
                    .foregroundStyle(vm.model.accentColor)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(vm.model.displayName).font(.headline)
                HStack(spacing: 4) {
                    Badge(text: vm.model.quant, color: .green)
                    Badge(text: vm.model.params, color: .blue)
                    if let rom = vm.model.romSize {
                        Badge(text: rom, color: .orange)
                    }
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Circle()
                    .fill(vm.modelLoadState == .ready ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(vm.modelLoadState.rawValue)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
