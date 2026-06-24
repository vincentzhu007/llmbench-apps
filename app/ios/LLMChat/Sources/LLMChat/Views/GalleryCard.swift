import SwiftUI

/// One tile in the Gallery grid. Shows the model's identity, specs, and whether
/// its bundle is actually present on this device.
struct GalleryCard: View {
    let model: ModelDescriptor

    private var isAvailable: Bool { model.resolveBundleURL() != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(model.accentColor.opacity(0.15))
                    Image(systemName: model.systemIcon)
                        .font(.title2)
                        .foregroundStyle(model.accentColor)
                }
                .frame(width: 48, height: 48)

                VStack(alignment: .leading, spacing: 2) {
                    Text(model.displayName)
                        .font(.headline)
                    Text(model.quant)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                InfoRow(label: "Params", value: model.params)
                InfoRow(label: "Vocab", value: model.vocab)
            }
            .font(.caption)

            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Circle()
                        .fill(isAvailable ? Color.green : Color.orange)
                        .frame(width: 7, height: 7)
                    Text(isAvailable ? "Available" : "Not found")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Badge(text: "~\(model.estimatedDecodeTokPerSec) tok/s", color: .orange)
                if let rom = model.romSize {
                    Badge(text: rom, color: .blue)
                }
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(model.accentColor.opacity(0.2), lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: 20))
    }
}

private struct InfoRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).foregroundStyle(.primary).monospacedDigit()
        }
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
