import SwiftUI

/// Root Gallery: a responsive grid of model cards. Tapping a card pushes the
/// per-model `ChatScreen`. iPhone renders a single column; iPad/Mac reflows
/// into multiple columns via an adaptive grid.
struct GalleryScreen: View {
    private let columns = [GridItem(.adaptive(minimum: 300), spacing: 16)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    Text("On-device LLMs")
                        .font(.title2.bold())
                    Text("Pick a model to chat. Prefill & decode speed are measured live.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                .padding(.top, 8)

                LazyVGrid(columns: columns, spacing: 16) {
                    ForEach(ModelRegistry.all) { model in
                        NavigationLink(value: model) {
                            GalleryCard(model: model)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding()
            }
            .navigationTitle("Models")
            .navigationDestination(for: ModelDescriptor.self) { model in
                ChatScreen(model: model)
            }
        }
    }
}
