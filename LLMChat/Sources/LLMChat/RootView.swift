import SwiftUI

/// Library entry point. The per-platform app targets (see `App/project.yml`)
/// host this inside their own `@main App`, e.g.:
///
///     @main struct LLMChatApp: App {
///         var body: some Scene {
///             WindowGroup { RootView().preferredColorScheme(.dark) }
///         }
///     }
public struct RootView: View {
    public init() {}

    public var body: some View {
        GalleryScreen()
    }
}
