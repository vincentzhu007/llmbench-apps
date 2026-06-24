import SwiftUI
import LLMChat

/// macOS SwiftUI host so the app can be launched with `swift run LLMChatRunner`
/// during development (no Xcode GUI required). The shipped iOS / macOS apps
/// live in App/project.yml and host the same `RootView`.
@main
struct LLMChatRunnerApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .frame(minWidth: 480, idealWidth: 820, minHeight: 600, idealHeight: 720)
                .preferredColorScheme(.dark)
        }
    }
}
