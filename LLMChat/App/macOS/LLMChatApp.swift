import SwiftUI
import LLMChat

@main
struct LLMChatApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .frame(minWidth: 480, minHeight: 600)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 820, height: 720)
    }
}
