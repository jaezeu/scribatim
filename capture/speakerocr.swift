// Susurro active-speaker OCR helper (experimental).
// Finds the Zoom / Microsoft Teams meeting window, captures it ~once per
// second with ScreenCaptureKit, and runs Apple's on-device Vision OCR on each
// frame. Emits one JSON line per frame to stdout:
//
//   {"time": 1720000000.0, "texts": [{"text": "Alice Chen", "x": 0.02,
//    "y": 0.06, "w": 0.1, "h": 0.03, "conf": 0.98}, ...]}
//
// Coordinates are normalized to the window, origin bottom-left (Vision's
// convention). Deciding which text is the speaker's name is the Python
// side's job — keeping this helper dumb means heuristics can evolve without
// recompiling.
//
// Requires the Screen Recording privacy permission (macOS prompts on first
// run). Frames are OCR'd in memory and discarded; nothing is written to disk
// and nothing leaves the machine.

import CoreMedia
import Foundation
import ScreenCaptureKit
import Vision

func log(_ message: String) {
    FileHandle.standardError.write(("[speaker] " + message + "\n").data(using: .utf8)!)
}

let MEETING_BUNDLES = [
    "us.zoom.xos",            // Zoom
    "com.microsoft.teams2",   // new Teams
    "com.microsoft.teams",    // classic Teams
]

// Main/app windows we must never OCR: Teams keeps its Chat/Activity window
// open next to the meeting pop-out, and chat sender names sit bottom-left
// too — capturing the wrong window would *mislabel* speakers, not just miss.
let NON_MEETING_TITLES: Set<String> = [
    "zoom", "zoom workplace", "zoom cloud meetings", "settings", "microsoft teams",
]
let MAIN_WINDOW_PREFIXES = [
    "chat |", "activity |", "calendar |", "calls |", "files |",
    "teams |", "apps |", "onedrive |", "copilot |", "communities |",
]

func looksLikeMeetingWindow(_ window: SCWindow) -> Bool {
    let title = (window.title ?? "").lowercased()
    if NON_MEETING_TITLES.contains(title) { return false }
    for prefix in MAIN_WINDOW_PREFIXES where title.hasPrefix(prefix) { return false }
    return true
}

func findMeetingWindow(_ content: SCShareableContent) -> SCWindow? {
    let candidates = content.windows.filter { w in
        guard let app = w.owningApplication else { return false }
        return MEETING_BUNDLES.contains(app.bundleIdentifier)
            && w.isOnScreen && w.frame.width >= 500 && w.frame.height >= 350
            && looksLikeMeetingWindow(w)
    }
    // among plausible meeting windows, the biggest is the meeting
    return candidates.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height })
}

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let ocrQueue = DispatchQueue(label: "susurro.speaker.ocr")
    private var announced = false

    func scan() {
        guard stream == nil else { return }
        SCShareableContent.getExcludingDesktopWindows(true, onScreenWindowsOnly: true) { content, error in
            if let error = error {
                log("""
                cannot list windows: \(error.localizedDescription)
                Grant permission in: System Settings → Privacy & Security →
                Screen & System Audio Recording → enable your terminal, then retry.
                """)
                exit(2)
            }
            guard let content = content, let window = findMeetingWindow(content) else {
                if !self.announced {
                    log("waiting for a Zoom/Teams meeting window…")
                    self.announced = true
                }
                return
            }
            self.attach(to: window)
        }
    }

    private func attach(to window: SCWindow) {
        let title = window.title ?? "?"
        let app = window.owningApplication?.applicationName ?? "?"
        log("capturing \(app) window \"\(title)\" (\(Int(window.frame.width))×\(Int(window.frame.height)))")

        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        // 2× the window's point size: retina-sharp text without full 5K frames
        config.width = min(Int(window.frame.width) * 2, 3200)
        config.height = min(Int(window.frame.height) * 2, 2000)
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)  // ≤1 fps
        config.queueDepth = 3
        config.showsCursor = false

        let s = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: ocrQueue)
        } catch {
            log("stream output failed: \(error.localizedDescription)")
            return
        }
        s.startCapture { error in
            if let error = error {
                log("capture failed: \(error.localizedDescription)")
                self.stream = nil
                return
            }
        }
        stream = s
        announced = false
    }

    // window closed / meeting ended → go back to scanning
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("capture stopped (\(error.localizedDescription)) — rescanning")
        self.stream = nil
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .screen,
              sampleBuffer.isValid,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        ocr(pixelBuffer)
    }

    private func ocr(_ pixelBuffer: CVPixelBuffer) {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate  // .fast is Latin-only; names can be CJK
        request.usesLanguageCorrection = false
        if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do { try handler.perform([request]) } catch { return }

        var texts: [[String: Any]] = []
        for observation in request.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let box = observation.boundingBox
            texts.append([
                "text": candidate.string,
                "x": (box.minX * 1000).rounded() / 1000,
                "y": (box.minY * 1000).rounded() / 1000,
                "w": (box.width * 1000).rounded() / 1000,
                "h": (box.height * 1000).rounded() / 1000,
                "conf": (Double(candidate.confidence) * 100).rounded() / 100,
            ])
        }
        let payload: [String: Any] = ["time": Date().timeIntervalSince1970, "texts": texts]
        guard var data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        data.append(0x0A)  // newline
        FileHandle.standardOutput.write(data)
    }
}

let capture = Capture()
capture.scan()
Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in capture.scan() }

for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler {
        log("shutting down")
        exit(0)
    }
    source.resume()
    _ = Unmanaged.passRetained(source)
}

// If the Python parent dies, our stdout pipe closes; exit instead of lingering.
signal(SIGPIPE) { _ in exit(0) }

RunLoop.main.run()
