// Scribatim active-speaker OCR helper (experimental).
// Finds the Zoom / Microsoft Teams meeting window, captures it ~once per
// second with ScreenCaptureKit, and runs Apple's on-device Vision OCR on each
// frame. Emits one JSON line per frame to stdout:
//
//   {"time": 1720000000.0, "texts": [{"text": "Alice Chen", "x": 0.02,
//    "y": 0.06, "w": 0.1, "h": 0.03, "conf": 0.98,
//    "gl": [0.31, 0.28, 258, 251]}, ...]}
//
// Coordinates are normalized to the window, origin bottom-left (Vision's
// convention). "gl" (present when non-trivial) samples the pixels in thin
// bands just left of and just below the text box — where Zoom/Teams draw
// the colored outline around the actively speaking tile, since name labels
// sit at a tile's bottom-left. It is [left_frac, bottom_frac, left_hue,
// bottom_hue]: the fraction of vividly colored pixels in each band and
// their mean hue in degrees. Purely mechanical: deciding which text is the
// speaker's name is the Python side's job — keeping this helper dumb means
// heuristics can evolve without recompiling.
//
// Requires the Screen Recording privacy permission (macOS prompts on first
// run). Frames are OCR'd in memory and discarded; nothing is written to disk
// and nothing leaves the machine.

import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit
import Vision

func log(_ message: String) {
    FileHandle.standardError.write(("[speaker] " + message + "\n").data(using: .utf8)!)
}

// CLI tools have no window-server connection until something initializes
// CoreGraphics; starting an SCStream without one aborts with
// "Assertion failed: (did_initialize), CGS_REQUIRE_INIT". Touching the main
// display establishes the connection up front.
_ = CGMainDisplayID()

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
    private let ocrQueue = DispatchQueue(label: "scribatim.speaker.ocr")
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
            // stream setup must run on the main thread, not SCK's callback queue
            DispatchQueue.main.async { self.attach(to: window) }
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
        config.pixelFormat = kCVPixelFormatType_32BGRA  // glow sampling reads BGRA

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

    // Fraction of vividly colored pixels in a pixel rectangle, and their mean
    // hue (degrees). Vivid = bright and saturated, like the accent-colored
    // outline meeting apps draw around the speaking tile — video content and
    // dark chrome mostly isn't.
    private func vividness(_ base: UnsafePointer<UInt8>, bytesPerRow: Int,
                           width: Int, height: Int,
                           x0: Int, x1: Int, y0: Int, y1: Int) -> (Double, Double)? {
        var total = 0, vivid = 0
        var sinSum = 0.0, cosSum = 0.0
        var y = max(0, y0)
        while y < min(height, y1) {
            var x = max(0, x0)
            while x < min(width, x1) {
                let p = base + y * bytesPerRow + x * 4  // BGRA
                let b = Double(p[0]) / 255, g = Double(p[1]) / 255, r = Double(p[2]) / 255
                let mx = max(r, max(g, b)), mn = min(r, min(g, b))
                total += 1
                if mx > 0.35, mx - mn > 0.45 * mx {
                    vivid += 1
                    let d = mx - mn
                    var hue: Double
                    if mx == r { hue = (g - b) / d }
                    else if mx == g { hue = (b - r) / d + 2 }
                    else { hue = (r - g) / d + 4 }
                    hue *= 60
                    if hue < 0 { hue += 360 }
                    sinSum += sin(hue * .pi / 180)
                    cosSum += cos(hue * .pi / 180)
                }
                x += 2
            }
            y += 2
        }
        guard total > 0 else { return nil }
        var hue = 0.0
        if vivid > 0 {
            hue = atan2(sinSum, cosSum) * 180 / .pi
            if hue < 0 { hue += 360 }
        }
        return (Double(vivid) / Double(total), hue)
    }

    // Sample the bands left of and below a text box (in Vision's normalized,
    // bottom-left-origin coordinates) and return the "gl" payload, or nil
    // when there is no meaningful color there.
    private func glow(_ pixelBuffer: CVPixelBuffer, box: CGRect,
                      width: Int, height: Int, bytesPerRow: Int,
                      base: UnsafePointer<UInt8>) -> [Double]? {
        // labels are one short line; skip paragraphs of shared content
        guard box.height < 0.06, box.width < 0.35 else { return nil }
        let x0 = Int(box.minX * Double(width))
        let x1 = Int(box.maxX * Double(width))
        let yTop = Int((1 - box.maxY) * Double(height))
        let yBottom = Int((1 - box.minY) * Double(height))
        guard let left = vividness(base, bytesPerRow: bytesPerRow, width: width,
                                   height: height, x0: x0 - 26, x1: x0 - 4,
                                   y0: yTop, y1: yBottom),
              let bottom = vividness(base, bytesPerRow: bytesPerRow, width: width,
                                     height: height, x0: x0, x1: x1,
                                     y0: yBottom + 3, y1: yBottom + 16)
        else { return nil }
        guard left.0 > 0.03, bottom.0 > 0.03 else { return nil }
        return [(left.0 * 1000).rounded() / 1000,
                (bottom.0 * 1000).rounded() / 1000,
                left.1.rounded(), bottom.1.rounded()]
    }

    private func ocr(_ pixelBuffer: CVPixelBuffer) {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate  // .fast is Latin-only; names can be CJK
        request.usesLanguageCorrection = false
        if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do { try handler.perform([request]) } catch { return }

        var pixelBase: UnsafePointer<UInt8>? = nil
        var pxWidth = 0, pxHeight = 0, pxBytesPerRow = 0
        var locked = false
        if CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
           CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess {
            locked = true
            if let raw = CVPixelBufferGetBaseAddress(pixelBuffer) {
                pixelBase = UnsafePointer(raw.assumingMemoryBound(to: UInt8.self))
                pxWidth = CVPixelBufferGetWidth(pixelBuffer)
                pxHeight = CVPixelBufferGetHeight(pixelBuffer)
                pxBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            }
        }
        defer {
            if locked { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        }

        var texts: [[String: Any]] = []
        for observation in request.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let box = observation.boundingBox
            var item: [String: Any] = [
                "text": candidate.string,
                "x": (box.minX * 1000).rounded() / 1000,
                "y": (box.minY * 1000).rounded() / 1000,
                "w": (box.width * 1000).rounded() / 1000,
                "h": (box.height * 1000).rounded() / 1000,
                "conf": (Double(candidate.confidence) * 100).rounded() / 100,
            ]
            if let base = pixelBase,
               let gl = glow(pixelBuffer, box: box, width: pxWidth, height: pxHeight,
                             bytesPerRow: pxBytesPerRow, base: base) {
                item["gl"] = gl
            }
            texts.append(item)
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
