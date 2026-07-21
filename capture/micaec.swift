// Scribatim microphone capture with Apple voice processing.
// Captures the default microphone through the system's voice-processing unit
// (the FaceTime engine): echo cancellation — anything the Mac is playing is
// subtracted from the mic signal — plus noise suppression and automatic gain.
// Lets the mic lane hear only the user even on open speakers.
//
// Same stdout protocol as the system tap:
//
//   line 1 (utf8): {"rate": 24000, "channels": 1}\n
//   then:          raw little-endian float32 samples
//
// Requires the Microphone privacy permission (macOS prompts on first run).
// Nothing is written to disk and nothing leaves the machine.

import AVFoundation
import CoreAudio
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("[mic-aec] ERROR: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

func log(_ message: String) {
    FileHandle.standardError.write(("[mic-aec] " + message + "\n").data(using: .utf8)!)
}

// MARK: - Input gain guard
//
// The voice-processing unit adjusts the *hardware* input gain, which is
// shared machine-wide. Disabling AGC (below) is supposed to stop that, but
// macOS still winds the physical input volume down when voice processing
// engages — and Teams/Zoom read the same turned-down device, so the user
// suddenly sounds faint *in their own meeting*. Snapshot the gain the user
// chose before the voice unit can touch it, put it back whenever it drops
// while we run, and leave it restored on exit.
final class InputGainGuard {
    private let device: AudioDeviceID
    private var baselines: [(address: AudioObjectPropertyAddress, value: Float32)] = []
    private var lastLog = Date.distantPast

    init?() {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var dev = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                         &addr, 0, nil, &size, &dev) == noErr,
              dev != kAudioObjectUnknown else { return nil }
        device = dev
        // main element first, then per-channel volumes (devices expose either)
        for element in [kAudioObjectPropertyElementMain, 1, 2] as [UInt32] {
            var a = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyVolumeScalar,
                mScope: kAudioDevicePropertyScopeInput,
                mElement: element)
            var settable = DarwinBoolean(false)
            guard AudioObjectHasProperty(device, &a),
                  AudioObjectIsPropertySettable(device, &a, &settable) == noErr,
                  settable.boolValue, let v = read(a) else { continue }
            baselines.append((a, v))
        }
        if baselines.isEmpty { return nil }
    }

    private func read(_ address: AudioObjectPropertyAddress) -> Float32? {
        var a = address
        var v = Float32(0)
        var size = UInt32(MemoryLayout<Float32>.size)
        return AudioObjectGetPropertyData(device, &a, 0, nil, &size, &v) == noErr ? v : nil
    }

    private func write(_ address: AudioObjectPropertyAddress, _ value: Float32) {
        var a = address
        var v = value
        AudioObjectSetPropertyData(device, &a, 0, nil,
                                   UInt32(MemoryLayout<Float32>.size), &v)
    }

    // Re-check on every gain change the HAL reports. Restoring only *drops*
    // below the snapshot means a user deliberately raising their mic volume
    // mid-meeting is never fought; enforce() writing the baseline back does
    // not re-trigger itself (the guard below it is then false).
    func watch() {
        for (address, _) in baselines {
            var a = address
            AudioObjectAddPropertyListenerBlock(device, &a, DispatchQueue.main) {
                [weak self] _, _ in self?.enforce()
            }
        }
    }

    func enforce() {
        for (address, baseline) in baselines {
            guard let current = read(address), current < baseline - 0.02 else { continue }
            write(address, baseline)
            if Date().timeIntervalSince(lastLog) > 5 {
                lastLog = Date()
                log(String(format:
                    "input gain was wound down to %.2f — restored to %.2f",
                    current, baseline))
            }
        }
    }

    func restore() {
        for (address, baseline) in baselines { write(address, baseline) }
    }
}

// Snapshot BEFORE voice processing is enabled — that's the gain the user set.
let gainGuard = InputGainGuard()
if gainGuard == nil {
    log("input gain guard unavailable (gain not adjustable on this device) — skipping")
}

let engine = AVAudioEngine()
let input = engine.inputNode

// Voice processing must be enabled on both I/O nodes; the output node renders
// silence — we never connect anything audible to it.
do {
    try input.setVoiceProcessingEnabled(true)
    try engine.outputNode.setVoiceProcessingEnabled(true)
} catch {
    fail("voice processing unavailable: \(error.localizedDescription)")
}

// Don't let the voice unit duck other apps' playback — that would quiet the
// very meeting audio the system tap is transcribing.
if #available(macOS 14.0, *) {
    input.voiceProcessingOtherAudioDuckingConfiguration =
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .min)
}

// AGC adjusts the *hardware* input gain, which is shared machine-wide: with
// meeting audio playing, it winds the mic down and the user goes quiet in
// the actual call (Teams/Zoom read the same turned-down mic). Echo
// cancellation is what we want from the voice unit — gain control is not.
input.isVoiceProcessingAGCEnabled = false

let format = input.outputFormat(forBus: 0)
let rate = Int(format.sampleRate)
let channels = Int(format.channelCount)
guard rate > 0, channels > 0 else {
    fail("no usable microphone input format (rate \(rate), \(channels) ch) — is a mic connected?")
}
log("voice-processing mic: \(rate) Hz, \(channels) ch")

let header = "{\"rate\": \(rate), \"channels\": 1}\n"
fwrite(header, 1, header.utf8.count, stdout)
fflush(stdout)

// The voice-processing unit can report phantom channels (e.g. 9 ch for a
// 1-ch laptop mic), most of them silent — averaging would dilute the voice.
// Stream the strongest channel of each buffer instead.
input.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
    guard let channelData = buffer.floatChannelData else { return }
    let frames = Int(buffer.frameLength)
    guard frames > 0 else { return }

    var best = 0
    if channels > 1 {
        var bestEnergy: Float = -1
        for ch in 0..<channels {
            let p = channelData[ch]
            var energy: Float = 0
            for i in 0..<frames { energy += p[i] * p[i] }
            if energy > bestEnergy { bestEnergy = energy; best = ch }
        }
    }
    _ = fwrite(channelData[best], 4, frames, stdout)
}

engine.prepare()
do {
    try engine.start()
} catch {
    fail("engine start failed: \(error.localizedDescription)")
}
log("streaming echo-cancelled microphone (ctrl-c to stop)")

// The voice unit typically drops the gain right as the engine starts (and
// sometimes again a moment later) — check immediately, shortly after, and
// then on every change the HAL reports.
gainGuard?.enforce()
gainGuard?.watch()
DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { gainGuard?.enforce() }

// MARK: - Clean shutdown

func teardown() {
    engine.stop()
    // the voice unit does not put the hardware gain back when it disengages
    gainGuard?.restore()
    fflush(stdout)
}

for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler {
        log("shutting down")
        teardown()
        exit(0)
    }
    source.resume()
    // Keep sources alive for the lifetime of the process.
    _ = Unmanaged.passRetained(source)
}

// If the Python parent dies, our stdout pipe closes; exit instead of lingering.
signal(SIGPIPE) { _ in
    teardown()
    exit(0)
}

RunLoop.main.run()
