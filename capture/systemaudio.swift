// Scribatim system-audio tap.
// Captures the Mac's system audio output (any app: Teams, Zoom, Meet, browser)
// via a Core Audio process tap (macOS 14.4+), mixes it to mono float32 at the
// device's native sample rate, and streams it to stdout:
//
//   line 1 (utf8): {"rate": 48000, "channels": 1}\n
//   then:          raw little-endian float32 samples
//
// Requires the "System Audio Recording" privacy permission (macOS prompts on
// first run). Nothing is written to disk and nothing leaves the machine.

import Foundation
import CoreAudio
import AudioToolbox

func fail(_ message: String, status: OSStatus = noErr) -> Never {
    let detail = status != noErr ? " (OSStatus \(status))" : ""
    FileHandle.standardError.write(("[tap] ERROR: " + message + detail + "\n").data(using: .utf8)!)
    exit(1)
}

func log(_ message: String) {
    FileHandle.standardError.write(("[tap] " + message + "\n").data(using: .utf8)!)
}

// MARK: - Default output device UID (the tap follows what you actually hear)

func defaultOutputDeviceUID() -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
    if status != noErr { fail("cannot find default output device", status: status) }

    var uid: CFString = "" as CFString
    address.mSelector = kAudioDevicePropertyDeviceUID
    size = UInt32(MemoryLayout<CFString>.size)
    status = withUnsafeMutablePointer(to: &uid) { ptr in
        AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, ptr)
    }
    if status != noErr { fail("cannot read output device UID", status: status) }
    return uid as String
}

// MARK: - Create the process tap (global: every app's output)

let tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDescription.name = "ScribatimTap"
tapDescription.isPrivate = true
tapDescription.muteBehavior = .unmuted  // never alter what the user hears

var tapID = AudioObjectID(kAudioObjectUnknown)
var status = AudioHardwareCreateProcessTap(tapDescription, &tapID)
if status != noErr {
    fail("""
    could not create system audio tap. Grant permission in:
    System Settings → Privacy & Security → Screen & System Audio Recording
    → System Audio Recording Only → enable your terminal / VS Code, then retry.
    """, status: status)
}

// Tap stream format (float32, device sample rate)
var asbd = AudioStreamBasicDescription()
var formatAddress = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
status = AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &asbdSize, &asbd)
if status != noErr { fail("cannot read tap format", status: status) }

let sampleRate = Int(asbd.mSampleRate)
let tapChannels = Int(asbd.mChannelsPerFrame)
log("tap created: \(sampleRate) Hz, \(tapChannels) ch")

// MARK: - Private aggregate device hosting the tap

let outputUID = defaultOutputDeviceUID()
let aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey: "Scribatim Aggregate",
    kAudioAggregateDeviceUIDKey: "com.scribatim.aggregate." + UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [
        [kAudioSubDeviceUIDKey: outputUID]
    ],
    kAudioAggregateDeviceTapListKey: [
        [
            kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
            kAudioSubTapDriftCompensationKey: true,
        ]
    ],
]

var aggregateID = AudioObjectID(kAudioObjectUnknown)
status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID)
if status != noErr { fail("cannot create aggregate device", status: status) }

// MARK: - Stream header, then raw audio on stdout

let header = "{\"rate\": \(sampleRate), \"channels\": 1}\n"
fwrite(header, 1, header.utf8.count, stdout)
fflush(stdout)

var monoScratch = [Float](repeating: 0, count: 8192)

var ioProcID: AudioDeviceIOProcID?
status = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateID, nil) {
    _, inInputData, _, _, _ in
    let bufferList = UnsafeMutableAudioBufferListPointer(
        UnsafeMutablePointer(mutating: inInputData))
    guard bufferList.count > 0 else { return }

    // Frame count from the first buffer
    let first = bufferList[0]
    let firstChannels = max(Int(first.mNumberChannels), 1)
    let frames = Int(first.mDataByteSize) / (4 * firstChannels)
    guard frames > 0 else { return }

    if monoScratch.count < frames { monoScratch = [Float](repeating: 0, count: frames) }
    for i in 0..<frames { monoScratch[i] = 0 }

    // Average every channel across every buffer into mono
    var totalChannels = 0
    for buffer in bufferList {
        guard let data = buffer.mData else { continue }
        let channels = max(Int(buffer.mNumberChannels), 1)
        let bufFrames = min(frames, Int(buffer.mDataByteSize) / (4 * channels))
        let samples = data.assumingMemoryBound(to: Float.self)
        for frame in 0..<bufFrames {
            var sum: Float = 0
            for ch in 0..<channels { sum += samples[frame * channels + ch] }
            monoScratch[frame] += sum
        }
        totalChannels += channels
    }
    if totalChannels > 1 {
        let scale = 1.0 / Float(totalChannels)
        for i in 0..<frames { monoScratch[i] *= scale }
    }

    monoScratch.withUnsafeBufferPointer { ptr in
        _ = fwrite(ptr.baseAddress!, 4, frames, stdout)
    }
}
if status != noErr { fail("cannot create IO proc", status: status) }

status = AudioDeviceStart(aggregateID, ioProcID)
if status != noErr { fail("cannot start aggregate device", status: status) }
log("streaming system audio (ctrl-c to stop)")

// MARK: - Clean shutdown

func teardown() {
    if let proc = ioProcID {
        AudioDeviceStop(aggregateID, proc)
        AudioDeviceDestroyIOProcID(aggregateID, proc)
    }
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
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

// SIGPIPE only fires on a *write*, and a wedged audio callback stops writing —
// an orphaned helper then holds its aggregate device open forever, which can
// hang every new CoreAudio client on the machine. Watch the parent directly:
// getppid() flips to 1 (launchd) the moment it dies, whatever the cause.
let parentWatch = DispatchSource.makeTimerSource(queue: .main)
parentWatch.schedule(deadline: .now() + 1, repeating: 1)
parentWatch.setEventHandler {
    if getppid() == 1 {
        log("parent process gone — shutting down")
        teardown()
        exit(0)
    }
}
parentWatch.resume()

RunLoop.main.run()
