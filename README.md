# RustyEcho

RustyEcho is a fast speech to text gateway using candle and Whisper

## How to Run

The example can be run directly to see how it works

Run with the default audio file
```bash
cargo run -p rustyecho-inference --example transcribe_file --release
```

Run with a custom audio file
```bash
cargo run -p rustyecho-inference --example transcribe_file --release -- "path\to\custom\file.wav"
```

Note that the audio must be a 16-bit PCM wav file
The pipeline automatically handles mono or stereo and any sample rate

## Performance

The default Whisper tiny.en model was tested on CPU
Here is the performance breakdown

| Metric | Time | What it means |
|--------|------|---------------|
| Model Load | 0.69s | Happens once when the server boots and not per request |
| Actual Transcription | 1.74s | Time taken to transcribe 11 seconds of audio |

The audio is 11 seconds long and it takes 1.74 seconds to transcribe it
This means the model processes audio about 6.3x faster than real time using just the CPU

Because the streaming design cuts audio into chunks of at most 5 seconds the decoding only takes about 0.8 seconds per chunk
This allows the server to keep up comfortably with live speech without falling behind

## Tradeoffs

A few things to keep in mind

* This test uses tiny.en which is the smallest and fastest model
* Using bigger models like base.en gives better accuracy but runs slower
* Under heavy load the speed may drop because the worker pool defaults to 2 parallel slots
* A third simultaneous stream will queue behind the first two
