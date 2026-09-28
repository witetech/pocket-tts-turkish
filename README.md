# pocket-tts-turkish

Turkish text-to-speech that runs on an ordinary CPU. This package runs the
[Pocket TTS Turkish](https://huggingface.co/wite-tech/pocket-tts-turkish-6l) model, a 6-layer
Turkish version of [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts), and adds the
text preparation the model needs: numbers, dates, times, prices and units are written out in
Turkish, and longer text is spoken one sentence at a time.

- About 5 times faster than real time on a single CPU thread, about 20 times on a GPU
- About 110 million parameters, 24 kHz output
- Six built-in voices, or clone a voice from a short recording
- Five emotion tags: `mutlu`, `üzgün`, `kızgın`, `şaşkın`, `sakin`

## Installation

```bash
pip install pocket-tts-turkish
```

Python 3.10 or newer. On Linux, pip installs the CUDA build of PyTorch by default. For a smaller,
CPU-only setup, install PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pocket-tts-turkish
```

## Quick start

```python
from pocket_tts_turkish import TurkishTTS, save_wav

tts = TurkishTTS.from_pretrained()  # downloads the model once, about 440 MB
audio = tts.generate("Randevunuz 15.10.2026 saat 14:30'da. Ücret 1.250 TL.", voice="female_1", emotion="sakin")
save_wav("randevu.wav", audio, tts.sample_rate)
```

`generate` returns a NumPy `float32` array at `tts.sample_rate` (24000 Hz). Pass `device="cpu"` or
`device="cuda"` to `from_pretrained` to choose where the model runs; the default uses a GPU when one
is available. Pass `seed=` to `generate` to get the same audio every time.

From the command line:

```bash
pocket-tts-turkish generate --text "Merhaba, size nasıl yardımcı olabilirim?" --voice male_1 -o merhaba.wav
pocket-tts-turkish generate --text-file metin.txt --emotion mutlu -o metin.wav
pocket-tts-turkish voices
```

## Voices

| voice | character |
|---|---|
| `female_1` | bright and energetic (default) |
| `female_2` | warm, clear and friendly |
| `female_3` | confident, clear articulation |
| `male_1` | calm and deep, narrator style |
| `male_2` | energetic and bright |
| `male_3` | confident, crisp articulation |

The built-in voices are synthetic. They were designed from text descriptions and are not
recordings of real people.

## Emotions

| tag | English name |
|---|---|
| `mutlu` | `happy` |
| `üzgün` | `sad` |
| `kızgın` | `angry` |
| `şaşkın` | `surprised` |
| `sakin` | `calm` |

Pass the tag or its English name as `emotion`; leave it out for a neutral voice. A tag can also
start the text itself, for example `"[mutlu] Harika bir haber aldım!"`.

## Cloning a voice

```python
voice = tts.voice_from_file("kayit.wav")
audio = tts.generate("Bu cümleyi benim sesimle okuyun.", voice=voice)
```

Use a clean recording of a single speaker, 5 to 10 seconds long. WAV, FLAC, OGG and MP3 all work.
The model needs a reference that stops between two words in the middle of a sentence: a reference
that ends in silence, or sounds finished, makes the model continue that sentence instead of reading
your text. `voice_from_file` therefore cuts the recording at a pause 3 to 5 seconds in, and warns
when it cannot find one. To cut a recording and listen to the result first:

```bash
pocket-tts-turkish cut-reference kayit.wav referans.wav
pocket-tts-turkish generate --text "Merhaba." --voice-file referans.wav -o test.wav
```

Only clone a voice with its owner's permission; see [Responsible use](#responsible-use).

## Text preparation

The model reads plain Turkish words. Before generation the package:

- writes numbers, decimals, percentages, dates, clock times, money and units as words
  (`1.250,50 TL` becomes `bin iki yüz elli virgül elli Türk lirası`, `60 km/sa` becomes `saatte altmış kilometre`)
- reads phone numbers and long codes in groups, the way they are said aloud
  (`0555 123 45 67` becomes `sıfır beş yüz elli beş, yüz yirmi üç, kırk beş, altmış yedi`)
- expands common abbreviations and acronyms (`Dr.`, `Mah.`, `TBMM`)
- replaces letters the model does not know (`w` becomes `v`, `x` becomes `ks`, `q` becomes `k`)
- lowercases the first word of each sentence, which the model reads more reliably, and puts a
  comma before it when there is no emotion tag
- generates each sentence separately and joins them with a short pause

To see exactly what the model will read:

```python
tts.prepare("Toplam 2.345 TL, son ödeme 30.09.2026.")
# [', toplam iki bin üç yüz kırk beş Türk lirası, son ödeme otuz Eylül iki bin yirmi altı.']
```

The same conversion is available without the model, as `pocket_tts_turkish.normalize()` or
`pocket-tts-turkish normalize --text "..."`. Words the model mispronounces can be respelled:

```python
tts = TurkishTTS.from_pretrained(pronunciations={"WhatsApp": "Vatsap", "iPhone": "ayfon"})
```

## Evaluation

The model was compared with eight public Turkish-capable systems: FreyaTTS-small, Piper
(`tr_TR-dfki-medium`), MMS-TTS, VoxCPM2, Trendyol-TTS, Chatterbox Multilingual, Qwen3-TTS 0.6B Turkish
and XTTS-v2. Two public test sets were used: Freya-TR-Eval (T1, 495 everyday conversational
sentences) and the FLEURS Turkish test set (T2, 200 long read sentences). Every system received the
same text, with numbers already written as words. The word error rate (WER) is the share of words
that Whisper large-v3 transcribes differently from the input text, measured on audio band-limited to
8 kHz as in the Freya-TR-Eval recipe; lower is better. Speed was measured on one RTX 5090.

### Speed and accuracy

![Speed against intelligibility](https://huggingface.co/wite-tech/pocket-tts-turkish-6l/resolve/main/figures/fig2_speed_vs_wer.png)

Each bubble is one system. Its horizontal position is the real-time factor, the synthesis time
divided by the length of the audio, on a log scale: further left is faster, and 0.1 means ten times
faster than real time. Its height is the WER on T1, and its area follows the number of parameters.
The best place to be is the lower left.

Pocket TTS Turkish reaches 1.9% WER at a real-time factor of 0.048, about 20 times faster than real
time, with 110 million parameters. The two systems with a lower WER, Trendyol-TTS (1.1%) and
Qwen3-TTS (1.7%), are 8 to 22 times larger and 7 to 9 times slower. The two systems that are faster,
Piper and MMS-TTS, make more errors (3.2% and 6.1%) and cannot clone a voice.

### What a real-time voice agent needs

![What a real-time voice agent needs, and which systems deliver it](https://huggingface.co/wite-tech/pocket-tts-turkish-6l/resolve/main/figures/fig8_capability_matrix.png)

The columns are nine requirements for a voice agent that answers in real time: it streams audio,
clones a reference voice, has an emotion control, runs on a CPU faster than real time, keeps the WER
at or below 2.5% on T1 and 5% on T2, synthesizes a 5-second reply in under half a second, uses less
than 2 GB of GPU memory, and has a licence that allows commercial use. A filled circle means the
system meets the requirement and a cross means it does not; the number on the right is the total.

Pocket TTS Turkish is the only system that meets all nine; the next best, VoxCPM2, meets six. A
requirement counts as met only where it was measured, so systems that were not run on a CPU are
marked as not running on one.

## Limitations

- Turkish only.
- The whole text is generated before it is returned; there is no streaming output yet.

## Responsible use

The model can imitate a voice from a few seconds of audio. Only clone a voice with the explicit
consent of the person it belongs to. Do not use the model for impersonation, fraud, fraudulent
calls or misinformation, or to present generated speech as a real recording of a person. The same
terms apply to the upstream Pocket TTS model.

## License

The code in this repository is released under the Apache License 2.0 (see [LICENSE](LICENSE)). The
model weights are published on Hugging Face under CC-BY-4.0; they are derived from Kyutai's Pocket
TTS, which is also released under CC-BY-4.0.

## Citation

```bibtex
@misc{pocket_tts_turkish_2026,
  title  = {Pocket TTS Turkish},
  author = {{Wite Tech}},
  year   = {2026},
  url    = {https://huggingface.co/wite-tech/pocket-tts-turkish-6l}
}
```

Please also cite the work this model is built on:

```bibtex
@article{rouard2025continuous,
  title         = {Continuous Audio Language Models},
  author        = {Rouard, Simon and Orsini, Manu and Roebel, Axel and Zeghidour, Neil and D\'efossez, Alexandre},
  year          = {2025},
  eprint        = {2509.06926},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```
