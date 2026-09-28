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
- reads phone numbers and long codes digit by digit
- expands common abbreviations and acronyms (`Dr.`, `Mah.`, `TBMM`)
- replaces letters the model does not know (`w` becomes `v`, `x` becomes `ks`, `q` becomes `k`)
- generates each sentence separately and joins them with a short pause

To see exactly what the model will read:

```python
tts.prepare("Toplam 2.345 TL, son ödeme 30.09.2026.")
# ['Toplam iki bin üç yüz kırk beş Türk lirası, son ödeme otuz Eylül iki bin yirmi altı.']
```

The same conversion is available without the model, as `pocket_tts_turkish.normalize()` or
`pocket-tts-turkish normalize --text "..."`. Words the model mispronounces can be respelled:

```python
tts = TurkishTTS.from_pretrained(pronunciations={"WhatsApp": "Vatsap", "iPhone": "ayfon"})
```

## Evaluation

The model was compared with eight public Turkish-capable systems on public test sets:

- **T1**: Freya-TR-Eval, 495 everyday conversational sentences
- **T2**: FLEURS Turkish test, 200 long read sentences
- **T3**: 50 sentences with large numbers, phone numbers, dates, prices and brand names

Every system received the same text, with numbers already written as words, and every
voice-cloning system received the same two reference voices.

| system | params (M) | WER T1 | WER T2 | CER T3 | UTMOS | speaker similarity | RTF GPU | RTF CPU | GPU memory (MiB) | cloning | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Pocket TTS Turkish** | 109.5 | 1.9 | 3.9 | 2.1 | 3.21 | 0.950 | 0.048 | 0.20 | 1704 | yes | CC-BY-4.0 |
| FreyaTTS-small | 183.2 | 11.7 | 43.1 | 10.5 | 2.81 | – | 0.049 | 1.91 | 2154 | no | Apache-2.0 |
| Piper tr_TR-dfki-medium | 15.8 | 3.2 | 5.1 | 2.0 | 3.70 | – | 0.015 | 0.01 | 504 | no | MIT |
| MMS-TTS tur | 36.3 | 6.1 | 8.5 | 6.1 | 3.79 | – | 0.004 | 0.19 | 1040 | no | CC-BY-NC-4.0 |
| VoxCPM2 | 2384.2 | 2.1 | 3.9 | 3.7 | 3.12 | 0.958 | 0.243 | – | 6934 | yes | Apache-2.0 |
| Trendyol-TTS | 2384.2 | 1.1 | 3.1 | 1.7 | 3.84 | – | 0.337 | – | 6934 | no | MIT |
| Chatterbox Multilingual | 799.9 | 2.1 | 8.6 | 2.2 | 3.54 | 0.959 | 0.232 | – | 4650 | yes | MIT |
| Qwen3-TTS 0.6B Turkish | 914.6 | 1.7 | 3.9 | 1.7 | 3.81 | 0.936 | 0.439 | – | 3152 | yes | Apache-2.0 |
| XTTS-v2 | 466.9 | 3.8 | 5.5 | 3.0 | 3.12 | 0.958 | 0.116 | – | 2896 | yes | CPML |

- **WER / CER** (%, lower is better): transcripts by Whisper large-v3 after the audio is
  band-limited to 8 kHz, compared with the input text after the same Turkish normalization, at
  corpus level (the Freya-TR-Eval recipe).
- **UTMOS** (1 to 5, higher is better): an automatic prediction of how natural the speech sounds.
- **Speaker similarity** (higher is better): cosine similarity of WavLM speaker embeddings between
  the output and the reference voice, for cloning systems.
- **RTF** (lower is faster): synthesis time divided by audio length, on one RTX 5090 or one CPU
  thread; "–" means not measured.
- Pocket TTS Turkish was run with the same sentence-by-sentence generation this package uses.
- The two reference voices given to all cloning systems (`female_1` and `male_3` here) are voices
  Pocket TTS Turkish was trained on, which favours it on speaker similarity.
- Trendyol-TTS does not follow a reference voice (similarity 0.68), so it is not counted as cloning.

## Limitations

- Turkish only.
- The first consonant of a sentence is occasionally clipped.
- Text is prepared by rules. Unusual formats, such as codes that mix letters and digits, may be read
  in an unexpected way; check them with `prepare()`.
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
