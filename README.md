# Achoose

A private personal assistant that runs on your own computer, with its own local brain.

This is the first step: a text chat in the terminal that can use a few simple tools,
and a benchmark to choose the best small model for your laptop.

## The safety rule

* **Reading is automatic.** Achoose may look at the time, list folders and read text files
  inside the folders you allow (your home folder by default).
* **Harmless actions are automatic.** Opening an app, a file, a folder or a website, locking the
  screen and sleep happen when you ask.
* **Changing things always asks you first.** Saving, sending, deleting, shutting down, restarting,
  installing or running anything waits for you to type `y` (or say "yes" in voice mode). If you press
  Enter or anything else, nothing happens.

## How it is built

```
achoose/
  brain/        the "brain" interface; any model plugs in here
    base.py       what every brain must do (chat, ask for tools)
    ollama.py     the local brain, a model running in Ollama
  tools/        things Achoose can do
    base.py       what a tool is, and its risk level (READ or CHANGE)
    builtin.py    time, list folder, read file, save note
    computer.py   open/close apps, open files/websites, move to Trash, lock, sleep, shut down, restart
    memory_tool.py  search_my_files (search inside indexed folders)
  memory/       private search index over folders you choose
    extract.py    text out of txt/code/html/pdf/docx/pptx, cut into pieces
    store.py      SQLite file with a word index + meaning fingerprints
    embed.py      fingerprints from a small Ollama model (embeddinggemma)
    indexer.py    reads new/changed files, forgets deleted ones
  voice/        voice mode: wake word, speech-to-text, Piper voice
    text.py       sentence splitting, cleaning text for speech, hearing "yes"
    engine.py     microphone, ears (faster-whisper), mouth (Piper), wake word
    loop.py       wake word -> listen -> think -> speak
    check.py      --voice-check: tests each part of voice mode
  agent.py      the thinking loop and the confirmation gate
  config.py     settings (see config.example.toml)
  cli.py        the text chat in your terminal
scripts/
  benchmark.py  compares models on your computer
training/
  train_wake_word.ipynb  trains "Hey Achoose" on Google Colab
tests/          checks that run without Ollama
```

The brain is swappable: everything talks to the `Brain` interface, so a cloud model
(used only when you ask, for hard jobs) can be added later without changing anything else.
Nothing here uses the internet, and there are no extra Python packages to install.

## Setup on Ubuntu

```bash
# 1. Install Ollama (runs local models)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Get the code
sudo apt install -y git python3
git clone https://github.com/gabdulaziz/a.c.h.o.o.s.e.git ~/achoose
cd ~/achoose

# 3. Download Achoose's brain (about 2.5 GB)
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 4. Chat
python3 -m achoose
```

## Choosing the model

`qwen3:4b-instruct-2507-q4_K_M` won our tests on an 8th-gen Intel laptop without a graphics card:
8/8 tool tests at about 4.4 s per request, 3 GB of RAM. To compare it with other models later:

```bash
ollama pull llama3.2:3b
python3 scripts/benchmark.py qwen3:4b-instruct-2507-q4_K_M llama3.2:3b
```

To use a different model, either run `ACHOOSE_MODEL=llama3.2:3b python3 -m achoose`, or copy
`config.example.toml` to `~/.config/achoose/config.toml` and change `model`.

## Memory setup (search inside your files)

Achoose can build a private search index of folders you choose (`~/Documents` by default)
and answer questions from them. Everything stays on your computer, in `~/Achoose/memory.db`.

```bash
cd ~/achoose
sudo apt install -y python3-venv          # lets Python make private package folders
python3 -m venv .venv                     # make one inside ~/achoose
.venv/bin/pip install -r requirements.txt # numpy + PDF/Word/PowerPoint readers, only in .venv
ollama pull embeddinggemma                # small model for "meaning fingerprints"
.venv/bin/python -m achoose               # start Achoose, then type /index
```

From then on start Achoose with `.venv/bin/python -m achoose` (from `~/achoose`).
To index other folders, set `memory_folders` in `~/.config/achoose/config.toml`.

## Voice setup (talk to Achoose)

Achoose can listen for **"Hey Jarvis"**, understand what you say and answer out loud in a calm
British voice. Everything runs on your computer; nothing is recorded or sent anywhere.

```bash
cd ~/achoose
sudo apt install -y libportaudio2                       # lets Python use the microphone and speakers
.venv/bin/pip install -r requirements-voice.txt         # speech-to-text, voice, wake word engine
.venv/bin/pip install --no-deps openwakeword==0.6.0     # wake word ("Hey Jarvis")
.venv/bin/python -m piper.download_voices en_GB-alan-medium --data-dir ~/Achoose/voices
.venv/bin/python -m achoose --voice                     # or --push-to-talk (press Enter, then speak)
```

The first start also downloads the speech-to-text model (about 145 MB) and the wake word files.
Say **"goodbye"** to leave voice mode. Before saving or changing anything, Achoose asks out loud;
only a clear "yes" counts. Voice, speed, wake word and "sir" can be changed in the settings file.

If voice mode does not react, run the voice check. It tests the speaker, the microphone level,
speech understanding and the wake word one by one, and says which part fails and what to change
(for example `input_device` to pick another microphone, or a lower `wake_threshold`):

```bash
.venv/bin/python -m achoose --voice-check
```

## Your own wake word ("Hey Achoose")

Achoose can wake up on its own name. The model is trained once on Google Colab (free, about 1 to 1.5 hours,
no recording of your voice needed) with the notebook [`training/train_wake_word.ipynb`](training/train_wake_word.ipynb).
It explains every step. At the end you download `hey_achoose.onnx` and put it here:

```bash
mkdir -p ~/Achoose/wake_words && mv ~/Downloads/hey_achoose.onnx ~/Achoose/wake_words/
```

Every `.onnx` file in `~/Achoose/wake_words/` is used automatically, and "Hey Jarvis" keeps working as a backup.
Run `--voice-check` to see the scores for each wake word.

## Controlling the computer

Things you can ask (typed or spoken):

| Ask | What happens | Asks first? |
|---|---|---|
| "Open VS Code", "open Firefox" | Starts the app from your app menu's list | no |
| "Open a new terminal window" | Uses the app's own "New Window" | no |
| "Open my Downloads folder", "open the Alparslan folder", "open youtube.com" | Opens it like a double-click / in your browser (a folder name is enough) | no |
| "How many videos are in the Alparslan folder?" | Lists/counts what is inside (videos, pictures, music, documents, or a type like pdf) | no |
| "Close VS Code", "close my Downloads folder" | Asks the app to quit (all its windows; folders are Files windows) | **yes** |
| "Delete Alparslan", "delete ~/Downloads/old.zip" | Finds it by name in your home folder (close spelling is fine; asks which one if there are several) and moves it to the **Trash** (restore it from the Files app) | **yes** |
| "Shut down" / "Restart" | Turns off / restarts after 5 seconds. Save your work first | **yes** |
| "Lock my screen" / "Go to sleep" | Locks the screen / suspends | no |

Achoose never runs a command the brain made up: apps are started only from the installed app list,
files only go to the Trash (never deleted for good), and only inside your home folder. Your main folders
(Documents, Downloads, ...), hidden settings folders and Achoose's own folders are refused.

### When Achoose mishears you

* Voice mode tells the speech model the names of your apps and folders (setting `stt_hints`), so an
  accent or a slip is more likely to come out as a real name.
* If a name is close but not exact ("Teratra"), Achoose asks **"Did you mean Terminal, sir?"** instead of
  giving up. Say yes and it carries on; deleting still asks "Shall I?" afterwards.

### Compare speech models on your voice

```bash
.venv/bin/python -m achoose --hear-test en --hear-model base.en    # the default: fast
.venv/bin/python -m achoose --hear-test en --hear-model small.en   # hears accents better, ~2-3x slower
```
If small.en is clearly better for you, put `stt_model = "small.en"` in `~/.config/achoose/config.toml`.

### Other languages (test only)

```bash
.venv/bin/python -m achoose --hear-test ru     # Russian; tg = Tajik, auto = it guesses
.venv/bin/python -m achoose --hear-test tg --hear-model small   # bigger, slower, better
```
Shows what a multilingual speech model heard and how long it took. Nothing in your settings changes.

## Chat commands

`/help`, `/tools`, `/stats` (speed of the last answer), `/reset` (forget the chat),
`/index` (read new or changed files in the memory folders), `/quit`.

## Running the tests

```bash
.venv/bin/python -m unittest -v   # memory tests are skipped if the packages are missing
```
