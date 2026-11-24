# Achoose

A private personal assistant that runs on your own computer, with its own local brain.

This is the first step: a text chat in the terminal that can use a few simple tools,
and a benchmark to choose the best small model for your laptop.

## The safety rule

* **Reading is automatic.** Achoose may look at the time, list folders and read text files
  inside the folders you allow (your home folder by default).
* **Changing things always asks you first.** Saving, sending, deleting, installing or running
  anything waits for you to type `y`. If you press Enter or anything else, nothing happens.

## How it is built

```
achoose/
  brain/        the "brain" interface; any model plugs in here
    base.py       what every brain must do (chat, ask for tools)
    ollama.py     the local brain, a model running in Ollama
  tools/        things Achoose can do
    base.py       what a tool is, and its risk level (READ or CHANGE)
    builtin.py    time, list folder, read file, save note
    memory_tool.py  search_my_files (search inside indexed folders)
  memory/       private search index over folders you choose
    extract.py    text out of txt/code/html/pdf/docx/pptx, cut into pieces
    store.py      SQLite file with a word index + meaning fingerprints
    embed.py      fingerprints from a small Ollama model (embeddinggemma)
    indexer.py    reads new/changed files, forgets deleted ones
  agent.py      the thinking loop and the confirmation gate
  config.py     settings (see config.example.toml)
  cli.py        the text chat in your terminal
scripts/
  benchmark.py  compares models on your computer
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

## Chat commands

`/help`, `/tools`, `/stats` (speed of the last answer), `/reset` (forget the chat),
`/index` (read new or changed files in the memory folders), `/quit`.

## Running the tests

```bash
.venv/bin/python -m unittest -v   # memory tests are skipped if the packages are missing
```
