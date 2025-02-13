# Katheryne - Genshin Impact AI Assistant

A Node.js and Python-powered AI assistant that knows everything about Teyvat! Need to optimize your team? Do you know which artifacts to slap on your Hu Tao? Katheryne has your back! 🌏✨

> "Ad Astra Abyssosque! Welcome to the Adventurers' Guild."

---

## 🎮 Features

- 📜 **Comprehensive game data collection** – because knowledge is power!
- 🧠 **Smart training models** – so Katheryne doesn’t just talk nonsense.
- 🔥 **Build recommendations** – min-max your way to victory!
- ⚔️ **Weapon and artifact suggestions** – never be underpowered again.
- 🏰 **Domain & Abyss strategies** – because floors 11 and 12 are pain.
- 🤝 **Team synergy insights** – let’s make your party *cracked*.

---

## 🏗️ Project Structure

```plaintext
Katheryne/
├── node/                # TypeScript/Node.js components
│   ├── src/            # Core logic
│   │   ├── client.ts   # API client
│   │   └── types.ts    # TypeScript types
│   ├── dist/           # Compiled JS
│   ├── package.json    # Dependencies
│   └── tsconfig.json   # TypeScript config
├── python/             # AI/ML components
│   ├── main.py        # Main entry point
│   ├── check_and_train.py  # Training pipeline
│   ├── generate_training_data.py  # Data generation
│   ├── train_assistant.py  # Assistant training
│   ├── models/        # ML model implementations
│   │   ├── genshin_assistant.py
│   │   └── genshin_model.py
│   ├── preprocessing/ # Data processing utilities
│   ├── scraper/      # Data collection tools
│   └── requirements.txt # Python dependencies
├── data/               # All the juicy game data
│   ├── raw/          # Raw JSON data files
│   ├── processed/    # Processed data files
│   └── models/       # Trained model files
└── docs/              # Documentation and guides
```

---

## 🚀 Quick Start

### 🔧 Prerequisites

- 🐍 Python 3.8+
- 🟢 Node.js 14+
- 🏴‍☠️ Git

### 💾 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Katheryne.git
   cd Katheryne
   ```

2. Install Node.js dependencies:
   ```bash
   cd node
   npm install
   npm run build
   cd ..
   ```

3. Install Python dependencies:
   ```bash
   cd python
   pip install -r requirements.txt
   cd ..
   ```

### 🛠️ Testing the Setup

1. Generate training data:
   ```bash
   python python/data/generate_training_data.py
   ```

   📜 **This creates:**
   - `training_data/training_data.json` – Training samples
   - `training_data/dataset_summary.json` – Dataset overview

2. Train the assistant:
   ```bash
   # Using the batch script (Windows)
   train.bat

   # Or directly with Python
   python python/train_assistant.py
   ```

   🔍 **During training you'll see:**
   - Training progress
   - Loss metrics
   - Model checkpoints

3. Verify data integrity:
   ```bash
   python python/verify_data.py
   ```

   🔍 **Checks:**
   - Data formatting ✅
   - Sample distribution ✅
   - Example queries & responses ✅

---

## 🏋️‍♂️ Building Your Own AI Model

### 1️⃣ Data Preparation

1. Edit `data/*.json` to tweak game info.
2. Modify `python/generate_training_data.py` to support new query types.
3. Run the generator:
   ```bash
   python python/generate_training_data.py
   ```

### 2️⃣ Training Time! 🧠

Basic:
```bash
python python/train.py
```

Advanced training:
```bash
python python/train.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model-type transformer \
  --save-dir models/custom
```

📌 **Training parameters:**
- `epochs`: How long Katheryne studies 📖 (default: 5)
- `batch-size`: How much info per session (default: 32)
- `learning-rate`: Brain expansion speed 🧠 (default: 0.001)
- `model-type`: Transformer or LSTM?
- `save-dir`: Where to store trained models

### 3️⃣ Evaluating Performance 🏆

Test your trained AI:
```bash
python python/evaluate.py --model-path models/custom
```

📊 **Outputs:**
- Accuracy results 📈
- Sample answers ✍️
- Error analysis 🔍

### 4️⃣ Deploying Katheryne!

Run the assistant:
```bash
python python/assistant.py --model-path models/custom
```

Try out some queries:
> "Tell me about Ganyu."
> "What's the best build for Hu Tao?"
> "Recommend a team comp for Abyss!"

---

## 🔮 Roadmap

✨ What’s next for Katheryne?

- 🌎 **Multi-language support**
- 📰 **Real-time updates with game patches**
- 🌐 **Web-based UI for easy model training**
- 🤖 **Discord bot integration**
- 👥 **Community contributions for dataset expansion**
- 📊 **Advanced team composition analysis**
- 📱 **Mobile app version**

---

## 🎁 Contributing

Want to help? **We welcome all travelers!** 🚀

Check out our **[Contributing Guide](docs/contributing.md)** to:
- Add new features 🛠️
- Improve training models 🧠
- Expand data sources 📜
- Report bugs 🐞

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and distribute!

---

## 💌 Contact & Support

📧 Email: coff33ninja69@gmail.com  
💬 Discord: DRAGOHN#1282  
🐛 Issues? [Open a ticket](https://github.com/yourusername/Katheryne/issues)

---

**Ad Astra Abyssosque, Traveler! May the RNG gods be ever in your favor!** 🎲✨

