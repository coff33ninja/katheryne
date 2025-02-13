3. Install Python dependencies:
   ```bash
   cd python
   pip install -r requirements.txt
   cd ..
   ```

### 🛠️ Testing the Setup

1. Generate training data:
   ```bash
   python python/generate_training_data.py
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
# Katheryne - Genshin Impact AI Assistant

A Node.js and Python-powered AI assistant that knows everything about Teyvat! Need to optimize your team? Wondering which artifacts to slap on your Hu Tao? Katheryne has your back! 🌏✨

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
