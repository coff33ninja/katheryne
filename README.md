# Katheryne - Genshin Impact AI Assistant

A Node.js and Python-powered AI assistant that knows everything about Teyvat! Need to optimize your team? Do you know which artifacts to slap on your Hu Tao? Katheryne has your back! 🌏✨

> "Ad Astra Abyssosque! Welcome to the Adventurers' Guild."

---

## 🎮 Features

- 📜 **Comprehensive game data collection** – because knowledge is power!
- 🧠 **Smart training models** – so Katheryne doesn't just talk nonsense.
- 🔥 **Build recommendations** – min-max your way to victory!
- ⚔️ **Weapon and artifact suggestions** – never be underpowered again.
- 🏰 **Domain & Abyss strategies** – because floors 11 and 12 are pain.
- 🤝 **Team synergy insights** – let's make your party *cracked*.

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

- Python 3.8+
- Node.js 14+
- Git
- CUDA-capable GPU (optional, for faster training)

### 💾 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/coff33ninja/katheryne.git
   cd katheryne
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

### 🛠️ Setup and Training

1. Generate training data:
   ```bash
   python python/generate_training_data.py
   ```
   This creates:
   - `training_data/training_data.json` – Training samples
   - `training_data/dataset_summary.json` – Dataset overview

2. Train the AI assistant:
   ```bash
   # For Windows users:
   train.bat

   # Or directly with Python:
   python python/train_assistant.py
   ```
   During training you'll see:
   - Training progress with loss metrics
   - Model checkpoints being saved
   - Early stopping information when needed

3. Train data embeddings (optional):
   ```bash
   python python/check_and_train.py
   ```
   This trains autoencoders for:
   - Character data
   - Artifact data
   - Weapon data

### 🤖 Using the Assistant

1. Interactive Mode:
   ```bash
   python python/main.py
   ```
   This starts a chat interface where you can ask questions like:
   - "What are Hu Tao's best artifacts?"
   - "Suggest a team for Abyss Floor 12"
   - "Is Staff of Homa good for Xiao?"

2. Using in Your Code:
   ```python
   from pathlib import Path
   from python.models.genshin_assistant import GenshinAssistant, GenshinAssistantDataset
   import torch

   # Initialize dataset to get vocabulary
   dataset = GenshinAssistantDataset(data_path=Path("data"))
   vocab_size = dataset.vocab_size

   # Initialize the assistant
   assistant = GenshinAssistant(
       vocab_size=vocab_size,
       embedding_dim=128,
       hidden_dim=256
   )

   # Load trained model
   checkpoint = torch.load("data/models/assistant_best.pt")
   assistant.load_state_dict(checkpoint['model_state_dict'])
   assistant.eval()

   # Generate a response
   response = assistant.generate_response(
       query="Tell me about Ganyu's abilities",
       max_length=64
   )
   print("AI Assistant Response:", response)
   ```

3. Using Embeddings:
   ```python
   from python.models.genshin_model import GenshinAutoencoder
   import torch

   # Load trained autoencoder
   model_path = "data/models/characters_autoencoder.pt"
   model = GenshinAutoencoder(input_dim=feature_dim)
   model.load_state_dict(torch.load(model_path))

   # Generate embeddings
   with torch.no_grad():
       embeddings = model.encode(features)
   ```

---

## 🔮 Roadmap

✨ What's next for Katheryne?

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

Check out our **[Contributing Guide](docs/Contributing.md)** to:
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
