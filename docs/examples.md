# Examples

## Table of Contents
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

## Basic Usage

### 1. Character Information

```python
from katheryne import GenshinAssistant

# Initialize assistant
assistant = GenshinAssistant()

# Get character information
response = assistant.generate_response(
    "Tell me about Hu Tao"
)
print(response)

# Output:
"""
Hu Tao is the 77th Director of the Wangsheng Funeral Parlor in Liyue. She is a
5-star Pyro Polearm user known for her high single-target damage and unique HP
management playstyle. Her Elemental Skill increases her ATK based on her max HP
while consuming HP, and her Elemental Burst deals massive Pyro DMG while healing
her based on the number of enemies hit.
"""
```

### 2. Weapon Information

```python
# Get weapon details
response = assistant.generate_response(
    "What is Staff of Homa good for?"
)
print(response)

# Output:
"""
Staff of Homa is a 5-star polearm that's particularly strong on HP-scaling
characters like Hu Tao and Zhongli. It provides CRIT DMG as a secondary stat
and increases HP%. Its passive also provides an ATK bonus based on the wielder's
max HP, with an increased bonus when below 50% HP.
"""
```

### 3. Team Building

```python
# Get team suggestions
response = assistant.generate_response(
    "Suggest a team for Hu Tao"
)
print(response)

# Output:
"""
A strong Hu Tao team composition:
1. Hu Tao (Main DPS)
2. Xingqiu (Sub DPS/Hydro Application)
3. Zhongli (Shield Support)
4. Yelan/Albedo (Sub DPS)

This team provides:
- Consistent vaporize reactions
- Strong shield protection
- Additional damage support
- Good energy generation
"""
```

### 4. Build Optimization

```python
# Get build recommendations
response = assistant.generate_response(
    "Best artifacts for Hu Tao?"
)
print(response)

# Output:
"""
Recommended artifact sets for Hu Tao:

1. 4-piece Crimson Witch of Flames
   - Best for vaporize reactions
   - Increases Pyro DMG and reaction damage

2. 4-piece Shimenawa's Reminiscence
   - Alternative option
   - Provides strong charged attack damage

Main stats:
- Sands: HP%
- Goblet: Pyro DMG
- Circlet: Crit Rate/DMG

Substats priority:
1. Crit Rate/DMG
2. HP%
3. EM
4. ATK%
"""
```

## Advanced Features

### 1. Custom Queries

```python
# Complex comparison
response = assistant.generate_response(
    "Compare Hu Tao and Xiangling for Pyro DPS"
)
print(response)

# Output:
"""
Hu Tao vs Xiangling comparison:

Hu Tao:
+ Higher single-target damage
+ More controlled damage windows
+ Strong charged attacks
- HP management required
- Limited AoE
- Stamina dependent

Xiangling:
+ Excellent AoE damage
+ Off-field DPS
+ F2P friendly
+ More flexible teams
- Energy dependent
- Requires Bennett
"""
```

### 2. Team Analysis

```python
# Analyze team synergy
response = assistant.generate_response(
    "Analyze team: Raiden Yelan Xiangling Bennett"
)
print(response)

# Output:
"""
Team Analysis - Raiden National variant:

Core mechanics:
1. Bennett provides ATK buff and healing
2. Xiangling's Pyronado for off-field Pyro
3. Yelan's Burst for off-field Hydro
4. Raiden drives reactions and recharges energy

Reactions:
- Vaporize (Xiangling + Yelan)
- Overvape (Pyro + Hydro + Electro)

Rotation:
1. Bennett Burst → Skill
2. Xiangling Burst → Skill
3. Yelan Skill → Burst
4. Raiden Burst combo
"""
```

### 3. Meta Analysis

```python
# Get meta insights
response = assistant.generate_response(
    "Current meta status of Kazuha"
)
print(response)

# Output:
"""
Kazuha's Meta Status (Current Version):

Tier: SS
Role: Universal Support

Strengths:
- Universal elemental DMG buff
- Strong CC capabilities
- Easy to build
- Fits in many teams

Meta Teams:
1. International (Childe/Xiangling)
2. Raiden Hypercarry
3. Freeze (Ayaka/Ganyu)
4. Mono Element teams

Future Outlook:
- Likely to remain meta
- Valuable for future elemental DPS
- Strong exploration utility
"""
```

### 4. Build Comparison

```python
# Compare different builds
response = assistant.generate_response(
    "Compare Raiden builds: EL vs Catch"
)
print(response)

# Output:
"""
Raiden Build Comparison:

Engulfing Lightning (EL):
+ Higher base ATK
+ Built-in ER scaling
+ Better stat distribution
- 5-star weapon cost
- Opportunity cost

The Catch R5:
+ F2P option
+ Strong burst damage
+ Crit rate bonus
+ Good ER substat
- Slightly lower damage ceiling
- Fishing required

Performance Difference:
- EL ~15-20% more damage
- Catch R5 ~85-90% of EL performance

Recommendation:
- Catch R5 is excellent cost-effective option
- EL worth only for optimization
"""
```

## Integration Examples

### 1. Discord Bot Integration

```python
import discord
from katheryne import GenshinAssistant

class KatheryneBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.assistant = GenshinAssistant()
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith('!genshin'):
            query = message.content[8:].strip()
            response = self.assistant.generate_response(query)
            await message.channel.send(response)

# Usage
bot = KatheryneBot()
bot.run('YOUR_DISCORD_TOKEN')
```

### 2. Web API Integration

```python
from fastapi import FastAPI
from katheryne import GenshinAssistant

app = FastAPI()
assistant = GenshinAssistant()

@app.post("/api/query")
async def query_assistant(request: dict):
    query = request.get("query", "")
    try:
        response = assistant.generate_response(query)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Batch Processing

```python
from typing import List
from concurrent.futures import ThreadPoolExecutor

def process_queries(
    assistant: GenshinAssistant,
    queries: List[str]
) -> List[dict]:
    def process_single(query: str) -> dict:
        try:
            response = assistant.generate_response(query)
            return {"query": query, "response": response, "status": "success"}
        except Exception as e:
            return {"query": query, "error": str(e), "status": "error"}
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single, queries))
    
    return results

# Usage
queries = [
    "Best Hu Tao team?",
    "Raiden artifacts?",
    "Kazuha build?"
]

assistant = GenshinAssistant()
results = process_queries(assistant, queries)
for result in results:
    print(f"\nQuery: {result['query']}")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Response: {result['response']}")
```

### 4. Command Line Interface

```python
import argparse
from katheryne import GenshinAssistant

def main():
    parser = argparse.ArgumentParser(description='Katheryne CLI')
    parser.add_argument('query', help='Query for the assistant')
    parser.add_argument('--max-length', type=int, default=64,
                       help='Maximum response length')
    
    args = parser.parse_args()
    assistant = GenshinAssistant()
    
    try:
        response = assistant.generate_response(
            args.query,
            max_length=args.max_length
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

# Usage: python cli.py "Tell me about Hu Tao" --max-length 128
```

## Best Practices

### 1. Error Handling

```python
from katheryne import GenshinAssistant, AssistantError

def safe_query(query: str) -> str:
    try:
        assistant = GenshinAssistant()
        response = assistant.generate_response(query)
        return response
    except AssistantError as e:
        logger.error(f"Assistant error: {e}")
        return "Sorry, I couldn't process that query."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred."
```

### 2. Caching

```python
from functools import lru_cache
from katheryne import GenshinAssistant

class CachedAssistant:
    def __init__(self):
        self.assistant = GenshinAssistant()
    
    @lru_cache(maxsize=1000)
    def query(self, query: str) -> str:
        return self.assistant.generate_response(query)

# Usage
cached_assistant = CachedAssistant()
response = cached_assistant.query("Tell me about Hu Tao")
```

### 3. Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls: int, period: float):
    def decorator(func):
        last_reset = time.time()
        calls_made = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            
            current_time = time.time()
            if current_time - last_reset > period:
                calls_made = 0
                last_reset = current_time
            
            if calls_made >= calls:
                raise Exception("Rate limit exceeded")
            
            calls_made += 1
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Usage
@rate_limit(calls=60, period=60.0)  # 60 calls per minute
def query_assistant(query: str) -> str:
    assistant = GenshinAssistant()
    return assistant.generate_response(query)
```

### 4. Logging

```python
import logging
from katheryne import GenshinAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='katheryne.log'
)

logger = logging.getLogger('katheryne')

class LoggedAssistant:
    def __init__(self):
        self.assistant = GenshinAssistant()
    
    def query(self, query: str) -> str:
        logger.info(f"Received query: {query}")
        try:
            response = self.assistant.generate_response(query)
            logger.info(f"Generated response: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

# Usage
assistant = LoggedAssistant()
try:
    response = assistant.query("Tell me about Hu Tao")
    print(response)
except Exception as e:
    print(f"Error: {e}")
```

These examples demonstrate various ways to use and integrate the Katheryne assistant. For more detailed information, check the [API Reference](APIReference.md) and [Developer Guide](DeveloperGuide.md).