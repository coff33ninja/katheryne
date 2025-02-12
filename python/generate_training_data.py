import json
from pathlib import Path
import random

def load_json_file(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            # Convert list to dict using name as key
            return {item['name']: item for item in data}
        return data

def generate_character_queries(characters: dict) -> list:
    """Generate queries about characters."""
    queries = []
    
    for char_name, char_data in characters.items():
        # Basic info query
        queries.append({
            "query": f"Tell me about {char_name}",
            "response": f"{char_name} is a {char_data.get('rarity', '?')}★ {char_data.get('element', '')} character who uses a {char_data.get('weapon', '')}. {char_data.get('description', '')}",
            "type": "character_info"
        })
        
        # Abilities query
        if 'abilities' in char_data:
            abilities_desc = []
            for ability in char_data['abilities']:
                abilities_desc.append(f"{ability['name']}: {ability['description']}")
            
            queries.append({
                "query": f"What are {char_name}'s abilities?",
                "response": f"{char_name}'s abilities include: " + ". ".join(abilities_desc),
                "type": "character_info"
            })
        
        # Build recommendations
        queries.append({
            "query": f"What's the best build for {char_name}?",
            "response": f"For {char_name}, recommended weapons include {', '.join(char_data.get('recommended_weapons', ['various options']))}. "
                       f"Artifacts should focus on {', '.join(char_data.get('recommended_stats', ['ATK', 'CRIT']))}.",
            "type": "character_info"
        })

    return queries

def generate_weapon_queries(weapons: dict) -> list:
    """Generate queries about weapons."""
    queries = []
    
    for weapon_name, weapon_data in weapons.items():
        # Basic info query
        queries.append({
            "query": f"Tell me about {weapon_name}",
            "response": f"{weapon_name} is a {weapon_data.get('rarity', '?')}★ {weapon_data.get('type', '')} with base ATK {weapon_data.get('baseAttack', '?')}. {weapon_data.get('description', '')}",
            "type": "weapon_info"
        })
        
        # Best characters query
        if 'recommended_characters' in weapon_data:
            queries.append({
                "query": f"Who can use {weapon_name}?",
                "response": f"{weapon_name} is best used on: {', '.join(weapon_data['recommended_characters'])}",
                "type": "weapon_info"
            })

    return queries

def generate_comparison_queries(characters: dict) -> list:
    """Generate character comparison queries."""
    queries = []
    char_names = list(characters.keys())
    
    for _ in range(50):  # Generate 50 comparison queries
        char1, char2 = random.sample(char_names, 2)
        char1_data = characters[char1]
        char2_data = characters[char2]
        
        queries.append({
            "query": f"Compare {char1} and {char2}",
            "response": f"{char1} is a {char1_data.get('rarity', '?')}★ {char1_data.get('element', '')} {char1_data.get('weapon', '')}, "
                       f"while {char2} is a {char2_data.get('rarity', '?')}★ {char2_data.get('element', '')} {char2_data.get('weapon', '')}. "
                       f"They have different roles: {char1} {char1_data.get('role_description', '')}, "
                       f"whereas {char2} {char2_data.get('role_description', '')}.",
            "type": "comparison"
        })

    return queries

def main():
    data_dir = Path(__file__).parent.parent / "data"
    training_dir = data_dir.parent / "training_data"
    training_dir.mkdir(exist_ok=True)
    
    # Load data
    try:
        characters = load_json_file(data_dir / "characters_detailed.json")
        print(f"Loaded {len(characters)} characters")
    except Exception as e:
        print(f"Error loading characters: {e}")
        characters = {}
    
    try:
        weapons = load_json_file(data_dir / "weapons.json")
        print(f"Loaded {len(weapons)} weapons")
    except Exception as e:
        print(f"Error loading weapons: {e}")
        weapons = {}
    
    # Generate queries
    all_queries = []
    
    if characters:
        char_queries = generate_character_queries(characters)
        print(f"Generated {len(char_queries)} character queries")
        all_queries.extend(char_queries)
        
        comp_queries = generate_comparison_queries(characters)
        print(f"Generated {len(comp_queries)} comparison queries")
        all_queries.extend(comp_queries)
    
    if weapons:
        weapon_queries = generate_weapon_queries(weapons)
        print(f"Generated {len(weapon_queries)} weapon queries")
        all_queries.extend(weapon_queries)
    
    if not all_queries:
        print("No queries generated! Check data files.")
        return
    
    # Save training data
    training_data_path = training_dir / "training_data.json"
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    print(f"Saved training data to {training_data_path}")
    
    # Update dataset summary
    summary = {
        "total_samples": len(all_queries),
        "character_queries": len([q for q in all_queries if q["type"] == "character_info"]),
        "weapon_queries": len([q for q in all_queries if q["type"] == "weapon_info"]),
        "comparison_queries": len([q for q in all_queries if q["type"] == "comparison"]),
        "sample_types": {
            "character_info": len([q for q in all_queries if q["type"] == "character_info"]),
            "weapon_info": len([q for q in all_queries if q["type"] == "weapon_info"]),
            "comparison": len([q for q in all_queries if q["type"] == "comparison"])
        }
    }
    
    summary_path = training_dir / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset summary to {summary_path}")
    
    print(f"\nGenerated {len(all_queries)} total training samples:")
    for type_name, count in summary["sample_types"].items():
        print(f"- {type_name}: {count} samples")

if __name__ == "__main__":
    main()