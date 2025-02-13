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
        
        # Advanced build queries with constellation and weapon details
        if 'constellations' in char_data:
            for const_level in [1, 2, 3, 4, 5, 6]:
                queries.append({
                    "query": f"For {char_name} at constellation {const_level}, what are the recommended boss drops and artifacts?",
                    "response": {
                        "constellation_effects": char_data['constellations'].get(str(const_level), "No data"),
                        "recommended_drops": char_data.get('boss_drops', ["Various boss materials"]),
                        "artifact_stats": {
                            "main_stats": char_data.get('recommended_main_stats', {
                                "sands": "ATK%",
                                "goblet": f"{char_data.get('element', '')} DMG Bonus",
                                "circlet": "CRIT Rate/DMG"
                            }),
                            "substats": char_data.get('recommended_substats', ["CRIT Rate", "CRIT DMG", "ATK%"])
                        },
                        "weapon_recommendations": {
                            "signature": char_data.get('signature_weapon', "No specific signature weapon"),
                            "alternatives": char_data.get('alternative_weapons', ["Various 4-star options"])
                        }
                    },
                    "type": "advanced_build"
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
            
        # Advanced weapon queries
        queries.append({
            "query": f"What are the best artifact stats to pair with {weapon_name}?",
            "response": {
                "recommended_stats": weapon_data.get('recommended_stats', {
                    "main_stats": ["ATK%", "Element DMG", "CRIT"],
                    "substats": ["CRIT Rate", "CRIT DMG", "ATK%"]
                }),
                "synergy_explanation": f"These stats complement {weapon_name}'s {weapon_data.get('passive_description', 'passive ability')}",
                "character_synergy": weapon_data.get('character_synergy', "Works well with various characters")
            },
            "type": "advanced_weapon"
        })

    return queries

def generate_domain_queries(domains: dict) -> list:
    """Generate queries about domains and their schedules."""
    queries = []
    
    for domain_name, domain_data in domains.items():
        # Domain schedule query
        queries.append({
            "query": f"What's the schedule for {domain_name} domain?",
            "response": {
                "schedule": domain_data.get('schedule', "Available daily"),
                "drop_rates": domain_data.get('drop_rates', {}),
                "recommended_characters": domain_data.get('recommended_characters', []),
                "farming_strategy": domain_data.get('farming_strategy', "No specific strategy available")
            },
            "type": "domain_schedule"
        })
        
        # Domain farming strategy
        queries.append({
            "query": f"What's the best strategy for farming {domain_name}?",
            "response": {
                "team_composition": domain_data.get('recommended_team', "Flexible team composition"),
                "character_elements": domain_data.get('recommended_elements', []),
                "rotation_schedule": domain_data.get('rotation', "Available at all times"),
                "efficiency_tips": domain_data.get('efficiency_tips', ["Complete higher difficulty for better rewards"])
            },
            "type": "domain_strategy"
        })

    return queries

def generate_team_composition_queries(characters: dict) -> list:
    """Generate team composition queries."""
    queries = []
    elements = ["Pyro", "Hydro", "Electro", "Cryo", "Anemo", "Geo", "Dendro"]
    
    # Generate queries for different team compositions
    for element_focus in elements:
        char_list = [name for name, data in characters.items() 
                    if data.get('element') == element_focus][:3]  # Get up to 3 characters of the focus element
        
        if char_list:
            queries.append({
                "query": f"Help me build a {element_focus}-focused team with {', '.join(char_list)}. What other characters, weapons, and artifacts would work well?",
                "response": {
                    "team_analysis": {
                        "core_characters": char_list,
                        "recommended_supports": [name for name, data in characters.items() 
                                              if data.get('element') != element_focus][:2],
                        "elemental_resonance": f"{element_focus} Resonance benefits"
                    },
                    "build_recommendations": {
                        char: {
                            "role": characters[char].get('recommended_role', "Support/Sub-DPS"),
                            "weapons": characters[char].get('recommended_weapons', ["Various options"]),
                            "artifacts": characters[char].get('recommended_artifacts', ["General damage set"])
                        } for char in char_list
                    },
                    "rotation_strategy": "Apply elemental skills in sequence for maximum reaction damage"
                },
                "type": "team_composition"
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
        
    try:
        domains = load_json_file(data_dir / "domains.json")
        print(f"Loaded {len(domains)} domains")
    except Exception as e:
        print(f"Error loading domains: {e}")
        domains = {}
    
    # Generate queries
    all_queries = []
    
    if characters:
        char_queries = generate_character_queries(characters)
        print(f"Generated {len(char_queries)} character queries")
        all_queries.extend(char_queries)
        
        comp_queries = generate_comparison_queries(characters)
        print(f"Generated {len(comp_queries)} comparison queries")
        all_queries.extend(comp_queries)
        
        team_queries = generate_team_composition_queries(characters)
        print(f"Generated {len(team_queries)} team composition queries")
        all_queries.extend(team_queries)
    
    if weapons:
        weapon_queries = generate_weapon_queries(weapons)
        print(f"Generated {len(weapon_queries)} weapon queries")
        all_queries.extend(weapon_queries)
        
    if domains:
        domain_queries = generate_domain_queries(domains)
        print(f"Generated {len(domain_queries)} domain queries")
        all_queries.extend(domain_queries)
    
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
        "query_types": {
            "character_info": len([q for q in all_queries if q["type"] == "character_info"]),
            "weapon_info": len([q for q in all_queries if q["type"] == "weapon_info"]),
            "comparison": len([q for q in all_queries if q["type"] == "comparison"]),
            "advanced_build": len([q for q in all_queries if q["type"] == "advanced_build"]),
            "advanced_weapon": len([q for q in all_queries if q["type"] == "advanced_weapon"]),
            "domain_schedule": len([q for q in all_queries if q["type"] == "domain_schedule"]),
            "domain_strategy": len([q for q in all_queries if q["type"] == "domain_strategy"]),
            "team_composition": len([q for q in all_queries if q["type"] == "team_composition"])
        }
    }
    
    summary_path = training_dir / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset summary to {summary_path}")
    
    print(f"\nGenerated {len(all_queries)} total training samples:")
    for type_name, count in summary["query_types"].items():
        print(f"- {type_name}: {count} samples")

if __name__ == "__main__":
    main()