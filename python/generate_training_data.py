import json
import os

def load_json(file_path: str) -> dict:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def save_json(file_path: str, data: dict) -> None:
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def load_data():
    """Load all data from JSON files."""
    characters = load_json('data/characters_detailed.json')
    weapons = load_json('data/weapons.json')
    domains = load_json('data/domains.json')
    artifacts = load_json('data/artifacts.json')
    team_synergies = load_json('data/team_synergies.json')
    return characters, weapons, domains, artifacts, team_synergies

def generate_character_queries(characters: dict) -> list:
    """Generate queries about characters."""
    queries = []
    
    for char_name, char_data in characters.items():
        # Basic character info
        queries.append({
            "query": f"Tell me about {char_name}",
            "response": {
                "name": char_name,
                "element": char_data.get('element'),
                "weapon": char_data.get('weapon'),
                "rarity": char_data.get('rarity'),
                "description": char_data.get('description')
            },
            "type": "character_info"
        })
        
        # Advanced build info
        queries.append({
            "query": f"What's the best build for {char_name}?",
            "response": {
                "recommended_role": char_data.get('recommended_role'),
                "weapons": char_data.get('recommended_weapons'),
                "artifacts": char_data.get('recommended_artifacts'),
                "main_stats": char_data.get('recommended_main_stats'),
                "substats": char_data.get('recommended_substats')
            },
            "type": "advanced_build"
        })
        
        # Constellation info
        if 'constellations' in char_data:
            queries.append({
                "query": f"What are {char_name}'s constellations?",
                "response": {
                    "constellations": char_data['constellations']
                },
                "type": "character_info"
            })
    
    return queries

def generate_weapon_queries(weapons: dict) -> list:
    """Generate queries about weapons."""
    queries = []
    
    for weapon_id, weapon_data in weapons.items():
        # Basic weapon info
        queries.append({
            "query": f"Tell me about {weapon_data['name']}",
            "response": {
                "name": weapon_data['name'],
                "type": weapon_data.get('type'),
                "rarity": weapon_data.get('rarity'),
                "baseAttack": weapon_data.get('baseAttack'),
                "description": weapon_data.get('description')
            },
            "type": "weapon_info"
        })
        
        # Advanced weapon info
        queries.append({
            "query": f"What characters can use {weapon_data['name']}?",
            "response": {
                "recommended_characters": weapon_data.get('recommended_characters', []),
                "character_synergy": weapon_data.get('character_synergy'),
                "recommended_stats": weapon_data.get('recommended_stats', {})
            },
            "type": "advanced_weapon"
        })
    
    return queries

def generate_domain_queries(domains: dict) -> list:
    """Generate queries about domains and their schedules."""
    queries = []
    difficulty_levels = ["I", "II", "III", "IV"]
    roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
    strategy_variations = [
        "Speed clear strategy",
        "F2P friendly approach",
        "Solo clear tactics",
        "Co-op strategy"
    ]
    
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
        
        # Strategy variations
        for strategy in strategy_variations:
            queries.append({
                "query": f"What's the best {strategy.lower()} for {domain_name}?",
                "response": {
                    "strategy_type": strategy,
                    "team_setup": {
                        "recommended": domain_data.get('recommended_characters', [])[:4],
                        "alternatives": "Flexible based on availability",
                        "minimum_requirements": "Depends on difficulty level"
                    },
                    "execution": {
                        "preparation": "Set up team buffs and shields",
                        "main_phase": "Focus on efficient clearing",
                        "adaptation": f"Adjust strategy based on {strategy.lower()} focus"
                    },
                    "tips": [
                        f"Optimize for {strategy.lower()}",
                        "Use appropriate elemental reactions",
                        "Manage energy efficiently"
                    ]
                },
                "type": "domain_strategy"
            })
        
        # Difficulty-specific strategies
        for level in difficulty_levels:
            level_num = roman_to_int[level]
            queries.append({
                "query": f"How to clear {domain_name} at difficulty level {level}?",
                "response": {
                    "difficulty_level": level,
                    "recommended_team": {
                        "main_dps": domain_data.get('recommended_characters', [])[:1],
                        "supports": domain_data.get('recommended_characters', [])[1:3],
                        "healer": domain_data.get('recommended_characters', [])[3:4]
                    },
                    "element_counters": domain_data.get('recommended_elements', []),
                    "strategy": f"Level {level} specific strategy: " + domain_data.get('farming_strategy', "Focus on efficient clearing"),
                    "minimum_requirements": {
                        "character_level": f"{40 + (level_num * 10)}+",
                        "talent_level": f"{1 + level_num}+",
                        "weapon_level": f"{40 + (level_num * 10)}+"
                    },
                    "mechanics": {
                        "key_challenges": f"Level {level} specific mechanics",
                        "counter_strategy": "Specific approach to domain mechanics",
                        "time_management": "Efficient rotation and skill usage"
                    }
                },
                "type": "domain_strategy"
            })
    
    return queries

def generate_artifact_queries(artifacts: dict) -> list:
    """Generate queries about artifacts and their stats."""
    queries = []
    
    for artifact_id, artifact_data in artifacts.items():
        # Basic artifact info
        queries.append({
            "query": f"What are the set bonuses for {artifact_data['name']}?",
            "response": {
                "set_name": artifact_data['name'],
                "bonuses": artifact_data['set_bonus'],
                "recommended_characters": artifact_data.get('recommended_characters', [])
            },
            "type": "artifact_info"
        })
        
        # Artifact stats recommendations
        queries.append({
            "query": f"What are the best stats for {artifact_data['name']}?",
            "response": {
                "recommended_stats": artifact_data['recommended_stats'],
                "character_recommendations": artifact_data.get('recommended_characters', []),
                "domain_location": artifact_data.get('domain', "Various sources")
            },
            "type": "artifact_stats"
        })
        
        # Farming guide
        queries.append({
            "query": f"How to farm {artifact_data['name']} efficiently?",
            "response": {
                "domain": artifact_data.get('domain', "Not available in domains"),
                "recommended_team": "Check domain requirements",
                "farming_tips": [
                    "Use condensed resin for efficiency",
                    "Focus on correct main stats first",
                    "Don't forget to use the artifact strongbox"
                ]
            },
            "type": "artifact_farming"
        })
    
    return queries

def generate_team_synergy_queries(team_synergies: dict) -> list:
    """Generate queries about team compositions and synergies."""
    queries = []
    
    for team_id, team_data in team_synergies.items():
        # Basic team comp info
        queries.append({
            "query": f"Tell me about the {team_data['name']} team composition",
            "response": {
                "team_name": team_data['name'],
                "core_members": team_data['core_characters'],
                "support_options": team_data['support_options'],
                "resonance": team_data['elemental_resonance'],
                "rotation": team_data['rotation'],
                "pros_cons": {
                    "advantages": team_data['advantages'],
                    "disadvantages": team_data['disadvantages']
                }
            },
            "type": "team_synergy"
        })
        
        # Rotation guide
        queries.append({
            "query": f"What's the rotation for {team_data['name']}?",
            "response": {
                "team_comp": team_data['core_characters'] + team_data['support_options'][:2],
                "rotation_details": team_data['rotation'],
                "tips": [
                    "Follow the rotation sequence carefully",
                    "Maintain buff uptime",
                    "Watch energy management"
                ]
            },
            "type": "team_rotation"
        })
        
        # Team building variations
        queries.append({
            "query": f"What are the possible variations for {team_data['name']}?",
            "response": {
                "core_requirements": team_data['core_characters'],
                "flexible_slots": team_data['support_options'],
                "resonance_options": team_data['elemental_resonance'],
                "build_focus": [
                    "Standard composition",
                    "F2P friendly version",
                    "Whale optimization"
                ]
            },
            "type": "team_building"
        })
    
    return queries

def generate_team_composition_queries(characters: dict) -> list:
    """Generate team composition queries."""
    queries = []
    elements = ["Pyro", "Hydro", "Electro", "Cryo", "Anemo", "Geo", "Dendro"]
    team_types = ["Main DPS", "Support", "Sub-DPS", "Healer"]
    team_focuses = ["Elemental Reactions", "Raw Damage", "Survivability", "Energy Generation"]
    
    # Generate queries for different team compositions
    for element_focus in elements:
        char_list = [name for name, data in characters.items() 
                    if data.get('element') == element_focus][:3]  # Get up to 3 characters of the focus element
        
        if char_list:
            # Element-focused team
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
            
            # Role-focused team
            for role in team_types:
                queries.append({
                    "query": f"Build a team around {char_list[0]} as a {role}. What characters and builds work best?",
                    "response": {
                        "team_analysis": {
                            "main_character": {
                                "name": char_list[0],
                                "role": role,
                                "build_focus": f"Optimize for {role} capabilities"
                            },
                            "recommended_supports": [name for name, data in characters.items() 
                                                  if name != char_list[0]][:3],
                            "team_synergy": f"Focus on supporting {char_list[0]}'s {role} role"
                        },
                        "build_recommendations": {
                            char_list[0]: {
                                "role": role,
                                "weapons": characters[char_list[0]].get('recommended_weapons', ["Various options"]),
                                "artifacts": characters[char_list[0]].get('recommended_artifacts', ["Role-appropriate set"]),
                                "main_stats": characters[char_list[0]].get('recommended_main_stats', {
                                    "sands": "ATK%/ER%",
                                    "goblet": f"{element_focus} DMG/ATK%",
                                    "circlet": "CRIT/Healing Bonus"
                                })
                            }
                        },
                        "rotation_tips": f"Prioritize {char_list[0]}'s abilities and use supports to enhance {role} performance"
                    },
                    "type": "team_composition"
                })
            
            # Team focus variations
            for focus in team_focuses:
                queries.append({
                    "query": f"Create a {focus}-focused team with {char_list[0]}",
                    "response": {
                        "team_concept": {
                            "focus": focus,
                            "main_character": char_list[0],
                            "team_style": f"{focus}-oriented gameplay"
                        },
                        "recommended_composition": {
                            "core": char_list[0],
                            "supports": [name for name, data in characters.items() 
                                       if name != char_list[0]][:3],
                            "synergy_explanation": f"Team built around {focus} mechanics"
                        },
                        "rotation_strategy": {
                            "setup": "Apply team buffs and debuffs",
                            "main_phase": f"Execute {focus}-focused combos",
                            "maintenance": "Maintain buffs and energy"
                        }
                    },
                    "type": "team_composition"
                })
    
    return queries

def main():
    """Main function to generate training data."""
    characters, weapons, domains, artifacts, team_synergies = load_data()
    
    # Generate queries
    queries = []
    queries.extend(generate_character_queries(characters))
    queries.extend(generate_team_composition_queries(characters))
    queries.extend(generate_weapon_queries(weapons))
    queries.extend(generate_domain_queries(domains))
    queries.extend(generate_artifact_queries(artifacts))
    queries.extend(generate_team_synergy_queries(team_synergies))
    
    # Save training data
    save_json('training_data/training_data.json', queries)
    
    # Generate and save dataset summary
    summary = {
        'total_samples': len(queries),
        'by_type': {}
    }
    
    for query in queries:
        query_type = query['type']
        if query_type not in summary['by_type']:
            summary['by_type'][query_type] = 0
        summary['by_type'][query_type] += 1
    
    save_json('training_data/dataset_summary.json', summary)
    
    # Print generation summary
    print(f"Loaded {len(characters)} characters")
    print(f"Loaded {len(weapons)} weapons")
    print(f"Loaded {len(domains)} domains")
    print(f"Loaded {len(artifacts)} artifacts")
    print(f"Loaded {len(team_synergies)} team synergies")
    
    print(f"\nGenerated {len(generate_character_queries(characters))} character queries")
    print(f"Generated {len(generate_team_composition_queries(characters))} team composition queries")
    print(f"Generated {len(generate_weapon_queries(weapons))} weapon queries")
    print(f"Generated {len(generate_domain_queries(domains))} domain queries")
    print(f"Generated {len(generate_artifact_queries(artifacts))} artifact queries")
    print(f"Generated {len(generate_team_synergy_queries(team_synergies))} team synergy queries")
    
    print(f"\nGenerated {len(queries)} total training samples:")
    for query_type, count in summary['by_type'].items():
        print(f"- {query_type}: {count} samples")

if __name__ == "__main__":
    main()