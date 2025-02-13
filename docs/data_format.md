# Data Format Specifications

This document describes the data formats used in the Katheryne project.

## Input Data Format

### Characters (data/characters/*.json)

```json
{
  "name": "Character Name",
  "element": "Element Type",
  "weapon": "Weapon Type",
  "rarity": 5,
  "description": "Character description",
  "recommended_role": "Main DPS/Support/etc",
  "recommended_weapons": ["Weapon1", "Weapon2"],
  "recommended_artifacts": ["Artifact Set 1", "Artifact Set 2"],
  "recommended_main_stats": {
    "sands": ["ATK%", "ER%"],
    "goblet": ["Element DMG"],
    "circlet": ["CRIT Rate", "CRIT DMG"]
  },
  "recommended_substats": ["CRIT Rate", "CRIT DMG", "ATK%"],
  "constellations": {
    "1": "Constellation 1 description",
    "2": "Constellation 2 description"
  }
}
```

### Weapons (data/weapons/*.json)

```json
{
  "name": "Weapon Name",
  "type": "Weapon Type",
  "rarity": 5,
  "baseAttack": 608,
  "description": "Weapon description",
  "passive_description": "Passive ability description",
  "recommended_characters": ["Character1", "Character2"],
  "character_synergy": "Synergy description",
  "recommended_stats": {
    "main_stats": ["ATK%", "Element DMG"],
    "substats": ["CRIT Rate", "CRIT DMG"]
  }
}
```

### Artifacts (data/artifacts/*.json)

```json
{
  "name": "Artifact Set Name",
  "set_bonus": {
    "2pc": "2-piece bonus description",
    "4pc": "4-piece bonus description"
  },
  "recommended_characters": ["Character1", "Character2"],
  "recommended_stats": {
    "main_stats": {
      "flower": "HP",
      "plume": "ATK",
      "sands": ["ATK%", "ER%"],
      "goblet": ["Element DMG"],
      "circlet": ["CRIT Rate", "CRIT DMG"]
    },
    "substats": ["CRIT Rate", "CRIT DMG", "ATK%", "ER%"]
  },
  "domain": "Domain Name"
}
```

## Training Data Format

### Query-Response Format (training_data/training_data.json)

```json
{
  "query": "User question text",
  "response": {
    "field1": "value1",
    "field2": "value2"
  },
  "type": "query_type"
}
```

Query types:
- character_info
- advanced_build
- weapon_info
- advanced_weapon
- domain_schedule
- domain_strategy
- artifact_info
- artifact_stats
- artifact_farming
- team_synergy
- team_rotation
- team_building
- team_composition

### Dataset Summary Format (training_data/dataset_summary.json)

```json
{
  "total_samples": 100,
  "by_type": {
    "character_info": 20,
    "advanced_build": 15,
    "weapon_info": 10
  }
}
```

## Model Input/Output Format

### Model Input Format

```python
{
    "text": "Query text",
    "type": "Query type",
    "context": {
        "character": "Character name",
        "weapon": "Weapon name",
        "artifact": "Artifact set name"
    }
}
```

### Model Output Format

```python
{
    "response": {
        "text": "Generated response",
        "data": {
            "field1": "value1",
            "field2": "value2"
        }
    },
    "confidence": 0.95,
    "sources": ["data_source1", "data_source2"]
}
```

## Adding New Data

1. Follow the JSON schemas above
2. Place files in appropriate directories
3. Run validation:
```bash
python python/validate_data.py
```

## Data Validation

The project includes validation for:
- JSON schema compliance
- Data consistency
- Required fields
- Value ranges
- Cross-references

See `python/validate_data.py` for implementation details.