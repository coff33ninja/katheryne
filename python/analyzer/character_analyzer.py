class CharacterAnalyzer:
    """Class for analyzing character data and recommending builds."""

    def __init__(self, character_data, weapons_data=None, artifacts_data=None):
        self.character_data = character_data
        self.weapons_data = weapons_data or {}
        self.artifacts_data = artifacts_data or {}

    def determine_role(self, character_name):
        """Determine the recommended role for a character."""
        character = self.character_data.get(character_name, {})
        return character.get("recommended_role", "Main DPS")
        
    def determine_scaling(self, character_name):
        """Determine which stats the character scales with."""
        character = self.character_data.get(character_name, {})
        
        # Extract recommended stats from character data
        main_stats = character.get("recommended_main_stats", {})
        substats = character.get("recommended_substats", [])
        
        # Combine all stats to determine scaling
        scaling_stats = []
        
        # Add main stats
        for piece, stat in main_stats.items():
            if isinstance(stat, str) and "/" in stat:
                # Handle cases like "CRIT Rate/DMG"
                options = [s.strip() for s in stat.split("/")]
                scaling_stats.extend(options)
            else:
                scaling_stats.append(stat)
        
        # Add substats
        scaling_stats.extend(substats)
        
        # Remove duplicates and return
        return list(set(scaling_stats))
        
    def score_weapon(self, weapon_name, character_name):
        """Score a weapon's suitability for a character."""
        character = self.character_data.get(character_name, {})
        weapon = self.weapons_data.get(weapon_name, {})
        
        if not character or not weapon:
            return 0
            
        score = 0
        
        # Check if weapon is in recommended weapons
        if weapon_name in character.get("recommended_weapons", []):
            score += 100
        elif weapon_name in character.get("alternative_weapons", []):
            score += 70
            
        # Check if weapon is the signature weapon
        if weapon_name == character.get("signature_weapon"):
            score += 50
            
        # Check weapon type match
        if weapon.get("type") == character.get("weapon"):
            score += 30
        else:
            return 0  # Wrong weapon type is an immediate disqualifier
            
        # Check stat alignment
        character_scaling = self.determine_scaling(character_name)
        weapon_main_stats = weapon.get("recommended_stats", {}).get("main_stats", [])
        weapon_substats = weapon.get("recommended_stats", {}).get("substats", [])
        
        for stat in weapon_main_stats:
            if stat in character_scaling:
                score += 20
                
        for stat in weapon_substats:
            if stat in character_scaling:
                score += 10
                
        return score

    def recommend_weapons(self, character_name):
        """Recommend weapons based on character needs."""
        if not self.weapons_data:
            # Fallback to placeholder if no weapon data available
            return ["Weapon A", "Weapon B"]
            
        # Get character info
        character = self.character_data.get(character_name)
        if not character:
            return []
            
        # Filter weapons by type
        character_weapon_type = character.get("weapon")
        suitable_weapons = [
            name for name, data in self.weapons_data.items()
            if data.get("type") == character_weapon_type
        ]
        
        # Score and sort weapons
        scored_weapons = [
            (name, self.score_weapon(name, character_name))
            for name in suitable_weapons
        ]
        
        # Sort by score (descending)
        scored_weapons.sort(key=lambda x: x[1], reverse=True)
        
        # Return top weapons with scores
        top_weapons = [
            {
                "name": name,
                "score": score,
                "reason": self._get_weapon_recommendation_reason(name, character_name)
            }
            for name, score in scored_weapons[:5]  # Top 5 weapons
        ]
        
        return top_weapons
        
    def _get_weapon_recommendation_reason(self, weapon_name, character_name):
        """Generate a reason for recommending a weapon."""
        character = self.character_data.get(character_name, {})
        weapon = self.weapons_data.get(weapon_name, {})
        
        reasons = []
        
        if weapon_name == character.get("signature_weapon"):
            reasons.append("Signature weapon designed for this character")
            
        if weapon_name in character.get("recommended_weapons", []):
            reasons.append("Officially recommended for this character")
            
        # Add reason based on passive
        passive = weapon.get("passive_description", "")
        if passive:
            reasons.append(f"Passive: {passive}")
            
        # Add reason based on stats
        character_scaling = self.determine_scaling(character_name)
        weapon_stats = weapon.get("recommended_stats", {}).get("main_stats", [])
        
        matching_stats = [stat for stat in weapon_stats if stat in character_scaling]
        if matching_stats:
            reasons.append(f"Provides useful stats: {', '.join(matching_stats)}")
            
        return reasons[0] if reasons else "Good overall option"
        
    def score_artifact_set(self, artifact_set_name, character_name):
        """Score an artifact set's suitability for a character."""
        character = self.character_data.get(character_name, {})
        artifact_set = self.artifacts_data.get(artifact_set_name, {})
        
        if not character or not artifact_set:
            return 0
            
        score = 0
        
        # Check if artifact is in recommended artifacts
        if artifact_set_name in character.get("recommended_artifacts", []):
            score += 100
            
        # Check set bonuses alignment with character needs
        character_element = character.get("element")
        character_role = self.determine_role(character_name)
        character_scaling = self.determine_scaling(character_name)
        
        # Check 2-piece bonus
        two_piece_bonus = artifact_set.get("set_bonus", {}).get("2_piece", "")
        if character_element and f"{character_element} DMG" in two_piece_bonus:
            score += 30
        if "ATK%" in two_piece_bonus and "ATK%" in character_scaling:
            score += 20
        if "HP%" in two_piece_bonus and "HP%" in character_scaling:
            score += 20
        if "DEF%" in two_piece_bonus and "DEF%" in character_scaling:
            score += 20
        if "Energy Recharge" in two_piece_bonus and "Energy Recharge" in character_scaling:
            score += 20
        if "Elemental Mastery" in two_piece_bonus and "Elemental Mastery" in character_scaling:
            score += 20
            
        # Check 4-piece bonus (more complex, simplified here)
        four_piece_bonus = artifact_set.get("set_bonus", {}).get("4_piece", "")
        if character_role == "Main DPS" and ("DMG" in four_piece_bonus or "ATK" in four_piece_bonus):
            score += 30
        if character_role == "Support" and ("Shield" in four_piece_bonus or "Healing" in four_piece_bonus):
            score += 30
        if character_role == "Sub DPS" and "Elemental Burst" in four_piece_bonus:
            score += 30
            
        # Check recommended characters
        if character_name in artifact_set.get("recommended_characters", []):
            score += 40
            
        return score

    def recommend_artifacts(self, character_name):
        """Recommend artifacts based on character needs."""
        if not self.artifacts_data:
            # Fallback to placeholder if no artifact data available
            return ["Artifact Set A", "Artifact Set B"]
            
        # Get character info
        character = self.character_data.get(character_name)
        if not character:
            return []
            
        # Score and sort artifact sets
        scored_artifacts = [
            (name, self.score_artifact_set(name, character_name))
            for name in self.artifacts_data.keys()
        ]
        
        # Sort by score (descending)
        scored_artifacts.sort(key=lambda x: x[1], reverse=True)
        
        # Return top artifact sets with scores and reasons
        top_artifacts = [
            {
                "name": name,
                "score": score,
                "reason": self._get_artifact_recommendation_reason(name, character_name),
                "recommended_stats": self._get_recommended_artifact_stats(name, character_name)
            }
            for name, score in scored_artifacts[:3]  # Top 3 artifact sets
        ]
        
        return top_artifacts
        
    def _get_artifact_recommendation_reason(self, artifact_set_name, character_name):
        """Generate a reason for recommending an artifact set."""
        character = self.character_data.get(character_name, {})
        artifact_set = self.artifacts_data.get(artifact_set_name, {})
        
        reasons = []
        
        if artifact_set_name in character.get("recommended_artifacts", []):
            reasons.append("Officially recommended for this character")
            
        # Add reason based on set bonuses
        two_piece = artifact_set.get("set_bonus", {}).get("2_piece", "")
        four_piece = artifact_set.get("set_bonus", {}).get("4_piece", "")
        
        if two_piece:
            reasons.append(f"2-Piece: {two_piece}")
        if four_piece:
            reasons.append(f"4-Piece: {four_piece}")
            
        return reasons[0] if reasons else "Good overall option"
        
    def _get_recommended_artifact_stats(self, artifact_set_name, character_name):
        """Get recommended main stats and substats for artifacts."""
        character = self.character_data.get(character_name, {})
        artifact_set = self.artifacts_data.get(artifact_set_name, {})
        
        # Get character role
        role = self.determine_role(character_name)
        role_lower = role.lower()
        
        # Try to get role-specific recommendations from artifact data
        artifact_recommendations = artifact_set.get("recommended_stats", {}).get("main_stats", {})
        
        # Find the closest matching role in artifact recommendations
        matching_role = None
        for available_role in artifact_recommendations.keys():
            if role_lower in available_role.lower():
                matching_role = available_role
                break
                
        # If no matching role found, use character's own recommendations
        if matching_role:
            main_stats = artifact_recommendations.get(matching_role, {})
        else:
            main_stats = character.get("recommended_main_stats", {})
            
        # Get substats (prioritize character's own recommendations)
        substats = character.get("recommended_substats", [])
        if not substats and artifact_set.get("recommended_stats", {}).get("substats"):
            substats = artifact_set.get("recommended_stats", {}).get("substats")
            
        return {
            "main_stats": main_stats,
            "substats": substats
        }

    def analyze_character(self, character_name):
        """Analyze character and provide comprehensive recommendations."""
        character = self.character_data.get(character_name)
        if not character:
            return {"error": f"Character '{character_name}' not found"}
            
        # Determine character role and scaling
        role = self.determine_role(character_name)
        scaling = self.determine_scaling(character_name)
        
        # Get weapon and artifact recommendations
        weapons = self.recommend_weapons(character_name)
        artifacts = self.recommend_artifacts(character_name)
        
        # Compile analysis results
        analysis = {
            "name": character_name,
            "element": character.get("element"),
            "weapon_type": character.get("weapon"),
            "rarity": character.get("rarity"),
            "role": role,
            "scaling_stats": scaling,
            "weapons": weapons,
            "artifacts": artifacts,
            "talent_priority": self._determine_talent_priority(character_name)
        }
        
        return analysis
        
    def _determine_talent_priority(self, character_name):
        """Determine talent priority based on character role."""
        role = self.determine_role(character_name)
        
        # Default priorities based on role
        if role == "Main DPS":
            return ["Normal Attack", "Elemental Burst", "Elemental Skill"]
        elif role == "Sub DPS":
            return ["Elemental Burst", "Elemental Skill", "Normal Attack"]
        elif role == "Support":
            return ["Elemental Burst", "Elemental Skill", "Normal Attack"]
        else:
            return ["Elemental Skill", "Elemental Burst", "Normal Attack"]
