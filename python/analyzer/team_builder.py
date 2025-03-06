class TeamBuilder:
    """Class for building teams based on character roles and synergies."""

    def __init__(self, characters, team_synergies=None, weapons_data=None, artifacts_data=None):
        self.characters = characters
        self.team_synergies = team_synergies or {}
        self.weapons_data = weapons_data or {}
        self.artifacts_data = artifacts_data or {}
        
        # Import here to avoid circular imports
        from analyzer.character_analyzer import CharacterAnalyzer
        self.analyzer = CharacterAnalyzer(characters, weapons_data, artifacts_data)

    def build_team(self, main_character):
        """Build a team around a main character."""
        if main_character not in self.characters:
            return {"team": [main_character, "Support A", "Support B", "Support C"]}
            
        # Get main character details
        main_char_data = self.characters.get(main_character, {})
        main_char_element = main_char_data.get("element")
        main_char_role = self.analyzer.determine_role(main_character)
        
        # Find synergistic teams in team_synergies
        synergistic_teams = []
        for team_id, team_data in self.team_synergies.items():
            if main_character in team_data.get("core_characters", []):
                synergistic_teams.append(team_data)
                
        if synergistic_teams:
            # Use the first synergistic team as a template
            team_template = synergistic_teams[0]
            core_chars = team_template.get("core_characters", [])
            support_options = team_template.get("support_options", [])
            
            # Start with core characters
            team = [char for char in core_chars if char in self.characters]
            
            # Add support characters until we have 4 total
            for support in support_options:
                if support in self.characters and support not in team:
                    team.append(support)
                if len(team) >= 4:
                    break
                    
            # If we still don't have 4 characters, add recommended supports
            if len(team) < 4:
                additional_supports = self._find_compatible_supports(team)
                for support in additional_supports:
                    if support not in team:
                        team.append(support)
                    if len(team) >= 4:
                        break
                        
            return {"team": team[:4]}  # Ensure we return exactly 4 characters
            
        else:
            # No synergistic team found, build from scratch
            team = [main_character]
            
            # Find characters that complement the main character
            complementary_chars = self._find_complementary_characters(main_character)
            for char in complementary_chars:
                if char not in team:
                    team.append(char)
                if len(team) >= 4:
                    break
                    
            # If we still don't have 4 characters, add generic supports
            if len(team) < 4:
                generic_supports = ["Bennett", "Zhongli", "Xingqiu", "Kazuha"]
                for support in generic_supports:
                    if support in self.characters and support not in team:
                        team.append(support)
                    if len(team) >= 4:
                        break
                        
            # If still not enough, add any characters
            if len(team) < 4:
                for char in self.characters:
                    if char not in team:
                        team.append(char)
                    if len(team) >= 4:
                        break
                        
            return {"team": team[:4]}  # Ensure we return exactly 4 characters
            
    def _find_complementary_characters(self, main_character):
        """Find characters that complement the main character."""
        main_char_data = self.characters.get(main_character, {})
        main_char_element = main_char_data.get("element")
        
        complementary_chars = []
        
        # Find characters that create good elemental reactions
        reaction_pairs = {
            "Pyro": ["Hydro", "Cryo", "Electro"],
            "Hydro": ["Pyro", "Cryo", "Electro"],
            "Cryo": ["Pyro", "Hydro", "Electro"],
            "Electro": ["Pyro", "Hydro", "Cryo"],
            "Anemo": ["Pyro", "Hydro", "Cryo", "Electro"],
            "Geo": ["Geo"],  # Geo resonance
            "Dendro": ["Hydro", "Electro", "Pyro"]
        }
        
        # Get complementary elements
        complementary_elements = reaction_pairs.get(main_char_element, [])
        
        # Find characters with complementary elements
        for char, data in self.characters.items():
            if char == main_character:
                continue
                
            char_element = data.get("element")
            
            # Prioritize characters with complementary elements
            if char_element in complementary_elements:
                complementary_chars.append(char)
                
        # Sort by rarity (higher first)
        complementary_chars.sort(
            key=lambda c: self.characters.get(c, {}).get("rarity", 0),
            reverse=True
        )
        
        return complementary_chars
        
    def _find_compatible_supports(self, current_team):
        """Find support characters compatible with the current team."""
        # Get elements in current team
        team_elements = [
            self.characters.get(char, {}).get("element")
            for char in current_team
            if char in self.characters
        ]
        
        # Find characters that provide missing utility
        has_healer = any(
            self.characters.get(char, {}).get("role") == "Healer"
            for char in current_team
            if char in self.characters
        )
        
        has_shield = any(
            "Shield" in self.characters.get(char, {}).get("abilities", [])
            for char in current_team
            if char in self.characters
        )
        
        # Score each character as a potential support
        scored_supports = []
        for char, data in self.characters.items():
            if char in current_team:
                continue
                
            score = 0
            char_element = data.get("element")
            char_role = data.get("role", "")
            
            # Bonus for healers if team has no healer
            if not has_healer and "Healer" in char_role:
                score += 30
                
            # Bonus for shielders if team has no shield
            if not has_shield and "Shield" in data.get("abilities", []):
                score += 25
                
            # Bonus for Anemo supports (VV set users)
            if char_element == "Anemo" and "Support" in char_role:
                score += 20
                
            # Bonus for unique elements (elemental diversity)
            if char_element not in team_elements:
                score += 15
                
            # Bonus for resonance (having 2 of same element)
            if team_elements.count(char_element) == 1:
                score += 10
                
            # Bonus for popular supports
            if char in ["Bennett", "Xingqiu", "Zhongli", "Kazuha"]:
                score += 20
                
            scored_supports.append((char, score))
            
        # Sort by score (descending)
        scored_supports.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the character names
        return [char for char, _ in scored_supports]

    def evaluate_team(self, team):
        """Evaluate a team composition and provide detailed analysis."""
        if not all(char in self.characters for char in team):
            # Fallback for unknown characters
            return {"suggestions": ["Adjust character X", "Add character Y"]}
            
        # Calculate team score based on various factors
        score = self._calculate_team_score(team)
        
        # Identify team strengths and weaknesses
        strengths = self._identify_team_strengths(team)
        weaknesses = self._identify_team_weaknesses(team)
        
        # Identify elemental reactions
        reactions = self._identify_elemental_reactions(team)
        
        # Identify resonances
        resonances = self._identify_elemental_resonances(team)
        
        # Identify team synergies
        synergies = self._identify_team_synergies(team)
        
        # Suggest rotation
        rotation = self._suggest_rotation(team)
        
        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "elemental_reactions": reactions,
            "elemental_resonances": resonances,
            "synergies": synergies,
            "suggested_rotation": rotation,
            "suggestions": self.suggest_team_adjustments(team).get("suggestions", [])
        }
        
    def _calculate_team_score(self, team):
        """Calculate a numerical score for the team composition."""
        score = 50  # Base score
        
        # Check if team exists in known team synergies
        for team_id, team_data in self.team_synergies.items():
            core_chars = team_data.get("core_characters", [])
            if all(char in team for char in core_chars):
                score += 20
                break
                
        # Check for elemental resonances
        resonances = self._identify_elemental_resonances(team)
        score += len(resonances) * 5
        
        # Check for role balance
        roles = [
            self.analyzer.determine_role(char)
            for char in team
            if char in self.characters
        ]
        
        has_main_dps = "Main DPS" in roles
        has_sub_dps = "Sub DPS" in roles
        has_support = "Support" in roles
        
        if has_main_dps:
            score += 10
        if has_sub_dps:
            score += 5
        if has_support:
            score += 5
            
        # Check for healing and shields
        has_healer = any(
            "Healer" in self.characters.get(char, {}).get("role", "")
            for char in team
            if char in self.characters
        )
        
        has_shield = any(
            "Shield" in self.characters.get(char, {}).get("abilities", [])
            for char in team
            if char in self.characters
        )
        
        if has_healer:
            score += 5
        if has_shield:
            score += 5
            
        # Check for elemental diversity
        elements = [
            self.characters.get(char, {}).get("element")
            for char in team
            if char in self.characters
        ]
        
        unique_elements = len(set(elements))
        score += unique_elements * 2
        
        # Cap score at 100
        return min(score, 100)
        
    def _identify_team_strengths(self, team):
        """Identify the strengths of the team composition."""
        strengths = []
        
        # Check for elemental resonances
        resonances = self._identify_elemental_resonances(team)
        if resonances:
            strengths.append(f"Elemental Resonance: {', '.join(resonances)}")
            
        # Check for role balance
        roles = [
            self.analyzer.determine_role(char)
            for char in team
            if char in self.characters
        ]
        
        if "Main DPS" in roles and "Support" in roles:
            strengths.append("Balanced team with main DPS and support")
            
        # Check for healing and shields
        has_healer = any(
            "Healer" in self.characters.get(char, {}).get("role", "")
            for char in team
            if char in self.characters
        )
        
        has_shield = any(
            "Shield" in self.characters.get(char, {}).get("abilities", [])
            for char in team
            if char in self.characters
        )
        
        if has_healer:
            strengths.append("Team includes healing")
        if has_shield:
            strengths.append("Team includes shields for protection")
            
        # Check for elemental reactions
        reactions = self._identify_elemental_reactions(team)
        if len(reactions) >= 3:
            strengths.append("Multiple elemental reactions possible")
        elif reactions:
            strengths.append(f"Good elemental reactions: {', '.join(reactions)}")
            
        # Check for known synergies
        for char1 in team:
            for char2 in team:
                if char1 != char2:
                    synergy = self._check_character_synergy(char1, char2)
                    if synergy:
                        strengths.append(synergy)
                        
        return strengths
        
    def _identify_team_weaknesses(self, team):
        """Identify the weaknesses of the team composition."""
        weaknesses = []
        
        # Check for role balance
        roles = [
            self.analyzer.determine_role(char)
            for char in team
            if char in self.characters
        ]
        
        if "Main DPS" not in roles:
            weaknesses.append("No dedicated main DPS character")
            
        # Check for healing and shields
        has_healer = any(
            "Healer" in self.characters.get(char, {}).get("role", "")
            for char in team
            if char in self.characters
        )
        
        has_shield = any(
            "Shield" in self.characters.get(char, {}).get("abilities", [])
            for char in team
            if char in self.characters
        )
        
        if not has_healer:
            weaknesses.append("No healing capability")
        if not has_shield:
            weaknesses.append("No shield protection")
            
        # Check for elemental diversity
        elements = [
            self.characters.get(char, {}).get("element")
            for char in team
            if char in self.characters
        ]
        
        unique_elements = len(set(elements))
        if unique_elements < 3:
            weaknesses.append("Limited elemental diversity")
            
        # Check for energy generation
        energy_issues = self._check_energy_issues(team)
        if energy_issues:
            weaknesses.append("Potential energy regeneration issues")
            
        # Check for specific element counters
        if not any(elem in elements for elem in ["Pyro", "Electro", "Cryo"]):
            weaknesses.append("May struggle against shields requiring specific elements")
            
        return weaknesses
        
    def _identify_elemental_reactions(self, team):
        """Identify possible elemental reactions in the team."""
        elements = [
            self.characters.get(char, {}).get("element")
            for char in team
            if char in self.characters
        ]
        
        reactions = []
        
        # Define possible reactions
        if "Pyro" in elements and "Hydro" in elements:
            reactions.append("Vaporize")
        if "Pyro" in elements and "Cryo" in elements:
            reactions.append("Melt")
        if "Electro" in elements and "Hydro" in elements:
            reactions.append("Electro-Charged")
        if "Electro" in elements and "Pyro" in elements:
            reactions.append("Overloaded")
        if "Electro" in elements and "Cryo" in elements:
            reactions.append("Superconduct")
        if "Anemo" in elements and len(set(elements) - {"Anemo", "Geo"}) > 0:
            reactions.append("Swirl")
        if "Geo" in elements:
            reactions.append("Crystallize")
        if "Dendro" in elements and "Hydro" in elements:
            reactions.append("Bloom")
        if "Dendro" in elements and "Electro" in elements:
            reactions.append("Quicken")
        if "Dendro" in elements and "Pyro" in elements:
            reactions.append("Burning")
            
        return reactions
        
    def _identify_elemental_resonances(self, team):
        """Identify elemental resonances in the team."""
        elements = [
            self.characters.get(char, {}).get("element")
            for char in team
            if char in self.characters
        ]
        
        resonances = []
        
        # Check for resonances (2+ of same element)
        element_counts = {}
        for elem in elements:
            if elem:
                element_counts[elem] = element_counts.get(elem, 0) + 1
                
        for elem, count in element_counts.items():
            if count >= 2:
                resonances.append(f"{elem} Resonance")
                
        return resonances
        
    def _check_character_synergy(self, char1, char2):
        """Check if two characters have specific synergy."""
        # Known character synergies
        synergies = {
            ("Xingqiu", "Hu Tao"): "Xingqiu enables Hu Tao's vaporize reactions",
            ("Bennett", "Xiangling"): "Bennett batteries Xiangling's burst",
            ("Zhongli", "Albedo"): "Geo construct resonance",
            ("Raiden Shogun", "Eula"): "Superconduct for physical damage",
            ("Xingqiu", "Diluc"): "Consistent vaporize reactions",
            ("Ganyu", "Mona"): "Freeze comp core",
            ("Kazuha", "Ayaka"): "Elemental damage boost for Cryo"
        }
        
        # Check both directions
        synergy = synergies.get((char1, char2)) or synergies.get((char2, char1))
        return synergy
        
    def _check_energy_issues(self, team):
        """Check if the team might have energy regeneration issues."""
        # Characters with high energy requirements
        high_energy_chars = [
            "Xiangling", "Beidou", "Xingqiu", "Eula", "Ayaka"
        ]
        
        # Characters that generate a lot of particles
        battery_chars = [
            "Bennett", "Raiden Shogun", "Venti", "Fischl", "Sucrose"
        ]
        
        # Check if we have high energy characters without batteries
        high_energy_in_team = [char for char in team if char in high_energy_chars]
        batteries_in_team = [char for char in team if char in battery_chars]
        
        return len(high_energy_in_team) > 0 and len(batteries_in_team) == 0
        
    def _identify_team_synergies(self, team):
        """Identify if the team matches any known team synergies."""
        matching_synergies = []
        
        for team_id, team_data in self.team_synergies.items():
            core_chars = team_data.get("core_characters", [])
            if all(char in team for char in core_chars):
                matching_synergies.append(team_data.get("name", team_id))
                
        return matching_synergies
        
    def _suggest_rotation(self, team):
        """Suggest a basic rotation for the team."""
        # Get character roles
        roles = {
            char: self.analyzer.determine_role(char)
            for char in team
            if char in self.characters
        }
        
        # Find supports and main DPS
        supports = [char for char, role in roles.items() if role in ["Support", "Sub DPS"]]
        main_dps = [char for char, role in roles.items() if role == "Main DPS"]
        
        if not main_dps:
            # If no main DPS, use the first character as main
            main_dps = [team[0]] if team else []
            
        # Basic rotation template
        rotation = []
        
        # Start with supports to set up buffs/reactions
        for support in supports:
            rotation.append(f"{support}: Use Elemental Skill/Burst")
            
        # Then use main DPS
        for dps in main_dps:
            rotation.append(f"{dps}: Use full damage rotation")
            
        # If we have a known team synergy, use its rotation instead
        for team_id, team_data in self.team_synergies.items():
            core_chars = team_data.get("core_characters", [])
            if all(char in team for char in core_chars):
                team_rotation = team_data.get("rotation", {})
                if team_rotation:
                    rotation = [
                        f"Setup: {team_rotation.get('setup', '')}",
                        f"Main phase: {team_rotation.get('main_phase', '')}",
                        f"Support: {team_rotation.get('support', '')}"
                    ]
                break
                
        return rotation
        
    def suggest_team_adjustments(self, team):
        """Suggest adjustments to improve the team composition."""
        if not all(char in self.characters for char in team):
            return {
                "suggestions": ["Replace unknown characters with known ones"],
                "alternative_characters": {}
            }
            
        # Evaluate current team
        weaknesses = self._identify_team_weaknesses(team)
        
        suggestions = []
        alternative_characters = {}
        
        # Address weaknesses with specific suggestions
        for weakness in weaknesses:
            if "No dedicated main DPS" in weakness:
                suggestions.append("Add a main DPS character")
                alternative_characters["main_dps"] = self._suggest_characters_by_role("Main DPS")
                
            elif "No healing capability" in weakness:
                suggestions.append("Add a healer for survivability")
                alternative_characters["healer"] = self._suggest_characters_by_ability("Healer")
                
            elif "No shield protection" in weakness:
                suggestions.append("Add a shielder for protection")
                alternative_characters["shielder"] = self._suggest_characters_by_ability("Shield")
                
            elif "Limited elemental diversity" in weakness:
                suggestions.append("Add more elemental diversity")
                elements = [
                    self.characters.get(char, {}).get("element")
                    for char in team
                    if char in self.characters
                ]
                missing_elements = [e for e in ["Pyro", "Hydro", "Cryo", "Electro", "Anemo", "Geo", "Dendro"] 
                                   if e not in elements]
                alternative_characters["elements"] = self._suggest_characters_by_elements(missing_elements)
                
            elif "energy regeneration issues" in weakness.lower():
                suggestions.append("Add an energy battery character")
                alternative_characters["battery"] = ["Bennett", "Raiden Shogun", "Venti", "Fischl", "Sucrose"]
                
        # If no specific weaknesses found, suggest general improvements
        if not suggestions:
            suggestions.append("Team looks solid, consider trying different elemental combinations for variety")
            
        return {
            "suggestions": suggestions,
            "alternative_characters": alternative_characters
        }
        
    def _suggest_characters_by_role(self, role):
        """Suggest characters that fulfill a specific role."""
        return [
            char for char, data in self.characters.items()
            if data.get("role") == role
        ][:5]  # Return top 5
        
    def _suggest_characters_by_ability(self, ability):
        """Suggest characters that have a specific ability."""
        return [
            char for char, data in self.characters.items()
            if ability in data.get("abilities", []) or ability in data.get("role", "")
        ][:5]  # Return top 5
        
    def _suggest_characters_by_elements(self, elements):
        """Suggest characters of specific elements."""
        suggestions = {}
        for element in elements:
            suggestions[element] = [
                char for char, data in self.characters.items()
                if data.get("element") == element
            ][:3]  # Top 3 per element
            
        return suggestions
