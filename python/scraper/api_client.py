import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import requests

class GenshinAPIClient:
    """Client for fetching data from Genshin Impact APIs."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints for Genshin Impact data
        from analyzer.character_analyzer import CharacterAnalyzer
        from analyzer.team_builder import TeamBuilder


        self.primary_api = "https://genshin.dev/api"
        self.secondary_api = "https://genshin.jmp.blue"

        # Known types from API documentation
        self.known_types = [
            "artifacts",
            "boss",
            "characters",
            "consumables",
            "domains",
            "elements",
            "enemies",
            "materials",
            "nations",
            "weapons"
        ]

        # Rate limiting
        self.request_delay = 1.0  # seconds between requests

    def verify_data(self) -> Dict[str, bool]:
        """Verify that all required data files exist and contain valid data."""
        verification_results = {}

        def verify_file(file_name: str, required_fields: List[str] = None) -> bool:
            file_path = self.raw_dir / file_name
            if not file_path.exists():
                print(f"Missing file: {file_name}")
                return False

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data:
                    print(f"Empty data in {file_name}")
                    return False

                if required_fields:
                    # Check first item for required fields
                    first_item = data[0] if isinstance(data, list) else data
                    missing_fields = [field for field in required_fields if field not in first_item]
                    if missing_fields:
                        print(f"Missing required fields in {file_name}: {missing_fields}")
                        return False

                return True
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {file_name}: {str(e)}")
                return False

        # Verify characters data
        verification_results['characters'] = verify_file('characters.json', ['name', 'vision', 'weapon'])
        verification_results['characters_details'] = verify_file('character_ascension.json', ['character', 'ascension_materials'])

        # Verify weapons data
        verification_results['weapons'] = verify_file('weapons.json', ['name', 'type', 'rarity'])
        verification_results['weapons_details'] = all([
            verify_file('weapons.json', ['refinements', 'ascension_materials'])
        ])

        # Verify artifacts data
        verification_results['artifacts'] = verify_file('artifacts.json', ['name', 'max_rarity'])
        verification_results['artifacts_details'] = all([
            verify_file('artifacts.json', ['rarity_variations', 'stat_scaling'])
        ])

        # Verify enemies data
        verification_results['enemies'] = verify_file('enemies.json', ['name'])
        verification_results['enemies_details'] = verify_file('enemies.json', ['drops', 'description'])

        # Verify domains data
        verification_results['domains'] = verify_file('domains.json', ['name'])
        verification_results['domains_details'] = verify_file('domains.json', ['rewards', 'requirements'])

        return verification_results

    def fetch_all_data(self) -> Dict[str, List]:
        """Fetch all data from APIs and initialize character analysis."""

        """Fetch all data from APIs."""
        print("Fetching data from Genshin Impact APIs...")
        data = {}

        try:
            print("Starting download of all data...")

            print("Downloading characters data... (1/5)")
            self.fetch_characters()

            print("Downloading weapons data... (2/5)")
            self.fetch_weapons()

            print("Downloading artifacts data... (3/5)")
            self.fetch_artifacts()

            print("Downloading enemies data... (4/5)")
            enemies = self._make_request(f"{self.secondary_api}/enemies")
            if enemies:
                enemies_data = []
                for enemy_id in enemies:
                    enemy_details = self._make_request(f"{self.secondary_api}/enemies/{enemy_id}")
                    if enemy_details:
                        enemy_details['id'] = enemy_id
                        enemies_data.append(enemy_details)
                if enemies_data:
                    self._save_json(enemies_data, "enemies.json")

            print("Downloading domains data... (5/5)")
            self.fetch_domains()

            print("Download complete!")

            # Process character data to extract constellations and talents
            self.process_character_data()

            # Verify downloaded data
            verification_results = self.verify_data()
            print("\nData verification results:")
            for key, success in verification_results.items():
                print(f"{key}: {'✓' if success else '✗'}")

            if not all(verification_results.values()):
                print("\nWarning: Some data failed verification. Check logs for details.")

        except Exception as e:
            print(f"\nError downloading all data: {str(e)}")
            print("Check logs for details.")
            raise

        return data

    def _make_request(self, url: str) -> Optional[Any]:
        """Make an API request with rate limiting and error handling."""
        try:
            time.sleep(self.request_delay)  # Rate limiting
            print(f"  Requesting: {url}")  # Debug output
            response = requests.get(url, timeout=10)

            # Print status code for debugging
            print(f"  Status: {response.status_code}")

            if response.status_code == 404:
                print(f"  Warning: Resource not found at {url}")
                return None
            elif response.status_code == 429:
                print(f"  Warning: Rate limit hit, waiting longer...")
                time.sleep(5)  # Wait longer on rate limit
                return self._make_request(url)  # Retry

            response.raise_for_status()

            try:
                data = response.json()
                if not data:
                    print(f"  Warning: Empty response from {url}")
                return data
            except json.JSONDecodeError as e:
                print(f"  Error: Invalid JSON from {url}: {str(e)}")
                print(f"  Response text: {response.text[:200]}...")  # Print first 200 chars
                return None

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {url}: {str(e)}")
            # Attempt to use the secondary API if the primary fails
            if url.startswith(self.primary_api):
                secondary_url = url.replace(self.primary_api, self.secondary_api)
                print(f"  Trying secondary API: {secondary_url}")
                try:
                    time.sleep(self.request_delay)  # Rate limiting
                    response = requests.get(secondary_url, timeout=10)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e:
                    print(f"  Error fetching {secondary_url}: {str(e)}")
                    print("  Failed to fetch data from both primary and secondary APIs.")
                    return None
            print("  Failed to fetch data from the primary API.")
            return None

    def _save_json(self, data: Union[Dict, List], filename: str):
        """Save data to JSON file."""
        filepath = self.raw_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {filepath}")

    def process_character_data(self):
        """Process character data into separate files for constellations and talents."""
        print("\nProcessing character data...")

        # Load the character data
        char_file = self.raw_dir / "characters.json"
        if not char_file.exists():
            print("Character data file not found")
            return

        with open(char_file, 'r', encoding='utf-8') as f:
            characters = json.load(f)

        # Extract constellations
        constellations_data = []
        talents_data = []

        for char in characters:
            char_name = char.get('name', '')

            # Process constellations
            if 'constellations' in char:
                for level, const in enumerate(char['constellations'], 1):
                    constellations_data.append({
                        'character': char_name,
                        'level': level,
                        'name': const.get('name', ''),
                        'description': const.get('description', ''),
                        'effects': const.get('effects', [])
                    })

            # Process talents
            if 'talents' in char:
                for talent in char['talents']:
                    talents_data.append({
                        'character': char_name,
                        'name': talent.get('name', ''),
                        'description': talent.get('description', ''),
                        'type': talent.get('type', ''),
                        'effects': talent.get('effects', [])
                    })

        # Save processed data
        if constellations_data:
            self._save_json(constellations_data, "constellations_detailed.json")
            print(f"Saved {len(constellations_data)} constellation entries")

        if talents_data:
            self._save_json(talents_data, "talents_detailed.json")
            print(f"Saved {len(talents_data)} talent entries")

    def fetch_characters(self):
        """Fetch character data including constellations and talents."""
        print("\nFetching characters...")

        # Get list of characters
        characters = self._make_request(f"{self.secondary_api}/characters")
        if not characters:
            print("Failed to fetch character list")
            return

        # Fetch details for each character
        character_data = []
        for char_id in characters:
            print(f"  Fetching character: {char_id}")
            # Get basic character details
            char_details = self._make_request(f"{self.secondary_api}/characters/{char_id}")
            if char_details:
                char_details['id'] = char_id
                char_details['name'] = char_details.get('name', char_id)

                # Fetch constellations
                constellations = self._make_request(f"{self.secondary_api}/characters/{char_id}/constellations")
                if constellations:
                    char_details['constellations'] = constellations

                # Fetch talents/abilities
                talents = self._make_request(f"{self.secondary_api}/characters/{char_id}/talents")
                if talents:
                    char_details['talents'] = talents

                character_data.append(char_details)

        if character_data:
            self._save_json(character_data, "characters.json")
            print(f"Saved {len(character_data)} characters with constellations and talents")

    def fetch_artifacts(self):
        """Fetch artifact data including set bonuses and rarity information."""
        print("\nFetching artifacts...")

        # Get list of artifacts
        artifacts = self._make_request(f"{self.secondary_api}/artifacts")
        if not artifacts:
            print("Failed to fetch artifact list")
            return

        # Fetch details for each artifact
        artifact_data = []
        for artifact_id in artifacts:
            print(f"  Fetching artifact: {artifact_id}")
            artifact_details = self._make_request(f"{self.secondary_api}/artifacts/{artifact_id}")
            if artifact_details:
                artifact_details['id'] = artifact_id
                artifact_details['name'] = artifact_details.get('name', artifact_id)

                # Add rarity variations if available
                rarities = self._make_request(f"{self.secondary_api}/artifacts/{artifact_id}/rarities")
                if rarities:
                    artifact_details['rarity_variations'] = rarities

                # Add stat scaling if available
                stats = self._make_request(f"{self.secondary_api}/artifacts/{artifact_id}/stats")
                if stats:
                    artifact_details['stat_scaling'] = stats

                artifact_data.append(artifact_details)

        if artifact_data:
            self._save_json(artifact_data, "artifacts.json")
            print(f"Saved {len(artifact_data)} artifacts with complete information")

    def fetch_weapons(self):
        """Fetch weapon data including refinements and ascension materials."""
        print("\nFetching weapons...")

        try:
            # Get list of weapons
            weapons = self._make_request(f"{self.secondary_api}/weapons")
            if not weapons:
                print("Failed to fetch weapon list")
                return

            # Fetch details for each weapon
            weapon_data = []
            total_weapons = len(weapons)

            for idx, weapon_id in enumerate(weapons, 1):
                print(f"  Fetching weapon: {weapon_id} ({idx}/{total_weapons})")
                try:
                    weapon_details = self._make_request(f"{self.secondary_api}/weapons/{weapon_id}")
                    if weapon_details:
                        weapon_details['id'] = weapon_id
                        weapon_details['name'] = weapon_details.get('name', weapon_id)

                        try:
                            # Add refinement levels data if available
                            refinements = self._make_request(f"{self.secondary_api}/weapons/{weapon_id}/refinements")
                            if refinements:
                                weapon_details['refinements'] = refinements
                        except Exception as e:
                            print(f"  Warning: Failed to fetch refinements for {weapon_id}: {str(e)}")

                        try:
                            # Add ascension materials if available
                            ascension = self._make_request(f"{self.secondary_api}/weapons/{weapon_id}/ascension")
                            if ascension:
                                weapon_details['ascension_materials'] = ascension
                        except Exception as e:
                            print(f"  Warning: Failed to fetch ascension materials for {weapon_id}: {str(e)}")

                        weapon_data.append(weapon_details)
                except Exception as e:
                    print(f"  Error fetching details for weapon {weapon_id}: {str(e)}")
                    continue

            if weapon_data:
                self._save_json(weapon_data, "weapons.json")
                print(f"Saved {len(weapon_data)} weapons with refinements and ascension data")
            else:
                print("Warning: No weapon data was collected")

        except Exception as e:
            print(f"Error in fetch_weapons: {str(e)}")
            raise

    def fetch_domains(self):
        """Fetch domain data including rewards and requirements."""
        print("\nFetching domains...")

        # Get list of domains
        domains = self._make_request(f"{self.secondary_api}/domains")
        if not domains:
            print("Failed to fetch domain list")
            return

        domain_data = []
        for domain_id in domains:
            print(f"  Fetching domain: {domain_id}")
            domain_details = self._make_request(f"{self.secondary_api}/domains/{domain_id}")
            if domain_details:
                domain_details['id'] = domain_id
                domain_details['name'] = domain_details.get('name', domain_id)

                # Add reward details if available
                rewards = self._make_request(f"{self.secondary_api}/domains/{domain_id}/rewards")
                if rewards:
                    domain_details['rewards'] = rewards

                # Add requirements if available
                requirements = self._make_request(f"{self.secondary_api}/domains/{domain_id}/requirements")
                if requirements:
                    domain_details['requirements'] = requirements

                domain_data.append(domain_details)

        if domain_data:
            self._save_json(domain_data, "domains.json")
            print(f"Saved {len(domain_data)} domains with rewards and requirements")

    def fetch_character_ascension(self):
        """Fetch character ascension and leveling materials."""
        print("\nFetching character ascension data...")

        # Get list of characters
        characters = self._make_request(f"{self.secondary_api}/characters")
        if not characters:
            print("Failed to fetch character list")
            return

        ascension_data = []
        for char_id in characters:
            print(f"  Fetching ascension data for: {char_id}")

            # Get ascension materials
            ascension = self._make_request(f"{self.secondary_api}/characters/{char_id}/ascension")
            if ascension:
                ascension_info = {
                    'character': char_id,
                    'ascension_materials': ascension
                }

                # Get talent level-up materials if available
                talent_materials = self._make_request(f"{self.secondary_api}/characters/{char_id}/talent-materials")
                if talent_materials:
                    ascension_info['talent_materials'] = talent_materials

                ascension_data.append(ascension_info)

        if ascension_data:
            self._save_json(ascension_data, "character_ascension.json")
            print(f"Saved ascension data for {len(ascension_data)} characters")
        """Fetch weapon data."""
        print("\nFetching weapons...")

        # Get list of weapons
        weapons = self._make_request(f"{self.secondary_api}/weapons")
        if not weapons:
            print("Failed to fetch weapon list")
            return

        # Fetch details for each weapon
        weapon_data = []
        for weapon_id in weapons:
            print(f"  Fetching weapon: {weapon_id}")
            weapon_details = self._make_request(f"{self.secondary_api}/weapons/{weapon_id}")
            if weapon_details:
                weapon_details['id'] = weapon_id
                weapon_details['name'] = weapon_details.get('name', weapon_id)
                weapon_data.append(weapon_details)

        if weapon_data:
            self._save_json(weapon_data, "weapons.json")
            print(f"Saved {len(weapon_data)} weapons")
