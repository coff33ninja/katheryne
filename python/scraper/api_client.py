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

        # API endpoints
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

    def fetch_all_data(self) -> Dict[str, List]:
        """Fetch all data from APIs."""
        print("Fetching data from Genshin Impact APIs...")
        data = {}

        # First fetch core types with dedicated methods
        self.fetch_characters()
        self.fetch_artifacts()
        self.fetch_weapons()

        # Then fetch remaining types
        remaining_types = [t for t in self.known_types if t not in ['characters', 'artifacts', 'weapons']]
        for data_type in remaining_types:
            print(f"\nFetching {data_type}...")
            items = self._make_request(f"{self.secondary_api}/{data_type}")

            if not items:
                print(f"No data received for {data_type}")
                continue

            if isinstance(items, list):
                # If we got a list of strings (names/ids), fetch details for each
                detailed_items = []
                for item_id in items:
                    print(f"  Fetching details for {item_id}...")
                    item_details = self._make_request(f"{self.secondary_api}/{data_type}/{item_id}")
                    if item_details:  # Only add if we got valid data
                        if isinstance(item_details, dict):
                            # Add identifier information
                            item_details['id'] = item_id
                            item_details['name'] = item_details.get('name', item_id)
                        detailed_items.append(item_details)
                data[data_type] = detailed_items
            elif isinstance(items, dict):
                # If we got a dictionary, store it directly
                data[data_type] = [items]
            else:
                print(f"Unexpected response type for {data_type}: {type(items)}")
                continue

            # Save individual type data
            if data[data_type]:
                self._save_json(data[data_type], f"{data_type}.json")
                print(f"Saved {len(data[data_type])} {data_type}")

        print("\nData fetching completed!")
        return data

    def _make_request(self, url: str) -> Optional[Any]:
        """Make an API request with rate limiting and error handling."""
        try:
            time.sleep(self.request_delay)  # Rate limiting
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            # Attempt to use the secondary API if the primary fails
            if url.startswith(self.primary_api):
                secondary_url = url.replace(self.primary_api, self.secondary_api)
                print(f"Trying secondary API: {secondary_url}")
                try:
                    time.sleep(self.request_delay)  # Rate limiting
                    response = requests.get(secondary_url, timeout=10)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {secondary_url}: {str(e)}")
                    print("Failed to fetch data from both primary and secondary APIs.")
                    return None
            print("Failed to fetch data from the primary API.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {url}: {str(e)}")
            return None

    def _save_json(self, data: Union[Dict, List], filename: str):
        """Save data to JSON file."""
        filepath = self.raw_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {filepath}")

    def fetch_characters(self):
        """Fetch character data."""
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
            char_details = self._make_request(f"{self.secondary_api}/characters/{char_id}")
            if char_details:
                char_details['id'] = char_id
                char_details['name'] = char_details.get('name', char_id)
                character_data.append(char_details)

        if character_data:
            self._save_json(character_data, "characters.json")
            print(f"Saved {len(character_data)} characters")

    def fetch_artifacts(self):
        """Fetch artifact data."""
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
                artifact_data.append(artifact_details)

        if artifact_data:
            self._save_json(artifact_data, "artifacts.json")
            print(f"Saved {len(artifact_data)} artifacts")

    def fetch_weapons(self):
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
