import requests
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenshinDataDownloader:
    """Downloads and manages Genshin Impact data from various APIs"""
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.endpoints = {
            "characters": "https://api.genshin.dev/characters",
            "weapons": "https://api.genshin.dev/weapons",
            "artifacts": "https://api.genshin.dev/artifacts",
            "enemies": "https://api.genshin.dev/enemies",
            "domains": "https://api.genshin.dev/domains"
        }
        
        # Ambr.top API (for detailed data)
        self.ambr_base_url = "https://api.ambr.top/v2/en"
        
        # Initialize cache
        self.cache_file = self.data_dir / "api_cache.json"
        self.cache: Dict = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cached API responses"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted, creating new cache")
                return {}
        return {}

    def _save_cache(self):
        """Save current cache to file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _make_request(self, url: str, cache_key: str, force_refresh: bool = False) -> Optional[Dict]:
        """Make an API request with caching"""
        if not force_refresh and cache_key in self.cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.cache[cache_key]
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.cache[cache_key] = data
            self._save_cache()
            return data
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def download_all(self, force_refresh: bool = False, callback=None) -> bool:
        """Download all data from APIs"""
        success = True
        total_endpoints = len(self.endpoints)
        
        for i, (data_type, endpoint) in enumerate(self.endpoints.items(), 1):
            if callback:
                callback(f"Downloading {data_type} data... ({i}/{total_endpoints})")
            
            data = self._make_request(endpoint, data_type, force_refresh)
            if data is None:
                success = False
                continue
            
            # Save to separate file
            output_file = self.data_dir / f"{data_type}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # If it's a list of items, fetch details for each
            if isinstance(data, list):
                details_dir = self.data_dir / data_type
                details_dir.mkdir(exist_ok=True)
                
                for item in tqdm(data, desc=f"Fetching {data_type} details"):
                    detail_url = f"{endpoint}/{item}"
                    detail_data = self._make_request(
                        detail_url,
                        f"{data_type}_{item}",
                        force_refresh
                    )
                    if detail_data:
                        detail_file = details_dir / f"{item}.json"
                        with open(detail_file, 'w', encoding='utf-8') as f:
                            json.dump(detail_data, f, indent=2, ensure_ascii=False)
            
            # Rate limiting
            time.sleep(1)
        
        if callback:
            callback("Download complete!")
        return success

    def download_character_data(self, force_refresh: bool = False, callback=None) -> bool:
        """Download character-specific data"""
        if callback:
            callback("Downloading character data...")
        
        data = self._make_request(self.endpoints["characters"], "characters", force_refresh)
        if data is None:
            return False
        
        characters_dir = self.data_dir / "characters"
        characters_dir.mkdir(exist_ok=True)
        
        for character in tqdm(data, desc="Fetching character details"):
            if callback:
                callback(f"Downloading data for {character}...")
            
            # Get basic character data
            char_url = f"{self.endpoints['characters']}/{character}"
            char_data = self._make_request(
                char_url,
                f"character_{character}",
                force_refresh
            )
            
            if char_data:
                char_file = characters_dir / f"{character}.json"
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data, f, indent=2, ensure_ascii=False)
            
            # Rate limiting
            time.sleep(0.5)
        
        if callback:
            callback("Character data download complete!")
        return True

    def download_weapon_data(self, force_refresh: bool = False, callback=None) -> bool:
        """Download weapon-specific data"""
        if callback:
            callback("Downloading weapon data...")
        
        data = self._make_request(self.endpoints["weapons"], "weapons", force_refresh)
        if data is None:
            return False
        
        weapons_dir = self.data_dir / "weapons"
        weapons_dir.mkdir(exist_ok=True)
        
        for weapon in tqdm(data, desc="Fetching weapon details"):
            if callback:
                callback(f"Downloading data for {weapon}...")
            
            weapon_url = f"{self.endpoints['weapons']}/{weapon}"
            weapon_data = self._make_request(
                weapon_url,
                f"weapon_{weapon}",
                force_refresh
            )
            
            if weapon_data:
                weapon_file = weapons_dir / f"{weapon}.json"
                with open(weapon_file, 'w', encoding='utf-8') as f:
                    json.dump(weapon_data, f, indent=2, ensure_ascii=False)
            
            time.sleep(0.5)
        
        if callback:
            callback("Weapon data download complete!")
        return True

    def download_artifact_data(self, force_refresh: bool = False, callback=None) -> bool:
        """Download artifact set data"""
        if callback:
            callback("Downloading artifact data...")
        
        data = self._make_request(self.endpoints["artifacts"], "artifacts", force_refresh)
        if data is None:
            return False
        
        artifacts_dir = self.data_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        for artifact in tqdm(data, desc="Fetching artifact details"):
            if callback:
                callback(f"Downloading data for {artifact}...")
            
            artifact_url = f"{self.endpoints['artifacts']}/{artifact}"
            artifact_data = self._make_request(
                artifact_url,
                f"artifact_{artifact}",
                force_refresh
            )
            
            if artifact_data:
                artifact_file = artifacts_dir / f"{artifact}.json"
                with open(artifact_file, 'w', encoding='utf-8') as f:
                    json.dump(artifact_data, f, indent=2, ensure_ascii=False)
            
            time.sleep(0.5)
        
        if callback:
            callback("Artifact data download complete!")
        return True

    def verify_data(self, callback=None) -> Dict[str, bool]:
        """Verify that all necessary data files exist"""
        verification = {}
        
        # Check main data files
        for data_type in self.endpoints.keys():
            main_file = self.data_dir / f"{data_type}.json"
            verification[data_type] = main_file.exists()
            
            # Check detail directories
            details_dir = self.data_dir / data_type
            if details_dir.exists():
                files = list(details_dir.glob("*.json"))
                verification[f"{data_type}_details"] = len(files) > 0
            else:
                verification[f"{data_type}_details"] = False
        
        if callback:
            missing = [k for k, v in verification.items() if not v]
            if missing:
                callback(f"Missing data: {', '.join(missing)}")
            else:
                callback("All data files present!")
        
        return verification