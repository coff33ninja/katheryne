from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class GenshinDataProcessor:
    """Process and prepare Genshin Impact data for AI training."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.encoders = {}
        self.scalers = {}
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process all data for AI training."""
        processed = {}
        
        # Process each data type if the file exists
        for data_type in ['characters', 'artifacts', 'weapons']:
            json_file = self.data_dir / "raw" / f"{data_type}.json"
            if json_file.exists():
                print(f"Processing {data_type}...")
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data:  # Only process if we have data
                            if data_type == 'characters':
                                df = self._process_characters(data)
                            elif data_type == 'artifacts':
                                df = self._process_artifacts(data)
                            elif data_type == 'weapons':
                                df = self._process_weapons(data)
                            
                            # Save processed data if we have valid features
                            if not df.empty and df.select_dtypes(include=['int64', 'float64']).columns.size > 0:
                                processed[data_type] = df
                                parquet_path = self.processed_dir / f"{data_type}.parquet"
                                df.to_parquet(parquet_path)
                                print(f"Processed {data_type}:")
                                print(f"- Shape: {df.shape}")
                                print(f"- Numeric columns: {list(df.select_dtypes(include=['int64', 'float64']).columns)}")
                                print(f"- Sample values:\n{df.head()}\n")
                            else:
                                print(f"Warning: No valid numerical features found for {data_type}")
                except Exception as e:
                    print(f"Error processing {data_type}: {e}")
                    continue
            else:
                print(f"Warning: No data file found at {json_file}")
        
        return processed

    def _extract_numeric_from_text(self, text: str) -> float:
        """Extract numeric value from text description."""
        if not isinstance(text, str):
            return 0.0
        matches = re.findall(r'(\d+(?:\.\d+)?)', text)
        return float(matches[0]) if matches else 0.0

    def _process_characters(self, characters: List[Dict]) -> pd.DataFrame:
        """Process character data into a format suitable for ML."""
        if not characters:
            return pd.DataFrame()
            
        df = pd.DataFrame(characters)
        
        # Encode categorical variables
        categorical_cols = ['element', 'weapon', 'region', 'vision']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self._encode_categorical(df[col], col)
        
        # Convert basic numeric columns
        numeric_cols = ['rarity', 'level', 'friendship', 'constellation']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract stats if available
        stats_cols = ['base_hp', 'base_attack', 'base_defense', 'crit_rate', 'crit_dmg']
        for col in stats_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna(0)
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if numeric_df.empty:
            print("Warning: No numeric columns found in character data")
            return pd.DataFrame()
        
        return numeric_df

    def _process_artifacts(self, artifacts: List[Dict]) -> pd.DataFrame:
        """Process artifact data for ML."""
        if not artifacts:
            return pd.DataFrame()
            
        df = pd.DataFrame(artifacts)
        
        # Basic numeric features
        if 'rarity' in df.columns:
            df['rarity'] = pd.to_numeric(df['rarity'], errors='coerce')
        
        # Extract bonus values
        if '2-piece_bonus' in df.columns:
            df['bonus_2pc_value'] = df['2-piece_bonus'].apply(self._extract_numeric_from_text)
        if '4-piece_bonus' in df.columns:
            df['bonus_4pc_value'] = df['4-piece_bonus'].apply(self._extract_numeric_from_text)
        
        # Extract stats if available
        stats_cols = ['hp', 'attack', 'defense', 'elemental_mastery', 'energy_recharge']
        for col in stats_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna(0)
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if numeric_df.empty:
            print("Warning: No numeric columns found in artifact data")
            return pd.DataFrame()
        
        return numeric_df

    def _process_weapons(self, weapons: List[Dict]) -> pd.DataFrame:
        """Process weapon data for ML."""
        if not weapons:
            return pd.DataFrame()
            
        df = pd.DataFrame(weapons)
        
        # Encode weapon type
        if 'type' in df.columns:
            df['type'] = self._encode_categorical(df['type'], 'weapon_type')
        
        # Basic numeric features
        numeric_cols = ['rarity', 'base_attack', 'sub_stat_value']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract refinement values if available
        if 'refinement' in df.columns:
            for i in range(1, 6):  # R1 to R5
                col = f'refinement_{i}'
                if col in df.columns:
                    df[col] = df[col].apply(self._extract_numeric_from_text)
        
        # Extract passive effect values
        if 'passive_desc' in df.columns:
            df['passive_value'] = df['passive_desc'].apply(self._extract_numeric_from_text)
        
        # Fill missing values
        df = df.fillna(0)
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if numeric_df.empty:
            print("Warning: No numeric columns found in weapon data")
            return pd.DataFrame()
        
        return numeric_df

    def _encode_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Encode categorical variables."""
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        series = series.fillna('unknown').astype(str)
        
        if column_name not in self.encoders:
            self.encoders[column_name] = LabelEncoder()
            return pd.Series(self.encoders[column_name].fit_transform(series))
        return pd.Series(self.encoders[column_name].transform(series))