import axios from 'axios';
import { Character, DetailedCharacter, Artifact, Weapon } from '../types';

export class GenshinDevClient {
  private baseUrl = 'https://genshin.dev/api/v1';

  constructor() {
    // Initialize axios instance if needed
  }

  async getAllCharacters(): Promise<Character[]> {
    const response = await axios.get<Character[]>(`${this.baseUrl}/characters`);
    return response.data;
  }

  async getCharacter(characterId: string): Promise<DetailedCharacter> {
    const response = await axios.get<DetailedCharacter>(
      `${this.baseUrl}/characters/${characterId}`,
    );
    return response.data;
  }

  async getAllArtifacts(): Promise<Artifact[]> {
    const response = await axios.get<Artifact[]>(`${this.baseUrl}/artifacts`);
    return response.data;
  }

  async getWeapon(weaponId: string): Promise<Weapon> {
    const response = await axios.get<Weapon>(
      `${this.baseUrl}/weapons/${weaponId}`,
    );
    return response.data;
  }
}
