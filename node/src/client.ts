import axios, { AxiosInstance } from 'axios';
import { Character, Artifact, Weapon, APIResponse } from './types';

export class GenshinClient {
  private client: AxiosInstance;
  private baseUrl: string;

  constructor() {
    this.baseUrl = 'https://genshin.jmp.blue';
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 10000,
    });
  }

  async getAllCharacters(): Promise<Character[]> {
    const response = await this.client.get<string[]>('/characters');
    const characters: Character[] = [];

    for (const charId of response.data) {
      try {
        const char = await this.getCharacter(charId);
        characters.push(char);
      } catch (error) {
        console.error(`Error fetching character ${charId}:`, error);
      }
    }

    return characters;
  }

  async getCharacter(id: string): Promise<Character> {
    const response = await this.client.get<Character>(`/characters/${id}`);
    return {
      ...response.data,
      id,
    };
  }

  async getAllArtifacts(): Promise<Artifact[]> {
    const response = await this.client.get<string[]>('/artifacts');
    const artifacts: Artifact[] = [];

    for (const artifactId of response.data) {
      try {
        const artifact = await this.getArtifact(artifactId);
        artifacts.push(artifact);
      } catch (error) {
        console.error(`Error fetching artifact ${artifactId}:`, error);
      }
    }

    return artifacts;
  }

  async getArtifact(id: string): Promise<Artifact> {
    const response = await this.client.get<Artifact>(`/artifacts/${id}`);
    return {
      ...response.data,
      id,
    };
  }

  async getAllWeapons(): Promise<Weapon[]> {
    const response = await this.client.get<string[]>('/weapons');
    const weapons: Weapon[] = [];

    for (const weaponId of response.data) {
      try {
        const weapon = await this.getWeapon(weaponId);
        weapons.push(weapon);
      } catch (error) {
        console.error(`Error fetching weapon ${weaponId}:`, error);
      }
    }

    return weapons;
  }

  async getWeapon(id: string): Promise<Weapon> {
    const response = await this.client.get<Weapon>(`/weapons/${id}`);
    return {
      ...response.data,
      id,
    };
  }
}
