import axios from 'axios';
import { Character, DetailedCharacter, Artifact, Weapon } from '../types';

export class GenshinDevClient {
  private baseUrl = 'https://genshin.dev/api/v1';
  private client = axios.create({
    baseURL: this.baseUrl,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json'
    }
  });

  async getAllCharacters(): Promise<Character[]> {
    try {
      const response = await this.client.get<Character[]>('/characters');
      return response.data.map(char => ({
        ...char,
        id: char.name.toLowerCase().replace(/ /g, '-'),
        description: char.description || 'No description available'
      }));
    } catch (error) {
      console.error('Error fetching characters:', error);
      return [];
    }
  }

  async getCharacter(characterId: string): Promise<DetailedCharacter | null> {
    try {
      const response = await this.client.get<DetailedCharacter>(
        `/characters/${characterId}`
      );
      const data = response.data;
      return {
        ...data,
        id: data.name.toLowerCase().replace(/ /g, '-'),
        description: data.description || 'No description available',
        skills: data.skills || [],
        constellations: data.constellations || [],
        talents: data.talents || []
      };
    } catch (error) {
      console.error(`Error fetching character ${characterId}:`, error);
      return null;
    }
  }

  async getAllArtifacts(): Promise<Artifact[]> {
    try {
      const response = await this.client.get<Artifact[]>('/artifacts');
      return response.data.map(artifact => ({
        ...artifact,
        id: artifact.name.toLowerCase().replace(/ /g, '-'),
        description: artifact.description || 'No description available'
      }));
    } catch (error) {
      console.error('Error fetching artifacts:', error);
      return [];
    }
  }

  async getWeapon(weaponId: string): Promise<Weapon | null> {
    try {
      const response = await this.client.get<Weapon>(`/weapons/${weaponId}`);
      const data = response.data;
      return {
        ...data,
        id: data.name.toLowerCase().replace(/ /g, '-'),
        description: data.description || 'No description available'
      };
    } catch (error) {
      console.error(`Error fetching weapon ${weaponId}:`, error);
      return null;
