import { Character, Artifact, Weapon } from './types';
export declare class GenshinClient {
    private client;
    private baseUrl;
    constructor();
    getAllCharacters(): Promise<Character[]>;
    getCharacter(id: string): Promise<Character>;
    getAllArtifacts(): Promise<Artifact[]>;
    getArtifact(id: string): Promise<Artifact>;
    getAllWeapons(): Promise<Weapon[]>;
    getWeapon(id: string): Promise<Weapon>;
}
