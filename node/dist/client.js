"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GenshinClient = void 0;
const axios_1 = __importDefault(require("axios"));
class GenshinClient {
    constructor() {
        this.baseUrl = 'https://genshin.jmp.blue';
        this.client = axios_1.default.create({
            baseURL: this.baseUrl,
            timeout: 10000,
        });
    }
    async getAllCharacters() {
        const response = await this.client.get('/characters');
        const characters = [];
        for (const charId of response.data) {
            try {
                const char = await this.getCharacter(charId);
                characters.push(char);
            }
            catch (error) {
                console.error(`Error fetching character ${charId}:`, error);
            }
        }
        return characters;
    }
    async getCharacter(id) {
        const response = await this.client.get(`/characters/${id}`);
        return {
            ...response.data,
            id,
        };
    }
    async getAllArtifacts() {
        const response = await this.client.get('/artifacts');
        const artifacts = [];
        for (const artifactId of response.data) {
            try {
                const artifact = await this.getArtifact(artifactId);
                artifacts.push(artifact);
            }
            catch (error) {
                console.error(`Error fetching artifact ${artifactId}:`, error);
            }
        }
        return artifacts;
    }
    async getArtifact(id) {
        const response = await this.client.get(`/artifacts/${id}`);
        return {
            ...response.data,
            id,
        };
    }
    async getAllWeapons() {
        const response = await this.client.get('/weapons');
        const weapons = [];
        for (const weaponId of response.data) {
            try {
                const weapon = await this.getWeapon(weaponId);
                weapons.push(weapon);
            }
            catch (error) {
                console.error(`Error fetching weapon ${weaponId}:`, error);
            }
        }
        return weapons;
    }
    async getWeapon(id) {
        const response = await this.client.get(`/weapons/${id}`);
        return {
            ...response.data,
            id,
        };
    }
}
exports.GenshinClient = GenshinClient;
