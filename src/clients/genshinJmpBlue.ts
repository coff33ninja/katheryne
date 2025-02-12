import axios from 'axios';
import { GenshinJmpBlueTypes } from '../types';

export class GenshinJmpBlueClient {
    private baseUrl = 'https://genshin.jmp.blue';

    constructor() {
        // Initialize axios instance if needed
    }

    async getTypes(): Promise<GenshinJmpBlueTypes[]> {
        const response = await axios.get<GenshinJmpBlueTypes[]>(this.baseUrl);
        return response.data;
    }

    async getData<T>(type: GenshinJmpBlueTypes): Promise<T> {
        const response = await axios.get<T>(`${this.baseUrl}/${type}`);
        return response.data;
    }

    async getSpecificData<T>(type: GenshinJmpBlueTypes, id: string): Promise<T> {
        const response = await axios.get<T>(`${this.baseUrl}/${type}/${id}`);
        return response.data;
    }
}