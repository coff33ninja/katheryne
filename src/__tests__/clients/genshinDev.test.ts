import { GenshinDevClient } from '../../clients/genshinDev';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('GenshinDevClient', () => {
  let client: GenshinDevClient;

  beforeEach(() => {
    client = new GenshinDevClient();
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  describe('getCharacter', () => {
    it('should fetch character data successfully', async () => {
      const mockCharacterData = {
        name: 'Hu Tao',
        vision: 'Pyro',
        weapon: 'Polearm',
        nation: 'Liyue',
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockCharacterData });

      const result = await client.getCharacter('hutao');

      expect(result).toEqual(mockCharacterData);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://api.genshin.dev/characters/hutao'
      );
      expect(mockedAxios.get).toHaveBeenCalledTimes(1);
    });

    it('should handle errors when fetching character data', async () => {
      const errorMessage = 'Character not found';
      mockedAxios.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(client.getCharacter('nonexistent')).rejects.toThrow(
        errorMessage
      );
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://api.genshin.dev/characters/nonexistent'
      );
    });
  });

  describe('getWeapon', () => {
    it('should fetch weapon data successfully', async () => {
      const mockWeaponData = {
        name: 'Staff of Homa',
        type: 'Polearm',
        rarity: 5,
        baseAttack: 608,
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockWeaponData });

      const result = await client.getWeapon('staff-of-homa');

      expect(result).toEqual(mockWeaponData);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://api.genshin.dev/weapons/staff-of-homa'
      );
      expect(mockedAxios.get).toHaveBeenCalledTimes(1);
    });

    it('should handle errors when fetching weapon data', async () => {
      const errorMessage = 'Weapon not found';
      mockedAxios.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(client.getWeapon('nonexistent')).rejects.toThrow(errorMessage);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://api.genshin.dev/weapons/nonexistent'
      );
    });
  });
});