import { GenshinJmpBlueClient } from '../../clients/genshinJmpBlue';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('GenshinJmpBlueClient', () => {
  let client: GenshinJmpBlueClient;

  beforeEach(() => {
    client = new GenshinJmpBlueClient();
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  describe('getCharacterInfo', () => {
    it('should fetch character info successfully', async () => {
      const mockCharacterInfo = {
        name: 'Hu Tao',
        element: 'Pyro',
        weapontype: 'Polearm',
        region: 'Liyue',
        rarity: 5,
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockCharacterInfo });

      const result = await client.getCharacterInfo('hutao');

      expect(result).toEqual(mockCharacterInfo);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://gsi.jmp.blue/characters/hutao'
      );
      expect(mockedAxios.get).toHaveBeenCalledTimes(1);
    });

    it('should handle errors when fetching character info', async () => {
      const errorMessage = 'Character not found';
      mockedAxios.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(client.getCharacterInfo('nonexistent')).rejects.toThrow(
        errorMessage
      );
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://gsi.jmp.blue/characters/nonexistent'
      );
    });
  });

  describe('getWeaponInfo', () => {
    it('should fetch weapon info successfully', async () => {
      const mockWeaponInfo = {
        name: 'Staff of Homa',
        type: 'Polearm',
        rarity: 5,
        baseattack: 608,
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockWeaponInfo });

      const result = await client.getWeaponInfo('staff-of-homa');

      expect(result).toEqual(mockWeaponInfo);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://gsi.jmp.blue/weapons/staff-of-homa'
      );
      expect(mockedAxios.get).toHaveBeenCalledTimes(1);
    });

    it('should handle errors when fetching weapon info', async () => {
      const errorMessage = 'Weapon not found';
      mockedAxios.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(client.getWeaponInfo('nonexistent')).rejects.toThrow(
        errorMessage
      );
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'https://gsi.jmp.blue/weapons/nonexistent'
      );
    });
  });
});