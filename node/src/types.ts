export interface Character {
  name: string;
  element: string;
  weapon: string;
  rarity: number;
  id?: string;
  description?: string;
}

export interface Artifact {
  name: string;
  rarity: number;
  '2-piece_bonus'?: string;
  '4-piece_bonus'?: string;
  id?: string;
}

export interface Weapon {
  name: string;
  type: string;
  rarity: number;
  baseAttack: number;
  subStat?: string;
  passiveDesc?: string;
  id?: string;
}

export interface APIResponse<T> {
  data: T;
  status: number;
  message?: string;
}
