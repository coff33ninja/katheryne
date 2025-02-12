// Common types used across both APIs
export interface Character {
    name: string;
    element: string;
    weapon: string;
    rarity: number;
}

export interface DetailedCharacter extends Character {
    skills: {
        normal_attack: string;
        elemental_skill: string;
        elemental_burst: string;
        passive_skills: string[];
    };
}

export interface Artifact {
    name: string;
    bonuses: string[];
}

export interface Weapon {
    name: string;
    type: string;
    rarity: number;
    base_attack: number;
    secondary_stat: string;
    passive: string;
}

// API specific types
export type GenshinJmpBlueTypes = 
    | "artifacts"
    | "boss"
    | "characters"
    | "consumables"
    | "domains"
    | "elements"
    | "enemies"
    | "materials"
    | "nations"
    | "weapons";