# API Reference

Complete reference for Katheryne's API client and endpoints.

## Table of Contents
- [Node.js Client](#nodejs-client)
- [TypeScript Types](#typescript-types)
- [API Endpoints](#api-endpoints)
- [Error Handling](#error-handling)

## Node.js Client

### Client Initialization

```typescript
import { GenshinClient } from './src/client';

const client = new GenshinClient({
  baseUrl?: string;
  timeout?: number;
  debug?: boolean;
});
```

### Character Methods

```typescript
// Get all characters
const characters = await client.getAllCharacters();

// Get specific character
const hutao = await client.getCharacter('hutao');

// Get character stats
const stats = await client.getCharacterStats('hutao');
```

### Weapon Methods

```typescript
// Get all weapons
const weapons = await client.getAllWeapons();

// Get specific weapon
const weapon = await client.getWeapon('staff-of-homa');

// Get weapon stats
const stats = await client.getWeaponStats('staff-of-homa');
```

### Artifact Methods

```typescript
// Get all artifacts
const artifacts = await client.getAllArtifacts();

// Get specific artifact set
const set = await client.getArtifactSet('crimson-witch');
```

## TypeScript Types

### Character Types

```typescript
interface Character {
  name: string;
  title: string;
  vision: ElementType;
  weapon: WeaponType;
  nation: NationType;
  affiliation: string;
  rarity: number;
  constellation: string;
  birthday: string;
  description: string;
  skillTalents: SkillTalent[];
  passiveTalents: PassiveTalent[];
  constellations: Constellation[];
}

interface SkillTalent {
  name: string;
  unlock: string;
  description: string;
  upgrades: SkillUpgrade[];
}

interface PassiveTalent {
  name: string;
  unlock: string;
  description: string;
}

interface Constellation {
  name: string;
  unlock: string;
  description: string;
  level: number;
}
```

### Weapon Types

```typescript
interface Weapon {
  name: string;
  type: WeaponType;
  rarity: number;
  baseAtk: number;
  subStat: string;
  passiveName: string;
  passiveDesc: string;
  location: string;
}

enum WeaponType {
  SWORD = 'SWORD',
  CLAYMORE = 'CLAYMORE',
  POLEARM = 'POLEARM',
  BOW = 'BOW',
  CATALYST = 'CATALYST'
}
```

### Artifact Types

```typescript
interface Artifact {
  name: string;
  max_rarity: number;
  two_piece_bonus: string;
  four_piece_bonus: string;
  pieces: ArtifactPiece[];
}

interface ArtifactPiece {
  name: string;
  relicType: RelicType;
}

enum RelicType {
  FLOWER = 'FLOWER',
  PLUME = 'PLUME',
  SANDS = 'SANDS',
  GOBLET = 'GOBLET',
  CIRCLET = 'CIRCLET'
}
```

## API Endpoints

### Character Endpoints

```typescript
GET /characters
GET /characters/{name}
GET /characters/{name}/stats
```

Response example:
```json
{
  "name": "Hu Tao",
  "vision": "PYRO",
  "weapon": "POLEARM",
  "nation": "LIYUE",
  "affiliation": "Wangsheng Funeral Parlor",
  "rarity": 5,
  "constellation": "Papilio Charontis",
  "birthday": "0000-07-15",
  "description": "The 77th Director of the Wangsheng Funeral Parlor..."
}
```

### Weapon Endpoints

```typescript
GET /weapons
GET /weapons/{name}
GET /weapons/{name}/stats
```

Response example:
```json
{
  "name": "Staff of Homa",
  "type": "POLEARM",
  "rarity": 5,
  "baseAtk": 46,
  "subStat": "CRIT_DMG",
  "passiveName": "Reckless Cinnabar",
  "passiveDesc": "HP increased by 20%. Additionally, provides an ATK Bonus..."
}
```

### Artifact Endpoints

```typescript
GET /artifacts
GET /artifacts/{name}
```

Response example:
```json
{
  "name": "Crimson Witch of Flames",
  "max_rarity": 5,
  "two_piece_bonus": "Pyro DMG Bonus +15%",
  "four_piece_bonus": "Increases Overloaded and Burning DMG by 40%..."
}
```

## Error Handling

### Error Types

```typescript
interface APIError {
  code: number;
  message: string;
  details?: any;
}

enum ErrorCode {
  NOT_FOUND = 404,
  RATE_LIMIT = 429,
  SERVER_ERROR = 500
}
```

### Error Handling Examples

```typescript
try {
  const character = await client.getCharacter('nonexistent');
} catch (error) {
  if (error instanceof APIError) {
    switch (error.code) {
      case ErrorCode.NOT_FOUND:
        console.error('Character not found');
        break;
      case ErrorCode.RATE_LIMIT:
        console.error('Rate limit exceeded');
        break;
      default:
        console.error('Unknown error:', error.message);
    }
  }
}
```

### Rate Limiting

The client implements automatic rate limiting:
- Default: 60 requests per minute
- Retries with exponential backoff
- Configurable limits:

```typescript
const client = new GenshinClient({
  rateLimit: {
    maxRequests: 60,
    perMinute: 1,
    retryAfter: 1000
  }
});
```

## Pagination

For endpoints that return lists:

```typescript
const options = {
  page: 1,
  limit: 20,
  sort: 'name',
  order: 'asc'
};

const characters = await client.getAllCharacters(options);
```

Response includes pagination metadata:
```typescript
interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    current: number;
    total: number;
    perPage: number;
    totalPages: number;
  }
}
```

## WebSocket API (if applicable)

```typescript
const ws = client.connectWebSocket();

ws.on('message', (data) => {
  console.log('Received:', data);
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## Authentication (if needed)

```typescript
const client = new GenshinClient({
  apiKey: 'your-api-key',
  // or
  auth: {
    username: 'user',
    password: 'pass'
  }
});
```

## Caching

The client implements automatic caching:

```typescript
const client = new GenshinClient({
  cache: {
    enabled: true,
    ttl: 3600, // 1 hour
    maxSize: 1000 // entries
  }
});
```

## Logging

Enable detailed logging:

```typescript
const client = new GenshinClient({
  debug: true,
  logLevel: 'debug', // 'error' | 'warn' | 'info' | 'debug'
  logger: customLogger // optional
});
```