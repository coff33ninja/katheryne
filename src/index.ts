import './server';
import { GenshinDevClient } from './clients/genshinDev';
import { GenshinJmpBlueClient } from './clients/genshinJmpBlue';

console.log('Katheryne API Integration Started with AI Training Endpoint');

// Export clients for external use
export { GenshinDevClient, GenshinJmpBlueClient };