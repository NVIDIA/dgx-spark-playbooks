import { defineToolPlugin } from 'openclaw/plugin-sdk/tool-plugin';
import { CONFIG_SCHEMA, HomeClient } from './client.js';
import { createTools } from './tools.js';

export default defineToolPlugin({
  id: 'living-home', name: 'Living Home', description: 'Local plans, property records, and selected Home Assistant status',
  activation: { onStartup: true }, configSchema: CONFIG_SCHEMA,
  tools: tool => createTools().map(definition => tool({
    name: definition.name, label: definition.label, description: definition.description, parameters: definition.parameters,
    factory: ({ config }) => createTools(new HomeClient(config)).find(item => item.name === definition.name)       ,
  })),
});


//# sourceURL=index.ts