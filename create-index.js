import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({ apiKey: 'pcsk_paVxn_CgtTEHUuYJtJLypJmmGHBoM4Y55kUukmM1oJJLoxBqZo4cgBmjccvhMSzZFrBWq' });

await pc.indexes.create({
  name: 'sip-discovery-3072',
  dimension: 3072,
  metric: 'cosine',
  spec: {
    serverless: {
      cloud: 'aws',
      region: 'us-east-1'
    }
  }
});

console.log("3072-dimension index created successfully!");