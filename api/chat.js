import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'API key is missing on the server' });
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

  try {
    // 1. Vectorize the question (gemini-embedding-2 outputs 3072 dimensions)
    const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-2" });
    const embedResult = await embeddingModel.embedContent(req.body.message);

    // 2. Query Pinecone (matches 3072-dimension index)
    const index = pc.index('sip-discovery-3072');
    const queryResponse = await index.query({
      vector: embedResult.embedding.values,
      topK: 30,
      includeMetadata: true,
    });

    // 3. Format the retrieved context
    const context = queryResponse.matches.map(m =>
      `Review: ${m.metadata?.text}\nTags: Segment: ${m.metadata?.user_segment}, Trigger: ${m.metadata?.drop_off_trigger}`
    ).join('\n\n');

    // 4. Generate the response with active model
    const chatModel = genAI.getGenerativeModel({ model: "gemini-3.8-flash" });
    const prompt = `You are a Principal Product Analyst. Answer the user query using only this context. Quantify your findings using the provided Tags.\n\nContext:\n${context}\n\nQuestion: ${req.body.message}`;

    const result = await chatModel.generateContent(prompt);
    return res.status(200).json({ answer: result.response.text() });

  } catch (error) {
    console.error("Backend Crash Details:", error);
    return res.status(500).json({ error: error.message });
  }
}