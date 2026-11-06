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
    // 1. Vectorize the question
    const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-2" });
    const embedResult = await embeddingModel.embedContent(req.body.message);

    // 2. Query Pinecone
    const index = pc.index('sip-discovery-3072');
    const queryResponse = await index.query({
      vector: embedResult.embedding.values,
      topK: 30,
      includeMetadata: true,
    });

    // 3. Format the context
    const context = queryResponse.matches.map(m =>
      `Review: ${m.metadata?.text}\nTags: Segment: ${m.metadata?.user_segment}, Trigger: ${m.metadata?.drop_off_trigger}`
    ).join('\n\n');

    // 4. Generate the answer with fallback handling
    const prompt = `You are a Principal Product Analyst. Answer the user query using only this context. Quantify your findings using the provided Tags.\n\nContext:\n${context}\n\nQuestion: ${req.body.message}`;

    let result;
    try {
      const primaryModel = genAI.getGenerativeModel({ model: "gemini-3.8-flash" });
      result = await primaryModel.generateContent(prompt);
    } catch (genError) {
      // If 3.8 is experiencing high demand (503), fall back to 3.5-flash-lite
      if (genError.message && (genError.message.includes('503') || genError.message.includes('demand'))) {
        console.warn("gemini-3.8-flash overloaded; routing to gemini-3.5-flash-lite fallback.");
        const fallbackModel = genAI.getGenerativeModel({ model: "gemini-3.5-flash-lite" });
        result = await fallbackModel.generateContent(prompt);
      } else {
        throw genError;
      }
    }

    return res.status(200).json({ answer: result.response.text() });

  } catch (error) {
    console.error("Backend Crash Details:", error);
    return res.status(500).json({ error: error.message });
  }
}