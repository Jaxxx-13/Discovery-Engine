import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method Not Allowed' });

  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

  try {
    // 1. Vectorize the question (Fixed model name)
    const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
    const embedResult = await embeddingModel.embedContent(req.body.message);

    // 2. Search Pinecone for top 30 relevant reviews (Fixed index dimensions)
    const index = pc.index('sip-discovery-768');
    const queryResponse = await index.query({
      vector: embedResult.embedding.values,
      topK: 30,
      includeMetadata: true,
    });

    // 3. Format the context
    const context = queryResponse.matches.map(m =>
      `Review: ${m.metadata?.text}\nTags: Segment: ${m.metadata?.user_segment}, Trigger: ${m.metadata?.drop_off_trigger}`
    ).join('\n\n');

    // 4. Generate the answer (Added -latest for API stability)
    const chatModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
    const prompt = `You are a Principal Product Analyst. Answer the user query using only this context. Quantify your findings using the provided Tags.\n\nContext:\n${context}\n\nQuestion: ${req.body.message}`;

    const result = await chatModel.generateContent(prompt);
    res.status(200).json({ answer: result.response.text() });
  } catch (error) {
    console.error("Backend Crash:", error); // This prints it to the Vercel dashboard
    res.status(500).json({ error: error.message });
  }
}