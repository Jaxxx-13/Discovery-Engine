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

    // 2. Query Pinecone (3072 dimensions)
    const index = pc.index('sip-discovery-3072');
    const queryResponse = await index.query({
      vector: embedResult.embedding.values,
      topK: 30,
      includeMetadata: true,
    });

    // 3. Format the context cleanly
    const context = queryResponse.matches.map((m, i) => {
      const text = m.metadata?.text || 'No review text';
      const segment = m.metadata?.user_segment || m.metadata?.segment || 'Unspecified';
      const trigger = m.metadata?.drop_off_trigger || m.metadata?.trigger || 'Unspecified';
      return `[Review ${i + 1}]: ${text}\nTags: Segment: ${segment}, Trigger: ${trigger}`;
    }).join('\n\n');

    // 4. Clean, focused system prompt
    const prompt = `You are an automated insights engine for an investment platform.

Tone and Persona Instructions:
- Never begin with "As a Principal Product Analyst..." or any self-referential titles.
- Start directly by referencing the evidence: use phrases like "On the basis of the data procured...", "Based on user retention trends...", or jump straight into the analysis.
- Exclude all device-specific, hardware-level, or operating system launch crashes (do NOT discuss specific mobile models like Samsung Galaxy M13, Android versions, cache wiping, or app update install bugs).
- Focus strictly on:
  1. Investor psychology (market volatility, loss aversion, panic during drawdowns).
  2. Financial/cash-flow constraints (liquidity crunches, lack of pause options).
  3. Platform UX, mandate management, transparency, and portfolio clarity.

Retrieved Review Context:
${context}

User Question: ${req.body.message}`;

    // 5. Generate content with automatic 503 fallback
    let responseText;
    try {
      const primaryModel = genAI.getGenerativeModel({ model: "gemini-3.8-flash" });
      const result = await primaryModel.generateContent(prompt);
      responseText = result.response.text();
    } catch (genError) {
      if (genError.message && (genError.message.includes('503') || genError.message.includes('demand'))) {
        console.warn("Primary model 3.8-flash overloaded; switching to 3.5-flash-lite fallback.");
        const fallbackModel = genAI.getGenerativeModel({ model: "gemini-3.5-flash-lite" });
        const fallbackResult = await fallbackModel.generateContent(prompt);
        responseText = fallbackResult.response.text();
      } else {
        throw genError;
      }
    }

    return res.status(200).json({ answer: responseText });

  } catch (error) {
    console.error("Backend Crash Details:", error);
    return res.status(500).json({ error: error.message });
  }
}