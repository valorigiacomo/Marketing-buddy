import { GoogleGenAI, GenerateContentResponse, Chat } from "@google/genai";

const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
const ai = new GoogleGenAI({ apiKey: apiKey });

export interface MarketingAnalysis {
  pros: string[];
  cons: string[];
  swot: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  businessPlan: string;
}

export async function analyzeMarketingIdea(idea: string, category: string): Promise<MarketingAnalysis> {
  const response = await ai.models.generateContent({
    model: "gemini-3.1-flash",
    contents: `Analizza la seguente idea di marketing per un progetto di tipo "${category}": "${idea}". 
    Fornisci un'analisi SWOT dettagliata, una lista di pro e contro e una breve struttura di business plan.
    Rispondi in ITALIANO.
    Restituisci la risposta in formato JSON con la seguente struttura:
    {
      "pros": ["string"],
      "cons": ["string"],
      "swot": {
        "strengths": ["string"],
        "weaknesses": ["string"],
        "opportunities": ["string"],
        "threats": ["string"]
      },
      "businessPlan": "stringa in formato markdown"
    }`,
    config: {
      responseMimeType: "application/json",
    },
  });

  try {
    return JSON.parse(response.text || "{}");
  } catch (e) {
    console.error("Failed to parse analysis JSON", e);
    throw new Error("Errore nell'analisi dell'idea");
  }
}

export async function analyzeLocation(location: string, idea: string, category: string, radius: number, businessType: string, lat?: number, lng?: number) {
  try {
    const query = businessType === 'all' 
      ? `Identifica le principali attività commerciali (bar, ristoranti, meccanici, negozi, centri sportivi) entro un raggio di ${radius}km da ${location} per supportare un progetto di tipo "${category}" con l'idea: "${idea}".`
      : `Identifica i principali ${businessType} entro un raggio di ${radius}km da ${location} per supportare un progetto di tipo "${category}" con l'idea: "${idea}".`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `${query} 
      Seleziona esclusivamente le attività con la MAGGIORE PROBABILITÀ di essere interessate a una sponsorizzazione basandoti sulla pertinenza con l'idea.
      Fornisci un elenco puntato dei potenziali sponsor, indicando per ognuno il motivo specifico per cui dovrebbero partecipare.
      Rispondi in ITALIANO. Sii concreto e diretto.`,
      config: {
        tools: [{ googleMaps: {} }],
        toolConfig: {
          retrievalConfig: {
            latLng: lat && lng ? { latitude: lat, longitude: lng } : undefined
          }
        }
      },
    });

    if (!response.text) {
      throw new Error("Nessuna risposta ricevuta dal servizio di mappe.");
    }

    return {
      text: response.text,
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks || []
    };
  } catch (error: any) {
    console.error("Errore in analyzeLocation:", error);
    throw new Error(error.message || "Errore durante la ricerca su Google Maps.");
  }
}

export async function findSponsorsAI(idea: string, category: string, location: string) {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Analizza l'idea: "${idea}" (categoria: ${category}) a "${location}".
      Identifica i 3-5 profili di sponsor ideali (categorie di aziende).
      Per ogni categoria fornisci:
      1. MOTIVAZIONE: Perché è il partner perfetto.
      2. STRATEGIA: Consiglio esperto su come approcciarli (breve e schematico).
      Usa un formato elenco puntato molto sintetico e professionale.
      Rispondi in ITALIANO.`,
    });

    return response.text || "Nessuna analisi generata.";
  } catch (error: any) {
    console.error("Errore in findSponsorsAI:", error);
    throw new Error(error.message || "Errore durante l'analisi degli sponsor AI.");
  }
}

export function createMarketingChat(): Chat {
  return ai.chats.create({
    model: "gemini-3-flash-preview",
    config: {
      systemInstruction: `Sei un esperto di marketing di livello mondiale con una profonda conoscenza del mercato italiano. 
      Fornisci consulenza strategica, idee creative e suggerimenti pratici. 
      È FONDAMENTALE che i tuoi consigli tengano conto della normativa italiana vigente, inclusi ma non limitati a:
      - GDPR e normativa sulla privacy italiana (Garante Privacy).
      - Codice del Consumo italiano.
      - Normative sulla pubblicità e sull'e-commerce in Italia.
      - Regole dell'AGCM (Autorità Garante della Concorrenza e del Mercato).
      Rispondi sempre in ITALIANO. Sii professionale, perspicace e incoraggiante.`,
    },
  });
}

export function createAccountantChat(): Chat {
  return ai.chats.create({
    model: "gemini-3.1-pro-preview",
    config: {
      systemInstruction: `Sei un Commercialista esperto iscritto all'Albo in Italia, specializzato in consulenza per startup, piccole medie imprese (PMI) e liberi professionisti.
      La tua missione è fornire informazioni precise, aggiornate e professionali sulla legge italiana, con particolare attenzione a:
      - Regime Forfettario vs Regime Ordinario.
      - Costituzione di società (SRL, SRLS, SAS, SNC).
      - Adempimenti fiscali e scadenze (IVA, IRPEF, IRES, IRAP).
      - Contributi previdenziali (INPS, Casse Professionali).
      - Agevolazioni fiscali, crediti d'imposta e incentivi per l'imprenditoria (es. Resto al Sud, Nuove Imprese a Tasso Zero).
      - Fatturazione elettronica.
      - Diritto del lavoro e contrattualistica base.
      
      IMPORTANTE:
      1. Cita sempre, quando possibile, le leggi o i decreti di riferimento (es. TUIR, Legge di Bilancio).
      2. Sii estremamente preciso ma usa un linguaggio comprensibile anche a chi non è esperto.
      3. Ricorda sempre all'utente che i tuoi consigli sono a scopo informativo e che per operazioni complesse è sempre necessario il supporto diretto di un professionista abilitato che analizzi il caso specifico.
      4. Rispondi sempre in ITALIANO.`,
    },
  });
}
