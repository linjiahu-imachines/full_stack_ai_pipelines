# English open-source landscape — real-time voice-to-voice AI (CEO note)

**Source:** Document from CEO  
**Archived:** 2026-05-15  

---

## Original document (verbatim)

The English open-source landscape for real-time voice-to-voice AI is more mature than the Japanese market, with several "Omni" models that eliminate the delay of traditional speech-to-text pipelines. [1]

Here are the primary open-source English models and research projects for real-time voice conversation:

### Top Open-Source English Voice-to-Voice Models (2026)

- **Moshi (Kyutai):** Originally a research prototype, Moshi is the first full-duplex, open-source conversational AI. It can listen and speak simultaneously with a latency of ~200ms, handling interruptions and emotional nuances naturally.

- **Mini-Omni / Mini-Omni2:** A lightweight, end-to-end model designed to mimic GPT-4o's capabilities. It uses a "batch parallel decoding" method to provide real-time audio output directly from the model without an external TTS system, significantly reducing lag.

- **GLM-4-Voice (Zhipu AI):** A high-performance, 9B parameter bilingual (English/Chinese) model. It supports real-time interruptions and allows users to change the AI's emotion, speed, and dialect through direct instructions during the conversation.

- **LLaMA-Omni:** Built on Llama-3.1-8B, this model targets seamless speech interaction by generating text and speech responses simultaneously from audio instructions. [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

### Key Technologies for English Pipelines

If you aren't using an "all-in-one" model like Moshi, the best open-source "Lego-style" components in 2026 are:

- **ASR (Listening):** OpenAI Whisper remains the industry standard for real-time English transcription.

- **TTS (Speaking):** Kokoro-82M is currently the top choice for low-latency, high-quality English voice generation that can run on consumer CPUs. Other popular options include Fish Speech and F5-TTS for realistic voice cloning.

- **Logic (Thinking):** Llama 3.2 or Gemma 2 are commonly used as the "brain" in these pipelines due to their high reasoning capabilities. [1, 12, 13, 14, 15]

### Summary of English Open-Source Options

| Model [5, 9, 11, 16, 17] | Strengths | Best Use Case |
|--------------------------|-----------|---------------|
| Moshi | Full-duplex (talk/listen at once) | Natural "human-like" chatting |
| Mini-Omni | Ultra-lightweight & fast | Mobile or edge device applications |
| GLM-4-Voice | High reasoning (9B) | Complex tasks with emotional control |
| LLaMA-Omni | Large context/Llama ecosystem | Integrated enterprise voice agents |

### References

[1] https://www.lindy.ai  
[2] https://github.com  
[3] https://www.youtube.com  
[4] https://huggingface.co  
[5] https://www.voiceaispace.com  
[6] https://www.youtube.com  
[7] https://neurohive.io  
[8] https://arxiv.org  
[9] https://arxiv.org  
[10] https://www.youtube.com  
[11] https://www.youtube.com  
[12] https://github.com  
[13] https://www.youtube.com  
[14] https://inworld.ai  
[15] https://www.youtube.com  
[16] https://www.youtube.com  
[17] https://huggingface.co  
