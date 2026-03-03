interface GenerateAndSpeakResponse {
    sentence: string;
    emotion: string;
    filename: string;
    audio_url: string;
    status_url: string;
    state: 'queued' | 'running' | 'done' | 'error';
    error?: string | null;
}

interface TtsJobResponse {
    state: 'queued' | 'running' | 'done' | 'error';
    error?: string | null;
}

const GENERATE_ENDPOINT = '/api/generate_and_speak_async';
const POLL_INTERVAL_MS = 700;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

let latestJob: GenerateAndSpeakResponse | null = null;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const normalizeEmotion = (emotion: string) => emotion.trim().toLowerCase();

const readErrorMessage = async (res: Response): Promise<string> => {
    try {
        const json = await res.json() as { detail?: string };
        return json.detail || `${res.status} ${res.statusText}`;
    } catch {
        return `${res.status} ${res.statusText}`;
    }
};

export const generateSentence = async (words: string[], emotion: string) => {
    if (words.length === 0) {
        throw new Error('No words provided');
    }

    const response = await fetch(GENERATE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words, emotion: normalizeEmotion(emotion) }),
    });

    if (!response.ok) {
        throw new Error(`Sentence generation failed: ${await readErrorMessage(response)}`);
    }

    const data = await response.json() as GenerateAndSpeakResponse;

    if (!data.sentence || !data.status_url || !data.audio_url || !data.filename) {
        throw new Error('Invalid generate response from backend');
    }

    latestJob = data;
    return { sentence: data.sentence };
};

export const generateAudio = async (sentence: string, emotion: string) => {
    const normalizedEmotion = normalizeEmotion(emotion);

    if (!latestJob) {
        throw new Error('No generation job found. Generate sentence first.');
    }

    if (
        latestJob.sentence !== sentence ||
        normalizeEmotion(latestJob.emotion) !== normalizedEmotion
    ) {
        throw new Error('Generation job mismatch. Please regenerate the sentence.');
    }

    const deadline = Date.now() + POLL_TIMEOUT_MS;

    while (Date.now() < deadline) {
        const statusResponse = await fetch(latestJob.status_url);

        if (!statusResponse.ok) {
            throw new Error(`TTS status check failed: ${await readErrorMessage(statusResponse)}`);
        }

        const status = await statusResponse.json() as TtsJobResponse;

        if (status.state === 'done') {
            return { audio_url: latestJob.audio_url, audio_file: latestJob.filename };
        }

        if (status.state === 'error') {
            throw new Error(status.error || 'TTS generation failed');
        }

        await sleep(POLL_INTERVAL_MS);
    }

    throw new Error('Timed out waiting for generated audio');
};
