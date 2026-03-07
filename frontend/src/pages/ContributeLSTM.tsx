import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { extractTemporalFeatures, hasTemporalSignal } from '../lib/handFeatures';

type PlannedCategory = {
    name: string;
    words: string[];
};

type WordStatus = {
    word: string;
    reps_collected: number;
    reps_target: number;
    is_planned: boolean;
    is_trained: boolean;
    category?: string | null;
};

type WordCheck = {
    word: string;
    exists_trained: boolean;
    exists_in_dataset: boolean;
    reps_collected: number;
    reps_target: number;
    is_planned: boolean;
    category?: string | null;
};

type TrainStatus = {
    state: 'idle' | 'training' | 'error' | string;
    message: string;
    progress: number;
    updated_at?: string;
    started_at?: string | null;
    finished_at?: string | null;
    diagnostics?: TrainDiagnostics | null;
};

type RepState = 'ready' | 'countdown' | 'recording' | 'saving' | 'saved';

type MetricSnapshot = {
    count?: number;
    loss?: number | null;
    accuracy?: number | null;
    avg_confidence?: number | null;
};

type SamplePrediction = {
    sign: string;
    predicted: string;
    confidence: number;
    sample?: string;
};

type TrainDiagnostics = {
    best_epoch?: number;
    best_val_acc?: number;
    history_tail?: Array<{
        epoch: number;
        train_loss?: number | null;
        train_acc?: number | null;
        val_loss?: number | null;
        val_acc?: number | null;
        lr?: number;
    }>;
    in_memory?: {
        train?: MetricSnapshot;
        val?: MetricSnapshot;
        sample_predictions?: SamplePrediction[];
    };
    checkpoint?: {
        train?: MetricSnapshot;
        val?: MetricSnapshot;
        sample_predictions?: SamplePrediction[];
    };
};

const N_FRAMES = 60;
const N_REPS = 15;
const FIRST_COUNTDOWN_SECONDS = 3;
const NEXT_COUNTDOWN_SECONDS = 1;

declare global {
    interface Window {
        Hands: any;
        Camera: any;
        drawConnectors: any;
        drawLandmarks: any;
        HAND_CONNECTIONS: any;
    }
}

const statusText = (entry: WordStatus | undefined) => {
    if (!entry) return 'Not started';
    if (entry.is_trained) return 'Trained';
    if (entry.reps_collected > 0) return `In progress (${entry.reps_collected}/${entry.reps_target})`;
    return 'Not started';
};

const formatPct = (value?: number | null) => {
    if (!Number.isFinite(value)) return 'n/a';
    return `${(Number(value) * 100).toFixed(1)}%`;
};

const formatLoss = (value?: number | null) => {
    if (!Number.isFinite(value)) return 'n/a';
    return Number(value).toFixed(4);
};

export function ContributeLSTM() {
    const [planned, setPlanned] = useState<PlannedCategory[]>([]);
    const [status, setStatus] = useState<WordStatus[]>([]);
    const [loading, setLoading] = useState(true);

    const [customWord, setCustomWord] = useState('');
    const [checkResult, setCheckResult] = useState<WordCheck | null>(null);
    const [selectedWord, setSelectedWord] = useState<string | null>(null);
    const [repsCollected, setRepsCollected] = useState(0);

    const [trainStatus, setTrainStatus] = useState<TrainStatus>({
        state: 'idle',
        message: 'Temporal training idle.',
        progress: 0,
    });
    const [trainError, setTrainError] = useState<string | null>(null);

    const [repState, setRepState] = useState<RepState>('ready');
    const [countdown, setCountdown] = useState(0);
    const [recordedFrames, setRecordedFrames] = useState(0);
    const [recorderHint, setRecorderHint] = useState('Select a word to start collecting reps.');
    const [queueActive, setQueueActive] = useState(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mpRef = useRef<{
        camera: any | null;
        hands: any | null;
        inputCanvas: HTMLCanvasElement | null;
        inputCtx: CanvasRenderingContext2D | null;
    }>({ camera: null, hands: null, inputCanvas: null, inputCtx: null });
    const repFramesRef = useRef<number[][]>([]);
    const saveTimeoutRef = useRef<number | null>(null);
    const savingRef = useRef(false);
    const firstRepRef = useRef(true);
    const repStateRef = useRef<RepState>('ready');
    const queueActiveRef = useRef(false);
    const selectedWordRef = useRef<string | null>(null);
    const repsCollectedRef = useRef(0);
    const prevTrainStateRef = useRef<string>('idle');

    useEffect(() => {
        repStateRef.current = repState;
    }, [repState]);
    useEffect(() => {
        queueActiveRef.current = queueActive;
    }, [queueActive]);
    useEffect(() => {
        selectedWordRef.current = selectedWord;
    }, [selectedWord]);
    useEffect(() => {
        repsCollectedRef.current = repsCollected;
    }, [repsCollected]);

    const statusByWord = useMemo(() => {
        const map = new Map<string, WordStatus>();
        for (const row of status) map.set(row.word, row);
        return map;
    }, [status]);

    const trainedSet = useMemo(() => new Set(status.filter((row) => row.is_trained).map((row) => row.word)), [status]);

    const nextPlannedWord = useMemo(() => {
        const allPlannedWords = planned.flatMap((group) => group.words);
        const incomplete = allPlannedWords
            .map((word) => statusByWord.get(word))
            .filter((entry): entry is WordStatus => Boolean(entry))
            .filter((entry) => !entry.is_trained);
        if (incomplete.length === 0) return null;
        incomplete.sort((a, b) => a.reps_collected - b.reps_collected);
        return incomplete[0].word;
    }, [planned, statusByWord]);

    const refreshStatus = useCallback(async () => {
        const response = await fetch('/api/temporal/status');
        if (!response.ok) {
            throw new Error(`Failed to load temporal status (${response.status})`);
        }
        const data = (await response.json()) as { collection?: WordStatus[] };
        setStatus(Array.isArray(data.collection) ? data.collection : []);
    }, []);

    const fetchTrainStatus = useCallback(async () => {
        try {
            const response = await fetch('/api/temporal/train/status');
            if (!response.ok) return;
            const data = (await response.json()) as TrainStatus;
            setTrainStatus({
                state: data.state ?? 'idle',
                message: data.message ?? '',
                progress: Number.isFinite(data.progress) ? data.progress : 0,
                updated_at: data.updated_at,
                started_at: data.started_at,
                finished_at: data.finished_at,
                diagnostics: data.diagnostics ?? null,
            });
        } catch (error) {
            console.error(error);
        }
    }, []);

    useEffect(() => {
        let ignore = false;
        const controller = new AbortController();

        (async () => {
            try {
                const [plannedRes, statusRes, trainRes] = await Promise.all([
                    fetch('/api/temporal/planned', { signal: controller.signal }),
                    fetch('/api/temporal/status', { signal: controller.signal }),
                    fetch('/api/temporal/train/status', { signal: controller.signal }),
                ]);

                const plannedJson = (await plannedRes.json()) as { categories?: PlannedCategory[] };
                const statusJson = (await statusRes.json()) as { collection?: WordStatus[] };
                const trainJson = (await trainRes.json()) as TrainStatus;

                if (ignore) return;
                setPlanned(Array.isArray(plannedJson.categories) ? plannedJson.categories : []);
                setStatus(Array.isArray(statusJson.collection) ? statusJson.collection : []);
                setTrainStatus({
                    state: trainJson.state ?? 'idle',
                    message: trainJson.message ?? '',
                    progress: Number.isFinite(trainJson.progress) ? trainJson.progress : 0,
                    updated_at: trainJson.updated_at,
                    started_at: trainJson.started_at,
                    finished_at: trainJson.finished_at,
                    diagnostics: trainJson.diagnostics ?? null,
                });
            } catch (error) {
                if (!ignore) console.error(error);
            } finally {
                if (!ignore) setLoading(false);
            }
        })();

        return () => {
            ignore = true;
            controller.abort();
        };
    }, []);

    useEffect(() => {
        if (!selectedWord) return;
        const entry = statusByWord.get(selectedWord);
        if (!entry) return;
        setRepsCollected(entry.reps_collected);
    }, [selectedWord, statusByWord]);

    useEffect(() => {
        if (trainStatus.state !== 'training') return;
        const interval = window.setInterval(() => {
            void fetchTrainStatus();
        }, 1500);
        return () => {
            window.clearInterval(interval);
        };
    }, [fetchTrainStatus, trainStatus.state]);

    useEffect(() => {
        const prev = prevTrainStateRef.current;
        if (prev === 'training' && trainStatus.state === 'idle') {
            void refreshStatus();
        }
        prevTrainStateRef.current = trainStatus.state;
    }, [refreshStatus, trainStatus.state]);

    const resetRepState = useCallback(() => {
        setQueueActive(false);
        setRepState('ready');
        setCountdown(0);
        setRecordedFrames(0);
        repFramesRef.current = [];
        savingRef.current = false;
        if (saveTimeoutRef.current !== null) {
            window.clearTimeout(saveTimeoutRef.current);
            saveTimeoutRef.current = null;
        }
        firstRepRef.current = true;
    }, []);

    const selectWord = useCallback((word: string, reps: number) => {
        setSelectedWord(word);
        setRepsCollected(reps);
        setCheckResult(null);
        resetRepState();
        setRecorderHint(`Ready to collect reps for ${word.replaceAll('_', ' ')}.`);
    }, [resetRepState]);

    const startCountdown = useCallback((seconds: number) => {
        if (!selectedWordRef.current) return;
        if (repsCollectedRef.current >= N_REPS) return;
        repFramesRef.current = [];
        setRecordedFrames(0);
        setCountdown(seconds);
        setRepState('countdown');
        setRecorderHint(`Get ready: ${seconds}`);
    }, []);

    const saveRep = useCallback(async (word: string, frames: number[][]) => {
        setRepState('saving');
        setRecorderHint('Uploading rep...');
        try {
            const response = await fetch('/api/temporal/reps/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word, frames }),
            });
            const data = (await response.json()) as { reps_collected?: number; detail?: string };
            if (!response.ok) {
                throw new Error(data.detail || `Upload failed (${response.status})`);
            }

            const next = Number.isFinite(data.reps_collected) ? Number(data.reps_collected) : repsCollectedRef.current;
            setRepsCollected(next);
            setRepState('saved');
            setRecorderHint(`Rep saved (${next}/${N_REPS}).`);
            firstRepRef.current = false;
            await refreshStatus();

            if (next >= N_REPS) {
                setQueueActive(false);
                setRepState('ready');
                setRecorderHint('Target reached. You can train/retrain now.');
                return;
            }

            if (queueActiveRef.current) {
                saveTimeoutRef.current = window.setTimeout(() => {
                    startCountdown(NEXT_COUNTDOWN_SECONDS);
                }, 700);
            } else {
                setRepState('ready');
            }
        } catch (error) {
            console.error(error);
            setQueueActive(false);
            setRepState('ready');
            setRecorderHint(error instanceof Error ? error.message : 'Failed to save rep.');
        } finally {
            savingRef.current = false;
            repFramesRef.current = [];
            setRecordedFrames(0);
        }
    }, [refreshStatus, startCountdown]);

    const onHandResults = useCallback((results: any) => {
        const videoEl = videoRef.current;
        const canvas = canvasRef.current;
        if (!videoEl || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const handLandmarks = results.multiHandLandmarks ?? [];
        for (let i = 0; i < handLandmarks.length; i += 1) {
            const lm = handLandmarks[i];
            const label = results.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
            const isRight = label === 'Right';
            const color = isRight ? '#007FFF' : '#FF7F40';
            window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: color + '55', lineWidth: 2 });
            window.drawLandmarks(ctx, lm, { color, lineWidth: 1, radius: 3 });
        }

        if (repStateRef.current !== 'recording') return;
        if (!selectedWordRef.current) return;

        const features = extractTemporalFeatures(results);
        repFramesRef.current.push(features);
        const count = repFramesRef.current.length;
        setRecordedFrames(count);
        if (handLandmarks.length === 0) {
            setRecorderHint(`Recording... ${count}/${N_FRAMES} (no hands detected)`);
        } else if (!hasTemporalSignal(features)) {
            setRecorderHint(`Recording... ${count}/${N_FRAMES} (tracking weak)`);
        } else {
            setRecorderHint(`Recording... ${count}/${N_FRAMES}`);
        }

        if (count >= N_FRAMES && !savingRef.current) {
            savingRef.current = true;
            const word = selectedWordRef.current;
            const frames = repFramesRef.current.slice(0, N_FRAMES);
            void saveRep(word, frames);
        }
    }, [saveRep]);

    const stopCamera = useCallback(() => {
        if (saveTimeoutRef.current !== null) {
            window.clearTimeout(saveTimeoutRef.current);
            saveTimeoutRef.current = null;
        }
        if (mpRef.current.camera) {
            try {
                mpRef.current.camera.stop();
            } catch { }
            mpRef.current.camera = null;
        }
        if (mpRef.current.hands) {
            try {
                mpRef.current.hands.close();
            } catch { }
            mpRef.current.hands = null;
        }
        mpRef.current.inputCanvas = null;
        mpRef.current.inputCtx = null;
        if (videoRef.current?.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
        }
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (canvas && ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }, []);

    useEffect(() => {
        let ignore = false;

        async function startCamera() {
            if (!videoRef.current || !canvasRef.current) return;
            if (!window.Hands || !window.Camera) {
                setRecorderHint('MediaPipe scripts are not loaded.');
                return;
            }
            try {
                const hands = new window.Hands({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
                });
                hands.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });
                hands.onResults(onHandResults);
                mpRef.current.hands = hands;

                const inputCanvas = document.createElement('canvas');
                const inputCtx = inputCanvas.getContext('2d');
                mpRef.current.inputCanvas = inputCanvas;
                mpRef.current.inputCtx = inputCtx;

                const camera = new window.Camera(videoRef.current, {
                    onFrame: async () => {
                        if (ignore) return;
                        if (mpRef.current.hands && videoRef.current) {
                            const w = videoRef.current.videoWidth;
                            const h = videoRef.current.videoHeight;
                            const ctx = mpRef.current.inputCtx;
                            const canvas = mpRef.current.inputCanvas;
                            if (ctx && canvas && w && h) {
                                if (canvas.width !== w) canvas.width = w;
                                if (canvas.height !== h) canvas.height = h;
                                ctx.save();
                                ctx.clearRect(0, 0, w, h);
                                ctx.translate(w, 0);
                                ctx.scale(-1, 1);
                                ctx.drawImage(videoRef.current, 0, 0, w, h);
                                ctx.restore();
                                await mpRef.current.hands.send({ image: canvas });
                            } else {
                                await mpRef.current.hands.send({ image: videoRef.current });
                            }
                        }
                    },
                    width: 640,
                    height: 480,
                });

                mpRef.current.camera = camera;
                await camera.start();
            } catch (error) {
                console.error(error);
                setRecorderHint('Camera failed to start.');
            }
        }

        if (trainStatus.state === 'training') {
            setQueueActive(false);
            setRepState('ready');
            setCountdown(0);
            setRecordedFrames(0);
            repFramesRef.current = [];
            savingRef.current = false;
            setRecorderHint('Training in progress. Camera paused to free resources.');
            stopCamera();
            return () => {
                ignore = true;
            };
        }

        void startCamera();

        return () => {
            ignore = true;
            stopCamera();
        };
    }, [onHandResults, stopCamera, trainStatus.state]);

    useEffect(() => {
        if (repState !== 'countdown') return;
        if (countdown <= 0) {
            setRepState('recording');
            setRecordedFrames(0);
            repFramesRef.current = [];
            setRecorderHint(`Recording rep ${Math.min(repsCollected + 1, N_REPS)}...`);
            return;
        }
        const timer = window.setTimeout(() => {
            setCountdown((value) => value - 1);
        }, 1000);
        return () => {
            window.clearTimeout(timer);
        };
    }, [countdown, repState, repsCollected]);

    const startCollection = () => {
        if (!selectedWord) return;
        if (repsCollected >= N_REPS) return;
        setQueueActive(true);
        if (repState === 'ready' || repState === 'saved') {
            startCountdown(firstRepRef.current ? FIRST_COUNTDOWN_SECONDS : NEXT_COUNTDOWN_SECONDS);
        }
    };

    const pauseCollection = () => {
        setQueueActive(false);
        if (repState !== 'saving') {
            setRepState('ready');
            setCountdown(0);
        }
        repFramesRef.current = [];
        setRecordedFrames(0);
        setRecorderHint('Paused. Continue when ready.');
    };

    const checkWord = async () => {
        const raw = customWord.trim();
        if (!raw) return;
        try {
            const response = await fetch('/api/temporal/signs/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word: raw }),
            });
            const data = (await response.json()) as WordCheck;
            if (!response.ok) {
                throw new Error('Word check failed');
            }
            setCheckResult(data);
        } catch (error) {
            console.error(error);
            setCheckResult(null);
        }
    };

    const startTraining = async () => {
        setTrainError(null);
        try {
            const response = await fetch('/api/temporal/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const data = (await response.json()) as { started?: boolean; message?: string };
            if (!response.ok) {
                throw new Error(data.message || `Training request failed (${response.status})`);
            }
            if (data.started === false) {
                setTrainError(data.message || 'Training did not start.');
            }
            await fetchTrainStatus();
        } catch (error) {
            setTrainError(error instanceof Error ? error.message : 'Failed to start training.');
        }
    };

    const trainingInFlight = trainStatus.state === 'training';
    const canTrain = selectedWord !== null && repsCollected >= N_REPS && !trainingInFlight;
    const progressPct = Math.min(100, Math.round((repsCollected / N_REPS) * 100));
    const repFramePct = Math.min(100, Math.round((recordedFrames / N_FRAMES) * 100));
    const checkpointVal = trainStatus.diagnostics?.checkpoint?.val;
    const checkpointSamples = trainStatus.diagnostics?.checkpoint?.sample_predictions ?? [];
    const historyTail = trainStatus.diagnostics?.history_tail ?? [];

    return (
        <div className="h-[calc(100dvh-var(--app-nav-h))] overflow-y-auto">
            <div className="mx-auto min-h-full max-w-[1140px] px-5 py-10 md:px-10">
                <header className="mb-6">
                    <h1 className="font-heading text-[clamp(2rem,4vw,3rem)] font-extrabold leading-[1.08] tracking-tight text-text">
                        Contribute <span className="text-[#c85a21]">Temporal</span> Signs
                    </h1>
                    <p className="mt-2 max-w-[760px] text-[0.95rem] text-muted">
                        Record exactly {N_REPS} reps per word, each rep with {N_FRAMES} normalized landmark frames. This dataset feeds LSTM retraining.
                    </p>
                </header>

                <section className="grid gap-6 lg:grid-cols-[1.12fr_0.88fr]">
                    <div className="space-y-6">
                        <article className="rounded-3xl border border-border-color bg-white p-5 shadow-[0_14px_30px_rgba(15,34,68,0.08)]">
                            <div className="flex items-center justify-between gap-3">
                                <h2 className="text-[0.78rem] font-extrabold uppercase tracking-[0.16em] text-muted">Planned words</h2>
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (!nextPlannedWord) return;
                                        const entry = statusByWord.get(nextPlannedWord);
                                        selectWord(nextPlannedWord, entry?.reps_collected ?? 0);
                                    }}
                                    disabled={!nextPlannedWord}
                                    className="rounded-xl border border-[#c8ddff] bg-[#f2f8ff] px-3 py-1.5 text-[0.72rem] font-bold uppercase tracking-[0.12em] text-brand disabled:cursor-not-allowed disabled:opacity-45"
                                >
                                    Next planned word
                                </button>
                            </div>

                            {loading ? (
                                <div className="mt-4 text-[0.9rem] text-muted">Loading planned words...</div>
                            ) : (
                                <div className="mt-4 grid gap-4">
                                    {planned.map((category) => (
                                        <article key={category.name} className="rounded-2xl border border-border-color bg-[#fbfdff] p-4">
                                            <div className="text-[0.8rem] font-extrabold uppercase tracking-[0.14em] text-muted">
                                                {category.name.replaceAll('_', ' ')}
                                            </div>
                                            <div className="mt-3 flex flex-wrap gap-2">
                                                {category.words.map((word) => {
                                                    const entry = statusByWord.get(word);
                                                    const isSelected = selectedWord === word;
                                                    const trained = entry?.is_trained ?? false;
                                                    return (
                                                        <button
                                                            key={word}
                                                            type="button"
                                                            onClick={() => selectWord(word, entry?.reps_collected ?? 0)}
                                                            title={statusText(entry)}
                                                            className={`rounded-full border px-3 py-1 text-[0.76rem] font-bold transition-colors ${isSelected
                                                                ? 'border-brand bg-[#edf5ff] text-brand'
                                                                : trained
                                                                    ? 'border-[#9fc9ff] bg-[#edf5ff] text-brand'
                                                                    : 'border-border-color bg-white text-text hover:border-[#bad6ff]'
                                                                }`}
                                                        >
                                                            {word.replaceAll('_', ' ')}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </article>
                                    ))}
                                </div>
                            )}
                        </article>

                        <article className="rounded-3xl border border-border-color bg-white p-5 shadow-[0_14px_30px_rgba(15,34,68,0.08)]">
                            <h2 className="text-[0.78rem] font-extrabold uppercase tracking-[0.16em] text-muted">Custom word check</h2>
                            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                                <input
                                    value={customWord}
                                    onChange={(event) => setCustomWord(event.target.value)}
                                    placeholder="Enter a word (e.g. more)"
                                    className="h-11 flex-1 rounded-xl border border-border-color bg-white px-3 text-[0.9rem] font-semibold text-text outline-none transition-colors focus:border-brand"
                                />
                                <button
                                    type="button"
                                    onClick={checkWord}
                                    className="h-11 rounded-xl bg-gradient-to-r from-brand to-brand-end px-4 text-[0.85rem] font-bold text-white"
                                >
                                    Check word
                                </button>
                            </div>

                            {checkResult && (
                                <div className="mt-4 rounded-2xl border border-border-color bg-[#f9fbff] p-4 text-[0.88rem] text-text">
                                    <div className="font-bold">{checkResult.word.replaceAll('_', ' ')}</div>
                                    <div className="mt-1 text-muted">Trained: {checkResult.exists_trained ? 'Yes' : 'No'}</div>
                                    <div className="text-muted">Dataset reps: {checkResult.reps_collected}/{checkResult.reps_target}</div>
                                    <div className="text-muted">Category: {checkResult.category ?? 'Custom'}</div>
                                    <button
                                        type="button"
                                        onClick={() => selectWord(checkResult.word, checkResult.reps_collected)}
                                        className="mt-3 rounded-xl border border-[#c8ddff] bg-[#f2f8ff] px-3 py-2 text-[0.76rem] font-bold uppercase tracking-[0.12em] text-brand"
                                    >
                                        Use this word
                                    </button>
                                </div>
                            )}
                        </article>
                    </div>

                    <div className="space-y-6">
                        <article className="rounded-3xl border border-border-color bg-white p-5 shadow-[0_14px_30px_rgba(15,34,68,0.08)]">
                            <div className="flex items-center justify-between gap-3">
                                <h2 className="text-[0.78rem] font-extrabold uppercase tracking-[0.16em] text-muted">Recording</h2>
                                <div className="rounded-full border border-[#d2e4ff] bg-[#f3f8ff] px-3 py-1 text-[0.7rem] font-bold uppercase tracking-[0.14em] text-muted">
                                    {repState}
                                </div>
                            </div>

                            <div className="mt-3 rounded-xl border border-border-color bg-[#f8fbff] p-3 text-[0.84rem] text-muted">
                                <div className="font-semibold text-text">
                                    Word: {selectedWord ? selectedWord.replaceAll('_', ' ') : 'None selected'}
                                </div>
                                <div className="mt-1">Status: {selectedWord ? statusText(statusByWord.get(selectedWord)) : 'Select a planned or custom word'}</div>
                                <div className="mt-1">Hint: {recorderHint}</div>
                            </div>

                            <div className="relative mt-4 overflow-hidden rounded-2xl border border-border-color bg-black aspect-4/3">
                                <video ref={videoRef} autoPlay muted playsInline className="h-full w-full scale-x-[-1] object-cover" />
                                <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                                <div className="absolute left-3 top-3 rounded-full border border-[#d2e4ff] bg-white/95 px-3 py-1 text-[0.66rem] font-bold uppercase tracking-[0.13em] text-muted">
                                    {queueActive ? 'Auto cycle on' : 'Auto cycle off'}
                                </div>
                                {trainingInFlight && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/58 px-6 text-center">
                                        <div>
                                            <div className="text-[0.72rem] font-bold uppercase tracking-[0.16em] text-white/70">Recorder paused</div>
                                            <div className="mt-2 text-lg font-bold text-white">Training is using the model resources.</div>
                                            <div className="mt-1 text-[0.86rem] text-white/78">Camera and hand landmark detection will resume automatically when training finishes.</div>
                                        </div>
                                    </div>
                                )}
                                {repState === 'countdown' && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/45">
                                        <div className="rounded-2xl border border-white/40 bg-white/15 px-8 py-5 text-center backdrop-blur">
                                            <div className="text-[0.68rem] font-bold uppercase tracking-[0.16em] text-white/80">Get Ready</div>
                                            <div className="mt-1 text-4xl font-extrabold text-white">{countdown}</div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="mt-4 rounded-xl border border-border-color bg-[#f9fbff] p-3">
                                <div className="flex items-center justify-between text-[0.78rem] font-semibold">
                                    <span className="text-muted">Rep progress</span>
                                    <span className="text-text">{repsCollected}/{N_REPS}</span>
                                </div>
                                <div className="mt-2 h-2 overflow-hidden rounded-full bg-[#d8e8ff]">
                                    <div
                                        className="h-full rounded-full bg-gradient-to-r from-brand to-brand-end transition-all duration-300"
                                        style={{ width: `${progressPct}%` }}
                                    />
                                </div>

                                <div className="mt-3 flex items-center justify-between text-[0.78rem] font-semibold">
                                    <span className="text-muted">Current rep frames</span>
                                    <span className="text-text">{recordedFrames}/{N_FRAMES}</span>
                                </div>
                                <div className="mt-2 h-2 overflow-hidden rounded-full bg-[#e4edf9]">
                                    <div
                                        className="h-full rounded-full bg-[#7baeff] transition-all duration-100"
                                        style={{ width: `${repFramePct}%` }}
                                    />
                                </div>
                            </div>

                            <div className="mt-4 flex gap-2">
                                <button
                                    type="button"
                                    onClick={startCollection}
                                    disabled={!selectedWord || repsCollected >= N_REPS || repState === 'saving' || trainingInFlight}
                                    className="flex-1 rounded-xl bg-gradient-to-r from-brand to-brand-end px-4 py-2.5 text-[0.8rem] font-bold uppercase tracking-[0.11em] text-white disabled:cursor-not-allowed disabled:opacity-45"
                                >
                                    Start / Continue
                                </button>
                                <button
                                    type="button"
                                    onClick={pauseCollection}
                                    disabled={repState === 'saving' || trainingInFlight}
                                    className="rounded-xl border border-border-color bg-white px-4 py-2.5 text-[0.8rem] font-bold uppercase tracking-[0.11em] text-muted disabled:cursor-not-allowed disabled:opacity-45"
                                >
                                    Pause
                                </button>
                            </div>
                        </article>

                        <article className="rounded-3xl border border-border-color bg-white p-5 shadow-[0_14px_30px_rgba(15,34,68,0.08)]">
                            <h2 className="text-[0.78rem] font-extrabold uppercase tracking-[0.16em] text-muted">Train / Retrain</h2>
                            <div className="mt-3 rounded-xl border border-border-color bg-[#f9fbff] p-3 text-[0.84rem]">
                                <div className="font-semibold text-text">State: {trainStatus.state}</div>
                                <div className="mt-1 text-muted">{trainStatus.message || 'No message.'}</div>
                                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#d8e8ff]">
                                    <div
                                        className={`h-full rounded-full transition-all duration-300 ${trainStatus.state === 'error' ? 'bg-[#d33f49]' : 'bg-gradient-to-r from-brand to-brand-end'}`}
                                        style={{ width: `${Math.round(Math.max(0, Math.min(1, trainStatus.progress)) * 100)}%` }}
                                    />
                                </div>
                                {trainError && <div className="mt-2 text-[0.78rem] font-semibold text-[#c0413f]">{trainError}</div>}
                                {trainStatus.diagnostics && (
                                    <div className="mt-3 rounded-xl border border-[#dbe8fb] bg-white p-3 text-[0.78rem] text-muted">
                                        <div className="font-semibold text-text">
                                            Best epoch: {trainStatus.diagnostics.best_epoch ?? 'n/a'} | Best val: {formatPct(trainStatus.diagnostics.best_val_acc)}
                                        </div>
                                        <div className="mt-1">
                                            Saved checkpoint: val {formatPct(checkpointVal?.accuracy)}, loss {formatLoss(checkpointVal?.loss)}, avg conf {formatPct(checkpointVal?.avg_confidence)}
                                        </div>
                                        {historyTail.length > 0 && (
                                            <div className="mt-1">
                                                Last epoch: train {formatPct(historyTail[historyTail.length - 1]?.train_acc)}, val {formatPct(historyTail[historyTail.length - 1]?.val_acc)}, lr {historyTail[historyTail.length - 1]?.lr?.toExponential(2) ?? 'n/a'}
                                            </div>
                                        )}
                                        {checkpointSamples.length > 0 && (
                                            <div className="mt-1">
                                                Samples: {checkpointSamples.map((entry) => `${entry.sign}->${entry.predicted} (${Math.round(entry.confidence * 100)}%)`).join(' | ')}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <button
                                type="button"
                                onClick={startTraining}
                                disabled={!canTrain}
                                className="mt-4 w-full rounded-xl bg-gradient-to-r from-[#006fe0] to-[#3e98ff] px-4 py-2.5 text-[0.8rem] font-bold uppercase tracking-[0.11em] text-white disabled:cursor-not-allowed disabled:opacity-45"
                            >
                                {trainingInFlight ? 'Training...' : 'Train / Retrain LSTM'}
                            </button>
                            <p className="mt-2 text-[0.75rem] text-muted">
                                Training unlocks once the selected word reaches {N_REPS}/{N_REPS} reps.
                            </p>
                        </article>

                        <article className="rounded-3xl border border-border-color bg-white p-5 shadow-[0_14px_30px_rgba(15,34,68,0.08)]">
                            <h2 className="text-[0.78rem] font-extrabold uppercase tracking-[0.16em] text-muted">Trained signs</h2>
                            <div className="mt-3 flex flex-wrap gap-2">
                                {Array.from(trainedSet).length === 0 ? (
                                    <div className="text-[0.84rem] text-muted">No trained temporal classes yet.</div>
                                ) : (
                                    Array.from(trainedSet).sort().map((word) => (
                                        <span key={word} className="rounded-full border border-[#9fc9ff] bg-[#edf5ff] px-3 py-1 text-[0.75rem] font-bold text-brand">
                                            {word.replaceAll('_', ' ')}
                                        </span>
                                    ))
                                )}
                            </div>
                        </article>
                    </div>
                </section>
            </div>
        </div>
    );
}
