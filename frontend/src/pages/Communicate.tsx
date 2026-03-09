import React, { useState, useRef, useEffect, useCallback } from 'react';
import { WebcamPane } from '../components/WebcamPane';
import type { SignDetectionMeta } from '../components/WebcamPane';
import { WordBuffer } from '../components/WordBuffer';
import { SentenceOutput } from '../components/SentenceOutput';
import type { EmotionType } from '../components/EmotionStrip';
import { generateSentence, generateAudio } from '../lib/api.ts';
import {
    Play, Square, RotateCcw, XCircle, Camera,
    Sliders, Zap, ChevronRight, Volume2, Activity
} from 'lucide-react';
import { useModel } from '../model/ModelContext';
import { motion, AnimatePresence } from 'framer-motion';

const DEFAULT_HOLD_FRAMES_MLP = 10;
const DEFAULT_MIN_CONFIDENCE = 0.60;
const LSTM_DUPLICATE_GUARD_MS = 250;
const EMOTION_OPTIONS: Array<{ type: EmotionType; emoji: string; label: string }> = [
    { type: 'neutral', emoji: '😐', label: 'NEUTRAL' },
    { type: 'happy', emoji: '🙂', label: 'HAPPY' },
    { type: 'sad', emoji: '😔', label: 'SAD' },
    { type: 'angry', emoji: '😠', label: 'ANGRY' },
    { type: 'fear', emoji: '😨', label: 'FEAR' },
    { type: 'disgust', emoji: '🤢', label: 'DISGUST' },
    { type: 'surprise', emoji: '😮', label: 'SURPRISE' },
];

type EmotionCounts = Record<EmotionType, number>;

const createEmotionCounts = (): EmotionCounts => ({
    neutral: 0,
    happy: 0,
    sad: 0,
    angry: 0,
    fear: 0,
    disgust: 0,
    surprise: 0,
});

const getMostFrequentEmotion = (counts: EmotionCounts): EmotionType => {
    let winner: EmotionType = 'neutral';
    let winnerCount = counts.neutral;

    (Object.entries(counts) as [EmotionType, number][]).forEach(([emotion, count]) => {
        if (count > winnerCount) {
            winner = emotion;
            winnerCount = count;
        }
    });

    return winner;
};

const GLOBAL_STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

@keyframes sentisign-pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(51,153,255,0.55), 0 0 0 0 rgba(51,153,255,0.25); }
    70%  { box-shadow: 0 0 0 14px rgba(51,153,255,0), 0 0 0 28px rgba(51,153,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(51,153,255,0), 0 0 0 0 rgba(51,153,255,0); }
}

@keyframes sentisign-glow-shift {
    0%,100% { opacity: 0.55; transform: scale(1); }
    50%     { opacity: 0.85; transform: scale(1.06); }
}

@keyframes sentisign-orbit {
    from { transform: rotate(0deg) translateX(110px) rotate(0deg); }
    to   { transform: rotate(360deg) translateX(110px) rotate(-360deg); }
}

@keyframes sentisign-orbit2 {
    from { transform: rotate(180deg) translateX(80px) rotate(-180deg); }
    to   { transform: rotate(540deg) translateX(80px) rotate(-540deg); }
}

@keyframes sentisign-float {
    0%,100% { transform: translateY(0px); }
    50%     { transform: translateY(-10px); }
}

@keyframes sentisign-dot-blink {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.2; }
}

.sentisign-pulse-ring  { animation: sentisign-pulse-ring 2s ease-out infinite; }
.sentisign-glow-shift  { animation: sentisign-glow-shift 3s ease-in-out infinite; }
.sentisign-orbit       { animation: sentisign-orbit 8s linear infinite; }
.sentisign-orbit2      { animation: sentisign-orbit2 6s linear infinite; }
.sentisign-float       { animation: sentisign-float 4s ease-in-out infinite; }
.sentisign-dot-blink   { animation: sentisign-dot-blink 1.1s ease-in-out infinite; }

.ss-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: rgba(51,153,255,0.18);
    border-radius: 4px;
    outline: none;
    cursor: pointer;
}

.ss-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3399ff, #66c0ff);
    box-shadow: 0 0 10px rgba(51,153,255,0.7);
    cursor: pointer;
    transition: transform 0.15s;
}

.ss-slider::-webkit-slider-thumb:hover {
    transform: scale(1.25);
}

.ss-slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border: none;
    border-radius: 50%;
    background: linear-gradient(135deg, #3399ff, #66c0ff);
    box-shadow: 0 0 10px rgba(51,153,255,0.7);
    cursor: pointer;
}

.ss-panel {
    background: rgba(8,16,36,0.72);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(51,153,255,0.14);
    border-radius: 20px;
}

.ss-panel-lit {
    background: rgba(8,16,36,0.85);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(51,153,255,0.28);
    border-radius: 20px;
    box-shadow: 0 0 0 1px rgba(51,153,255,0.06) inset, 0 24px 60px rgba(0,0,0,0.5);
}

.ss-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 6px;
}

.ss-main-grid {
    gap: 12px;
    padding: 12px;
}

.ss-card-pad {
    padding: 12px 14px;
}

.ss-session-btn {
    height: 44px;
}

.ss-mini-btn {
    height: 34px;
}

.ss-emotion-grid {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 4px;
}

.ss-emotion-btn {
    min-width: 0;
    height: 46px;
}

.ss-generate-btn {
    height: 44px;
}

.ss-active-row {
    min-height: 36px;
}

.ss-word-buffer-shell {
    height: 116px;
}

@media (max-height: 800px) {
    .ss-main-grid {
        gap: 10px;
        padding: 10px;
    }

    .ss-card-pad {
        padding: 10px 12px;
    }

    .ss-session-btn,
    .ss-generate-btn {
        height: 38px;
    }

    .ss-mini-btn {
        height: 32px;
    }

    .ss-emotion-btn {
        height: 42px;
    }

    .ss-active-row {
        min-height: 32px;
    }

    .ss-word-buffer-shell {
        height: 92px;
    }
}
`;

export const Communicate: React.FC = () => {
    const { model, sessionResetNonce } = useModel();
    const activeModel = model ?? 'mlp';

    const [initialized, setInitialized] = useState(false);
    const [minConfidence, setMinConfidence] = useState(DEFAULT_MIN_CONFIDENCE);
    const [holdFramesSetting, setHoldFramesSetting] = useState(DEFAULT_HOLD_FRAMES_MLP);
    const [cameraReady, setCameraReady] = useState(false);

    const [commitResetNonce, setCommitResetNonce] = useState(0);
    const [wordBuffer, setWordBuffer] = useState<string[]>([]);
    const [sentence, setSentence] = useState<string>('');
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [audioFilename, setAudioFilename] = useState<string | null>(null);
    const [generationError, setGenerationError] = useState<string | null>(null);
    const [generationStage, setGenerationStage] = useState<'idle' | 'sentence' | 'audio'>('idle');
    const [sessionActive, setSessionActive] = useState<boolean>(false);
    const [emotionCounts, setEmotionCounts] = useState<EmotionCounts>(() => createEmotionCounts());
    const [emotionOverride, setEmotionOverride] = useState<EmotionType | null>(null);
    const [signLabel, setSignLabel] = useState<string>('No sign detected');
    const [confidence, setConfidence] = useState<number>(0);

    const detectedEmotion = getMostFrequentEmotion(emotionCounts);
    const selectedEmotion = emotionOverride ?? detectedEmotion;
    const canGenerate = wordBuffer.length > 0;
    const isGenerating = generationStage !== 'idle';

    const trackingRef = useRef<{
        holdCounter: number;
        currentClass: string | null;
        lastWord: string;
        lastCommittedClass: string | null;
        lastCommitAt: number;
        repeatArmed: boolean;
    }>({
        holdCounter: 0,
        currentClass: null,
        lastWord: '',
        lastCommittedClass: null,
        lastCommitAt: 0,
        repeatArmed: true,
    });

    const generationAbortRef = useRef<AbortController | null>(null);
    const generationRunIdRef = useRef<number>(0);
    const styleInjectedRef = useRef(false);

    useEffect(() => {
        if (styleInjectedRef.current) {
            return;
        }

        styleInjectedRef.current = true;
        const el = document.createElement('style');
        el.id = 'sentisign-styles';
        el.textContent = GLOBAL_STYLES;
        document.head.appendChild(el);

        return () => {
            el.remove();
            styleInjectedRef.current = false;
        };
    }, []);

    useEffect(() => {
        return () => {
            generationAbortRef.current?.abort();
        };
    }, []);

    const cancelGeneration = useCallback(() => {
        generationAbortRef.current?.abort();
        generationAbortRef.current = null;
        setGenerationError(null);
        setGenerationStage('idle');
    }, []);

    const commitDetectedWord = useCallback((word: string) => {
        setWordBuffer((prev) => [...prev, word]);
        setCommitResetNonce((prev) => prev + 1);
    }, []);

    const handleEmotionDetected = useCallback((emotion: EmotionType) => {
        if (!sessionActive) {
            return;
        }

        setEmotionCounts((prev) => ({
            ...prev,
            [emotion]: prev[emotion] + 1,
        }));
    }, [sessionActive]);

    const handleSignDetected = useCallback(
        (word: string | null, cls: string | null, conf: number, meta?: SignDetectionMeta) => {
            const isTemporalReset = activeModel === 'lstm' && meta?.phase === 'reset';
            const shouldDisplay = !isTemporalReset && conf >= minConfidence;
            const nextConfidence = shouldDisplay ? conf : 0;
            const nextSignLabel = shouldDisplay
                ? (word ? `${cls} → ${word}` : (cls ?? 'No sign detected'))
                : 'No sign detected';

            setConfidence((prev) => (Object.is(prev, nextConfidence) ? prev : nextConfidence));
            setSignLabel((prev) => (prev === nextSignLabel ? prev : nextSignLabel));

            if (!sessionActive) {
                trackingRef.current.currentClass = null;
                trackingRef.current.holdCounter = 0;
                trackingRef.current.repeatArmed = true;
                return;
            }

            if (activeModel === 'lstm') {
                if (meta?.phase !== 'final') {
                    trackingRef.current.currentClass = null;
                    trackingRef.current.holdCounter = 0;
                    return;
                }

                if (!cls || !word || conf < minConfidence) {
                    return;
                }

                const now = Date.now();
                if (
                    trackingRef.current.lastCommittedClass === cls &&
                    now - trackingRef.current.lastCommitAt < LSTM_DUPLICATE_GUARD_MS
                ) {
                    return;
                }

                commitDetectedWord(word);
                trackingRef.current.currentClass = cls;
                trackingRef.current.lastWord = word;
                trackingRef.current.lastCommittedClass = cls;
                trackingRef.current.lastCommitAt = now;
                trackingRef.current.holdCounter = 0;
                trackingRef.current.repeatArmed = true;
                return;
            }

            if (!cls) {
                trackingRef.current.currentClass = null;
                trackingRef.current.holdCounter = 0;
                return;
            }

            if (cls !== trackingRef.current.currentClass) {
                trackingRef.current.currentClass = cls;
                trackingRef.current.holdCounter = 0;
                return;
            }

            if (conf >= minConfidence) {
                trackingRef.current.holdCounter += 1;
            } else {
                trackingRef.current.holdCounter = Math.max(0, trackingRef.current.holdCounter - 1);
            }

            if (
                trackingRef.current.holdCounter >= holdFramesSetting &&
                word &&
                word !== trackingRef.current.lastWord
            ) {
                commitDetectedWord(word);
                trackingRef.current.lastWord = word;
                trackingRef.current.lastCommitAt = Date.now();
                trackingRef.current.holdCounter = 0;
            }
        },
        [activeModel, commitDetectedWord, sessionActive, minConfidence, holdFramesSetting]
    );

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.code === 'Space' && sessionActive) {
                e.preventDefault();
                setWordBuffer((prev) => [...prev, `Word_${prev.length + 1}`]);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [sessionActive]);

    const removeLastWord = () => setWordBuffer((prev) => prev.slice(0, -1));

    const clearWords = useCallback(() => {
        cancelGeneration();
        setWordBuffer([]);
        setSentence('');
        setAudioUrl(null);
        setAudioFilename(null);
        setEmotionCounts(createEmotionCounts());
        setSignLabel('No sign detected');
        setConfidence(0);
        trackingRef.current.holdCounter = 0;
        trackingRef.current.currentClass = null;
        trackingRef.current.lastWord = '';
        trackingRef.current.lastCommittedClass = null;
        trackingRef.current.lastCommitAt = 0;
        trackingRef.current.repeatArmed = true;
    }, [cancelGeneration]);

    useEffect(() => {
        if (sessionResetNonce === 0) {
            return;
        }

        setSessionActive(false);
        clearWords();
    }, [clearWords, sessionResetNonce]);

    const handleGenerateAndSpeak = async () => {
        if (wordBuffer.length === 0) {
            return;
        }

        if (sessionActive) {
            setSessionActive(false);
            trackingRef.current.holdCounter = 0;
            trackingRef.current.currentClass = null;
            trackingRef.current.repeatArmed = true;
        }

        generationAbortRef.current?.abort();
        const abortController = new AbortController();
        generationAbortRef.current = abortController;
        generationRunIdRef.current += 1;
        const runId = generationRunIdRef.current;

        setGenerationError(null);
        setGenerationStage('sentence');

        try {
            const data = await generateSentence(wordBuffer, selectedEmotion, { signal: abortController.signal });
            if (generationRunIdRef.current !== runId) {
                return;
            }

            if (data.sentence) {
                setSentence(data.sentence);
                setGenerationStage('audio');

                const audioData = await generateAudio(data.sentence, selectedEmotion, { signal: abortController.signal });
                if (generationRunIdRef.current !== runId) {
                    return;
                }

                if (audioData.audio_url) {
                    setAudioUrl(`${audioData.audio_url}?t=${Date.now()}`);
                    setAudioFilename(audioData.audio_file);
                }
            }
        } catch (error) {
            if (error instanceof DOMException && error.name === 'AbortError') {
                return;
            }

            if (error instanceof Error && error.name === 'AbortError') {
                return;
            }

            console.error('Generation error:', error);
            setGenerationError(error instanceof Error ? error.message : 'Generation failed');
        } finally {
            if (generationRunIdRef.current === runId) {
                generationAbortRef.current = null;
                setGenerationStage('idle');
            }
        }
    };

    const handleStartSession = () => {
        setSessionActive(true);
        setEmotionCounts(createEmotionCounts());
        trackingRef.current.holdCounter = 0;
        trackingRef.current.currentClass = null;
        trackingRef.current.lastWord = '';
        trackingRef.current.lastCommittedClass = null;
        trackingRef.current.lastCommitAt = 0;
        trackingRef.current.repeatArmed = true;
    };

    const handleStopSession = () => setSessionActive(false);

    if (!initialized) {
        return (
            <div
                style={{
                    minHeight: 'calc(100dvh - var(--app-nav-h, 0px))',
                    background: 'linear-gradient(135deg, #040c1e 0%, #060f24 50%, #03091a 100%)',
                    fontFamily: "'Sora', sans-serif",
                    position: 'relative',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
                    <div
                        className="sentisign-glow-shift"
                        style={{
                            position: 'absolute',
                            top: '-10%',
                            left: '-5%',
                            width: '55%',
                            height: '60%',
                            background: 'radial-gradient(ellipse at center, rgba(51,153,255,0.12) 0%, transparent 70%)',
                        }}
                    />
                    <div
                        className="sentisign-glow-shift"
                        style={{
                            position: 'absolute',
                            bottom: '-15%',
                            right: '-5%',
                            width: '50%',
                            height: '55%',
                            background: 'radial-gradient(ellipse at center, rgba(0,180,255,0.09) 0%, transparent 70%)',
                            animationDelay: '1.5s',
                        }}
                    />
                    <div
                        style={{
                            position: 'absolute',
                            inset: 0,
                            backgroundImage: `linear-gradient(rgba(51,153,255,0.04) 1px, transparent 1px),
                                linear-gradient(90deg, rgba(51,153,255,0.04) 1px, transparent 1px)`,
                            backgroundSize: '60px 60px',
                        }}
                    />
                </div>

                <div style={{ width: '100%', maxWidth: 960, padding: '2rem', position: 'relative', zIndex: 10 }}>
                    <motion.div
                        initial={{ opacity: 0, y: 24 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                        style={{ textAlign: 'center', marginBottom: '2.5rem' }}
                    >
                        <div className="sentisign-float" style={{ position: 'relative', display: 'inline-block', marginBottom: '1.5rem' }}>
                            <div
                                style={{
                                    width: 80,
                                    height: 80,
                                    borderRadius: '50%',
                                    background: 'linear-gradient(135deg, rgba(51,153,255,0.15), rgba(0,180,255,0.06))',
                                    border: '1px solid rgba(51,153,255,0.3)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    boxShadow: '0 0 40px rgba(51,153,255,0.25), 0 0 80px rgba(51,153,255,0.1)',
                                    margin: '0 auto',
                                }}
                            >
                                <Camera size={34} color="#3399ff" strokeWidth={1.5} />
                            </div>

                            <div style={{ position: 'absolute', inset: 0, width: 80, height: 80, margin: 'auto' }}>
                                <div
                                    className="sentisign-orbit"
                                    style={{
                                        position: 'absolute',
                                        top: '50%',
                                        left: '50%',
                                        width: 8,
                                        height: 8,
                                        borderRadius: '50%',
                                        background: '#3399ff',
                                        boxShadow: '0 0 12px #3399ff',
                                        marginTop: -4,
                                        marginLeft: -4,
                                    }}
                                />
                            </div>
                            <div style={{ position: 'absolute', inset: 0, width: 80, height: 80, margin: 'auto' }}>
                                <div
                                    className="sentisign-orbit2"
                                    style={{
                                        position: 'absolute',
                                        top: '50%',
                                        left: '50%',
                                        width: 5,
                                        height: 5,
                                        borderRadius: '50%',
                                        background: 'rgba(102,192,255,0.8)',
                                        boxShadow: '0 0 8px rgba(102,192,255,0.9)',
                                        marginTop: -2.5,
                                        marginLeft: -2.5,
                                    }}
                                />
                            </div>
                        </div>

                        <h1
                            style={{
                                fontFamily: "'Sora', sans-serif",
                                fontSize: 'clamp(2.6rem,5vw,4rem)',
                                fontWeight: 800,
                                letterSpacing: '-0.03em',
                                lineHeight: 1.05,
                                margin: '0 0 0.5rem',
                                background: 'linear-gradient(135deg, #ffffff 30%, #a8d4ff 65%, #3399ff 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text',
                            }}
                        >
                            Senti
                            <span
                                style={{
                                    background: 'linear-gradient(135deg, #3399ff, #00c8ff)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    backgroundClip: 'text',
                                }}
                            >
                                Sign
                            </span>
                        </h1>
                        <p
                            style={{
                                fontFamily: "'Sora', sans-serif",
                                color: 'rgba(180,210,255,0.65)',
                                fontSize: '1rem',
                                fontWeight: 300,
                                letterSpacing: '0.02em',
                                maxWidth: 480,
                                margin: '0 auto',
                                lineHeight: 1.6,
                            }}
                        >
                            Real-time ASL recognition · Emotion-aware TTS
                        </p>
                    </motion.div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', alignItems: 'start' }}>
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.6, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
                            className="ss-panel"
                            style={{ padding: '1.75rem' }}
                        >
                            <p
                                style={{
                                    color: 'rgba(51,153,255,0.8)',
                                    fontSize: '0.65rem',
                                    fontWeight: 600,
                                    letterSpacing: '0.2em',
                                    textTransform: 'uppercase',
                                    fontFamily: "'JetBrains Mono', monospace",
                                    marginBottom: '1.25rem',
                                }}
                            >
                                SYSTEM OVERVIEW
                            </p>

                            {[
                                { icon: <Activity size={16} />, color: '#3399ff', label: 'Sign Recognition', desc: `MediaPipe landmarks → ${activeModel.toUpperCase()} classifier` },
                                { icon: <Zap size={16} />, color: '#ffb347', label: 'Emotion Detection', desc: 'ResNet CNN · 7 facial categories' },
                                { icon: <Sliders size={16} />, color: '#3399ff', label: 'Sentence Generation', desc: 'Flan-T5-Large · zero-shot NLG' },
                                { icon: <Volume2 size={16} />, color: '#44d9a0', label: 'Speech Synthesis', desc: 'Cartesia Sonic-3 · emotion-conditioned' },
                            ].map((feature, index) => (
                                <motion.div
                                    key={feature.label}
                                    initial={{ opacity: 0, x: -12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.25 + index * 0.1, duration: 0.5 }}
                                    style={{ display: 'flex', gap: '0.85rem', marginBottom: index < 3 ? '1rem' : 0, alignItems: 'flex-start' }}
                                >
                                    <div
                                        style={{
                                            width: 34,
                                            height: 34,
                                            flexShrink: 0,
                                            borderRadius: 10,
                                            background: `rgba(${feature.color === '#3399ff' ? '51,153,255' : feature.color === '#ffb347' ? '255,179,71' : '68,217,160'},0.12)`,
                                            border: `1px solid ${feature.color}26`,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: feature.color,
                                        }}
                                    >
                                        {feature.icon}
                                    </div>
                                    <div>
                                        <div style={{ color: 'rgba(220,235,255,0.92)', fontSize: '0.875rem', fontWeight: 600, marginBottom: 2 }}>
                                            {feature.label}
                                        </div>
                                        <div style={{ color: 'rgba(140,170,220,0.6)', fontSize: '0.75rem', fontWeight: 300, lineHeight: 1.4 }}>
                                            {feature.desc}
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
                            className="ss-panel-lit"
                            style={{ padding: '1.75rem' }}
                        >
                            <p
                                style={{
                                    color: 'rgba(51,153,255,0.8)',
                                    fontSize: '0.65rem',
                                    fontWeight: 600,
                                    letterSpacing: '0.2em',
                                    textTransform: 'uppercase',
                                    fontFamily: "'JetBrains Mono', monospace",
                                    marginBottom: '1.5rem',
                                }}
                            >
                                DETECTION PARAMETERS
                            </p>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <span style={{ color: 'rgba(160,190,240,0.75)', fontSize: '0.78rem', fontWeight: 400 }}>Active model</span>
                                    <span
                                        className="ss-badge"
                                        style={{
                                            background: 'rgba(51,153,255,0.12)',
                                            border: '1px solid rgba(51,153,255,0.3)',
                                            color: '#66bfff',
                                        }}
                                    >
                                        {activeModel.toUpperCase()}
                                    </span>
                                </div>
                            </div>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.6rem' }}>
                                    <span style={{ color: 'rgba(160,190,240,0.75)', fontSize: '0.78rem', fontWeight: 400 }}>Min confidence</span>
                                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#3399ff', fontSize: '0.82rem', fontWeight: 500 }}>
                                        {Math.round(minConfidence * 100)}%
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min={0.30}
                                    max={0.95}
                                    step={0.01}
                                    value={minConfidence}
                                    onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                                    className="ss-slider"
                                    style={{ '--slider-val': `${((minConfidence - 0.30) / 0.65) * 100}%` } as React.CSSProperties}
                                />
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.3rem' }}>
                                    <span style={{ color: 'rgba(120,150,200,0.4)', fontSize: '0.65rem' }}>30%</span>
                                    <span style={{ color: 'rgba(120,150,200,0.4)', fontSize: '0.65rem' }}>95%</span>
                                </div>
                            </div>

                            {activeModel === 'mlp' && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.6rem' }}>
                                        <span style={{ color: 'rgba(160,190,240,0.75)', fontSize: '0.78rem', fontWeight: 400 }}>Hold frames</span>
                                        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#3399ff', fontSize: '0.82rem', fontWeight: 500 }}>
                                            {holdFramesSetting}f
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min={3}
                                        max={25}
                                        step={1}
                                        value={holdFramesSetting}
                                        onChange={(e) => setHoldFramesSetting(parseInt(e.target.value, 10))}
                                        className="ss-slider"
                                    />
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.3rem' }}>
                                        <span style={{ color: 'rgba(120,150,200,0.4)', fontSize: '0.65rem' }}>3</span>
                                        <span style={{ color: 'rgba(120,150,200,0.4)', fontSize: '0.65rem' }}>25</span>
                                    </div>
                                </div>
                            )}

                            <div style={{ borderTop: '1px solid rgba(51,153,255,0.1)', paddingTop: '1rem', marginBottom: '1.25rem' }}>
                                <p style={{ color: 'rgba(120,150,200,0.5)', fontSize: '0.72rem', lineHeight: 1.5, fontWeight: 300 }}>
                                    Camera access will be requested on initialization. Ensure adequate lighting and that your hand(s) are visible.
                                </p>
                            </div>

                            <motion.button
                                type="button"
                                onClick={() => {
                                    setInitialized(true);
                                    handleStartSession();
                                    setTimeout(() => setCameraReady(true), 800);
                                }}
                                whileHover={{ scale: 1.025, y: -2 }}
                                whileTap={{ scale: 0.97 }}
                                style={{
                                    width: '100%',
                                    height: 52,
                                    borderRadius: 14,
                                    background: 'linear-gradient(135deg, #1a6fff 0%, #3399ff 50%, #00c2ff 100%)',
                                    border: 'none',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '0.6rem',
                                    color: '#fff',
                                    fontSize: '0.92rem',
                                    fontWeight: 700,
                                    fontFamily: "'Sora', sans-serif",
                                    letterSpacing: '0.04em',
                                    boxShadow: '0 8px 32px rgba(51,153,255,0.45), 0 2px 8px rgba(0,0,0,0.3)',
                                }}
                            >
                                <Camera size={17} strokeWidth={2.5} />
                                Initialize Camera
                                <ChevronRight size={16} strokeWidth={2.5} />
                            </motion.button>
                        </motion.div>
                    </div>
                </div>
            </div>
        );
    }

    const confPct = Math.round(confidence * 100);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="ss-main-grid"
            style={{
                height: 'calc(100dvh - var(--app-nav-h, 0px))',
                display: 'grid',
                gridTemplateColumns: 'minmax(0,1fr) 370px',
                background: 'linear-gradient(135deg, #040c1e 0%, #060f24 50%, #03091a 100%)',
                fontFamily: "'Sora', sans-serif",
                overflow: 'hidden',
            }}
        >
            <section style={{ display: 'flex', flexDirection: 'column', gap: '10px', minHeight: 0, height: '100%', overflow: 'hidden' }}>
                <motion.div
                    initial={{ opacity: 0, scale: 0.92, y: 30 }}
                    animate={{ opacity: cameraReady ? 1 : 0, scale: cameraReady ? 1 : 0.92, y: cameraReady ? 0 : 30 }}
                    transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
                    style={{
                        flex: 1,
                        minHeight: 0,
                        borderRadius: 22,
                        overflow: 'hidden',
                        position: 'relative',
                        border: sessionActive ? '1px solid rgba(51,153,255,0.55)' : '1px solid rgba(51,153,255,0.2)',
                        boxShadow: sessionActive
                            ? '0 0 0 1px rgba(51,153,255,0.12), 0 30px 80px rgba(0,0,0,0.7), 0 0 60px rgba(51,153,255,0.15)'
                            : '0 20px 60px rgba(0,0,0,0.6)',
                        transition: 'border-color 0.4s, box-shadow 0.4s',
                    }}
                    className={sessionActive ? 'sentisign-pulse-ring' : ''}
                >
                    <div
                        style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            zIndex: 20,
                            padding: '0.85rem 1rem',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            background: 'linear-gradient(180deg, rgba(4,12,30,0.42) 0%, rgba(4,12,30,0.14) 52%, transparent 100%)',
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            <div
                                className={sessionActive ? 'sentisign-dot-blink' : ''}
                                style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    background: sessionActive ? '#3399ff' : 'rgba(100,140,200,0.4)',
                                    boxShadow: sessionActive ? '0 0 10px #3399ff' : 'none',
                                }}
                            />
                            <span
                                style={{
                                    fontFamily: "'JetBrains Mono', monospace",
                                    fontSize: '0.68rem',
                                    fontWeight: 500,
                                    letterSpacing: '0.14em',
                                    color: sessionActive ? 'rgba(51,153,255,0.9)' : 'rgba(100,140,200,0.5)',
                                    textTransform: 'uppercase',
                                }}
                            >
                                {sessionActive ? 'LIVE DETECTION' : 'STANDBY'}
                            </span>
                        </div>

                        <AnimatePresence>
                            {sessionActive && confidence > 0 && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.8 }}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        background: 'rgba(4,12,30,0.75)',
                                        border: '1px solid rgba(51,153,255,0.25)',
                                        borderRadius: 8,
                                        padding: '4px 10px',
                                    }}
                                >
                                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'rgba(180,210,255,0.6)' }}>
                                        CONF
                                    </span>
                                    <span
                                        style={{
                                            fontFamily: "'JetBrains Mono', monospace",
                                            fontSize: '0.82rem',
                                            fontWeight: 700,
                                            color: confPct >= 80 ? '#44d9a0' : confPct >= 60 ? '#3399ff' : '#ffb347',
                                        }}
                                    >
                                        {confPct}%
                                    </span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <div style={{ width: '100%', height: '100%', background: '#020810' }}>
                        <WebcamPane
                            model={activeModel}
                            isActive={sessionActive}
                            commitResetNonce={commitResetNonce}
                            onEmotionDetected={handleEmotionDetected}
                            onSignDetected={handleSignDetected}
                            currentEmotion={detectedEmotion}
                            wordLabel={sessionActive ? signLabel : 'No sign detected'}
                            confidence={sessionActive ? confidence : 0}
                        />
                    </div>

                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: cameraReady ? 1 : 0, y: cameraReady ? 0 : 12 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                    className="ss-word-buffer-shell"
                    style={{ flexShrink: 0 }}
                >
                    <WordBuffer words={wordBuffer} />
                </motion.div>
            </section>

            <motion.aside
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: cameraReady ? 1 : 0, x: cameraReady ? 0 : 30 }}
                transition={{ duration: 0.6, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
                style={{ display: 'flex', flexDirection: 'column', gap: '10px', minHeight: 0, height: '100%', overflow: 'hidden' }}
            >
                <div className="ss-panel-lit ss-card-pad" style={{ flexShrink: 0, marginTop: 'auto' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.7rem' }}>
                        <div>
                            <h2
                                style={{
                                    fontFamily: "'Sora', sans-serif",
                                    fontSize: '1.05rem',
                                    fontWeight: 700,
                                    color: 'rgba(220,235,255,0.9)',
                                    margin: 0,
                                    letterSpacing: '-0.02em',
                                }}
                            >
                                Senti<span style={{ color: '#3399ff' }}>Sign</span>
                            </h2>
                            <p style={{ color: 'rgba(120,150,200,0.55)', fontSize: '0.68rem', margin: '2px 0 0', fontWeight: 300 }}>
                                Sign to speech · {activeModel.toUpperCase()} model
                            </p>
                        </div>
                        <div
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.4rem',
                                background: 'rgba(4,12,30,0.6)',
                                border: '1px solid rgba(51,153,255,0.18)',
                                borderRadius: 8,
                                padding: '5px 9px',
                                flexShrink: 0,
                            }}
                        >
                            <div
                                className={sessionActive ? 'sentisign-dot-blink' : ''}
                                style={{
                                    width: 6,
                                    height: 6,
                                    borderRadius: '50%',
                                    background: sessionActive ? '#3399ff' : 'rgba(80,110,170,0.4)',
                                    boxShadow: sessionActive ? '0 0 8px #3399ff' : 'none',
                                }}
                            />
                            <span
                                style={{
                                    fontFamily: "'JetBrains Mono', monospace",
                                    fontSize: '0.6rem',
                                    letterSpacing: '0.14em',
                                    color: sessionActive ? 'rgba(51,153,255,0.85)' : 'rgba(80,110,170,0.5)',
                                    textTransform: 'uppercase',
                                    fontWeight: 500,
                                }}
                            >
                                {sessionActive ? 'LIVE' : 'IDLE'}
                            </span>
                        </div>
                    </div>

                    <motion.button
                        type="button"
                        className="ss-session-btn"
                        onClick={sessionActive ? handleStopSession : handleStartSession}
                        whileHover={{ scale: 1.02, y: -1 }}
                        whileTap={{ scale: 0.97 }}
                        style={{
                            width: '100%',
                            borderRadius: 12,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '0.55rem',
                            fontSize: '0.84rem',
                            fontWeight: 700,
                            fontFamily: "'Sora', sans-serif",
                            letterSpacing: '0.04em',
                            transition: 'background 0.3s, box-shadow 0.3s',
                            ...(sessionActive
                                ? {
                                      background: 'linear-gradient(135deg, rgba(255,80,60,0.15), rgba(255,100,80,0.08))',
                                      border: '1px solid rgba(255,80,60,0.35)',
                                      color: '#ff7a6a',
                                      boxShadow: '0 4px 20px rgba(255,60,40,0.15)',
                                  }
                                : {
                                      background: 'linear-gradient(135deg, #1a6fff 0%, #3399ff 60%, #00c2ff 100%)',
                                      border: 'none',
                                      color: '#fff',
                                      boxShadow: '0 6px 24px rgba(51,153,255,0.4)',
                                  }),
                        }}
                    >
                        {sessionActive ? (
                            <>
                                <Square size={14} fill="currentColor" /> Stop Session
                            </>
                        ) : (
                            <>
                                <Play size={14} fill="currentColor" /> Start Session
                            </>
                        )}
                    </motion.button>

                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.45rem' }}>
                        {[
                            { icon: <RotateCcw size={13} />, label: 'Undo', action: removeLastWord },
                            { icon: <XCircle size={13} />, label: 'Reset', action: clearWords },
                        ].map((btn) => (
                            <motion.button
                                key={btn.label}
                                type="button"
                                className="ss-mini-btn"
                                onClick={btn.action}
                                whileHover={{ scale: 1.03, y: -1 }}
                                whileTap={{ scale: 0.96 }}
                                style={{
                                    flex: 1,
                                    borderRadius: 10,
                                    background: 'rgba(51,153,255,0.06)',
                                    border: '1px solid rgba(51,153,255,0.12)',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '0.35rem',
                                    color: 'rgba(140,175,240,0.7)',
                                    fontSize: '0.72rem',
                                    fontWeight: 600,
                                    fontFamily: "'Sora', sans-serif",
                                    letterSpacing: '0.02em',
                                }}
                            >
                                {btn.icon} {btn.label}
                            </motion.button>
                        ))}
                    </div>
                </div>

                <div className="ss-panel ss-card-pad" style={{ flexShrink: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
                        <span
                            style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: '0.62rem',
                                fontWeight: 500,
                                letterSpacing: '0.18em',
                                color: 'rgba(255,179,71,0.65)',
                                textTransform: 'uppercase',
                            }}
                        >
                            Emotion
                        </span>
                        <button
                            type="button"
                            onClick={() => setEmotionOverride(null)}
                            disabled={emotionOverride === null}
                            style={{
                                background: 'rgba(255,179,71,0.08)',
                                border: '1px solid rgba(255,179,71,0.2)',
                                borderRadius: 999,
                                padding: '4px 10px',
                                color: emotionOverride === null ? 'rgba(255,179,71,0.3)' : 'rgba(255,179,71,0.75)',
                                fontSize: '0.64rem',
                                fontWeight: 700,
                                fontFamily: "'JetBrains Mono', monospace",
                                letterSpacing: '0.12em',
                                textTransform: 'uppercase',
                                cursor: emotionOverride === null ? 'not-allowed' : 'pointer',
                            }}
                        >
                            Auto
                        </button>
                    </div>

                    <div className="ss-emotion-grid">
                        {EMOTION_OPTIONS.map((emotion) => {
                            const isSelected = selectedEmotion === emotion.type;
                            const isDetected = detectedEmotion === emotion.type;

                            return (
                                <button
                                    key={emotion.type}
                                    type="button"
                                    className="ss-emotion-btn"
                                    onClick={() => setEmotionOverride(emotion.type)}
                                    style={{
                                        borderRadius: 12,
                                        border: isSelected
                                            ? '1px solid rgba(51,153,255,0.52)'
                                            : isDetected
                                                ? '1px solid rgba(51,153,255,0.22)'
                                                : '1px solid rgba(255,255,255,0.08)',
                                        background: isSelected
                                            ? 'linear-gradient(180deg, rgba(51,153,255,0.2), rgba(51,153,255,0.08))'
                                            : isDetected
                                                ? 'rgba(51,153,255,0.08)'
                                                : 'rgba(255,255,255,0.03)',
                                        boxShadow: isSelected
                                            ? '0 0 0 1px rgba(51,153,255,0.08) inset, 0 12px 24px rgba(51,153,255,0.14)'
                                            : isDetected
                                                ? '0 0 18px rgba(51,153,255,0.08)'
                                                : 'none',
                                        color: isSelected ? '#dff0ff' : isDetected ? '#b7dcff' : 'rgba(200,220,255,0.72)',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '0.18rem',
                                        cursor: 'pointer',
                                        transition: 'all 0.22s ease',
                                    }}
                                >
                                    <span style={{ fontSize: '1.1rem', lineHeight: 1 }}>{emotion.emoji}</span>
                                    <span
                                        style={{
                                            fontFamily: "'JetBrains Mono', monospace",
                                            fontSize: '0.48rem',
                                            fontWeight: 700,
                                            letterSpacing: '0.08em',
                                            lineHeight: 1,
                                            whiteSpace: 'nowrap',
                                        }}
                                    >
                                        {emotion.label}
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                <div className="ss-panel-lit ss-card-pad" style={{ flexShrink: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.55rem' }}>
                        <span
                            style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: '0.62rem',
                                fontWeight: 500,
                                letterSpacing: '0.18em',
                                color: 'rgba(68,217,160,0.65)',
                                textTransform: 'uppercase',
                            }}
                        >
                            Sentence Output
                        </span>
                        <span
                            style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: '0.58rem',
                                color: emotionOverride ? 'rgba(51,153,255,0.7)' : 'rgba(120,150,200,0.45)',
                                textTransform: 'uppercase',
                                letterSpacing: '0.1em',
                            }}
                        >
                            {emotionOverride ? 'Manual' : 'Auto'}
                        </span>
                    </div>

                    {generationError && (
                        <div
                            style={{
                                marginBottom: '0.5rem',
                                borderRadius: 10,
                                padding: '6px 10px',
                                background: 'rgba(220,50,40,0.1)',
                                border: '1px solid rgba(220,50,40,0.25)',
                                color: 'rgba(255,120,110,0.82)',
                                fontSize: '0.68rem',
                                fontWeight: 600,
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                            }}
                        >
                            {generationError}
                        </div>
                    )}

                    <SentenceOutput
                        sentence={sentence}
                        audioUrl={audioUrl}
                        audioFilename={audioFilename}
                        generationStage={generationStage}
                        compact
                    />
                </div>

                <motion.button
                    type="button"
                    className="ss-generate-btn"
                    onClick={isGenerating ? cancelGeneration : handleGenerateAndSpeak}
                    disabled={!isGenerating && !canGenerate}
                    whileHover={(!isGenerating && !canGenerate) ? {} : { scale: 1.02, y: -2 }}
                    whileTap={(!isGenerating && !canGenerate) ? {} : { scale: 0.97 }}
                    style={{
                        width: '100%',
                        borderRadius: 12,
                        cursor: (!isGenerating && !canGenerate) ? 'not-allowed' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.55rem',
                        fontSize: '0.84rem',
                        fontWeight: 700,
                        fontFamily: "'Sora', sans-serif",
                        letterSpacing: '0.04em',
                        opacity: (!isGenerating && !canGenerate) ? 0.35 : 1,
                        flexShrink: 0,
                        ...(isGenerating
                            ? {
                                  background: 'linear-gradient(135deg, rgba(220,50,40,0.2), rgba(255,80,60,0.12))',
                                  border: '1px solid rgba(220,50,40,0.35)',
                                  color: '#ff7a6a',
                              }
                            : {
                                  background: 'linear-gradient(135deg, rgba(68,217,160,0.15) 0%, rgba(0,200,140,0.1) 100%)',
                                  border: '1px solid rgba(68,217,160,0.3)',
                                  color: '#44d9a0',
                                  boxShadow: canGenerate ? '0 4px 20px rgba(68,217,160,0.15)' : 'none',
                              }),
                    }}
                >
                    {isGenerating ? (
                        <>
                            <span
                                style={{
                                    width: 14,
                                    height: 14,
                                    borderRadius: '50%',
                                    border: '2px solid rgba(255,120,110,0.3)',
                                    borderTopColor: '#ff7a6a',
                                    display: 'inline-block',
                                    animation: 'sentisign-pulse-ring 0.7s linear infinite',
                                }}
                            />
                            Cancel
                            <span
                                style={{
                                    fontFamily: "'JetBrains Mono', monospace",
                                    fontSize: '0.58rem',
                                    letterSpacing: '0.1em',
                                    background: 'rgba(255,120,110,0.15)',
                                    border: '1px solid rgba(255,120,110,0.25)',
                                    borderRadius: 5,
                                    padding: '1px 6px',
                                    color: 'rgba(255,120,110,0.8)',
                                }}
                            >
                                {generationStage === 'sentence' ? 'NLG' : 'TTS'}
                            </span>
                        </>
                    ) : (
                        <>
                            <Volume2 size={16} /> Generate &amp; Speak
                        </>
                    )}
                </motion.button>

                <div
                    className="ss-active-row"
                    style={{
                        marginTop: 'auto',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '0.5rem',
                        flexShrink: 0,
                        minWidth: 0,
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', minWidth: 0 }}>
                        <span
                            className="ss-badge"
                            style={{
                                background: 'rgba(51,153,255,0.08)',
                                border: '1px solid rgba(51,153,255,0.18)',
                                color: 'rgba(51,153,255,0.68)',
                                padding: '2px 6px',
                                fontSize: '0.56rem',
                            }}
                        >
                            CONF ≥ {Math.round(minConfidence * 100)}%
                        </span>
                        {activeModel === 'mlp' && (
                            <span
                                className="ss-badge"
                                style={{
                                    background: 'rgba(51,153,255,0.08)',
                                    border: '1px solid rgba(51,153,255,0.18)',
                                    color: 'rgba(51,153,255,0.68)',
                                    padding: '2px 6px',
                                    fontSize: '0.56rem',
                                }}
                            >
                                HOLD {holdFramesSetting}f
                            </span>
                        )}
                        <span
                            className="ss-badge"
                            style={{
                                background: 'rgba(51,153,255,0.08)',
                                border: '1px solid rgba(51,153,255,0.18)',
                                color: 'rgba(51,153,255,0.68)',
                                padding: '2px 6px',
                                fontSize: '0.56rem',
                            }}
                        >
                            {activeModel.toUpperCase()}
                        </span>
                    </div>

                    <button
                        type="button"
                        onClick={() => {
                            setInitialized(false);
                            setSessionActive(false);
                            clearWords();
                            setCameraReady(false);
                        }}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            color: 'rgba(130,165,220,0.6)',
                            fontSize: '0.6rem',
                            fontFamily: "'JetBrains Mono', monospace",
                            letterSpacing: '0.12em',
                            textTransform: 'uppercase',
                            textDecoration: 'underline',
                            padding: 0,
                            flexShrink: 0,
                        }}
                    >
                        reconfigure
                    </button>
                </div>
            </motion.aside>
        </motion.div>
    );
};
