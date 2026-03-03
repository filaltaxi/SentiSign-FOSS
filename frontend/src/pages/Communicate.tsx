import React, { useState, useRef, useEffect, useCallback } from 'react';
import { WebcamPane } from '../components/WebcamPane';
import { WordBuffer } from '../components/WordBuffer';
import { EmotionStrip } from '../components/EmotionStrip';
import { SentenceOutput } from '../components/SentenceOutput';
import type { EmotionType } from '../components/EmotionStrip';
import { generateSentence, generateAudio } from '../lib/api.ts';
import { Play, Square, RotateCcw, XCircle } from 'lucide-react';

const HOLD_FRAMES = 10;
const MIN_CONFIDENCE = 0.60;

export const Communicate: React.FC = () => {
    const [wordBuffer, setWordBuffer] = useState<string[]>([]);
    const [sentence, setSentence] = useState<string>('');
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [audioFilename, setAudioFilename] = useState<string | null>(null);
    const [generationStage, setGenerationStage] = useState<'idle' | 'sentence' | 'audio'>('idle');
    const [sessionActive, setSessionActive] = useState<boolean>(false);
    const [detectedEmotion, setDetectedEmotion] = useState<EmotionType>('neutral');
    const [emotionOverride, setEmotionOverride] = useState<EmotionType | null>(null);
    const [signLabel, setSignLabel] = useState<string>('No sign detected');
    const [confidence, setConfidence] = useState<number>(0);
    const selectedEmotion = emotionOverride ?? detectedEmotion;
    const canGenerate = wordBuffer.length > 0;
    const isGenerating = generationStage !== 'idle';

    const trackingRef = useRef<{ holdCounter: number; currentClass: string | null; lastWord: string }>({
        holdCounter: 0,
        currentClass: null,
        lastWord: '',
    });

    const generationAbortRef = useRef<AbortController | null>(null);
    const generationRunIdRef = useRef<number>(0);

    const cancelGeneration = useCallback(() => {
        generationAbortRef.current?.abort();
        generationAbortRef.current = null;
        setGenerationStage('idle');
    }, []);

    useEffect(() => {
        return () => {
            generationAbortRef.current?.abort();
        };
    }, []);

    const handleSignDetected = useCallback(
        (word: string | null, cls: string | null, conf: number) => {
            setConfidence(conf);
            setSignLabel(word ? `${cls} -> ${word}` : (cls ?? 'No sign detected'));

            if (!sessionActive || !cls) {
                trackingRef.current.currentClass = null;
                trackingRef.current.holdCounter = 0;
                return;
            }

            if (cls !== trackingRef.current.currentClass) {
                trackingRef.current.currentClass = cls;
                trackingRef.current.holdCounter = 0;
                return;
            }

            if (conf >= MIN_CONFIDENCE) {
                trackingRef.current.holdCounter += 1;
            } else {
                trackingRef.current.holdCounter = Math.max(0, trackingRef.current.holdCounter - 1);
            }

            if (
                trackingRef.current.holdCounter >= HOLD_FRAMES &&
                word &&
                word !== trackingRef.current.lastWord
            ) {
                setWordBuffer((prev) => [...prev, word]);
                trackingRef.current.lastWord = word;
                trackingRef.current.holdCounter = 0;
            }
        },
        [sessionActive]
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

    const removeLastWord = () => {
        setWordBuffer((prev) => prev.slice(0, -1));
    };

    const clearWords = () => {
        cancelGeneration();
        setWordBuffer([]);
        setSentence('');
        setAudioUrl(null);
        setAudioFilename(null);
        setSignLabel('No sign detected');
        setConfidence(0);
        trackingRef.current.holdCounter = 0;
        trackingRef.current.currentClass = null;
        trackingRef.current.lastWord = '';
    };

    const handleGenerateAndSpeak = async () => {
        if (wordBuffer.length === 0) return;
        if (sessionActive) {
            setSessionActive(false);
            trackingRef.current.holdCounter = 0;
            trackingRef.current.currentClass = null;
        }

        generationAbortRef.current?.abort();
        const abortController = new AbortController();
        generationAbortRef.current = abortController;

        generationRunIdRef.current += 1;
        const runId = generationRunIdRef.current;

        setGenerationStage('sentence');
        try {
            const data = await generateSentence(wordBuffer, selectedEmotion, { signal: abortController.signal });
            if (generationRunIdRef.current !== runId) return;

            if (data.sentence) {
                setSentence(data.sentence);
                setGenerationStage('audio');
                const audioData = await generateAudio(data.sentence, selectedEmotion, { signal: abortController.signal });
                if (generationRunIdRef.current !== runId) return;

                if (audioData.audio_url) {
                    setAudioUrl(`${audioData.audio_url}?t=${Date.now()}`);
                    setAudioFilename(audioData.audio_file);
                }
            }
        } catch (error) {
            if (error instanceof DOMException && error.name === 'AbortError') return;
            if (error instanceof Error && error.name === 'AbortError') return;
            console.error('Generation error:', error);
        } finally {
            if (generationRunIdRef.current === runId) {
                generationAbortRef.current = null;
                setGenerationStage('idle');
            }
        }
    };

    return (
        <div className="mx-auto grid h-[calc(100dvh-var(--app-nav-h))] w-full max-w-[1400px] grid-cols-1 items-start gap-4 overflow-hidden px-4 py-4 sm:px-8 lg:grid-cols-[minmax(0,1fr)_390px] lg:gap-6 [@media(max-height:820px)]:gap-3 [@media(max-height:820px)]:py-3">
            <section className="min-h-0 rounded-[26px] border border-border-color bg-surface/95 p-4 shadow-[0_16px_36px_rgba(15,34,68,0.12)] lg:p-5 flex flex-col gap-3 overflow-hidden [@media(max-height:820px)]:p-3">
                <header className="flex items-end justify-between gap-4">
                    <div>
                        <h1 className="font-heading text-[clamp(1.65rem,3.4vw,2.6rem)] font-extrabold leading-[1.12] pb-[0.06em] tracking-tight text-text">
                            Senti<span className="inline-block text-brand-end [text-shadow:0_1px_0_rgba(0,0,0,0.12),0_14px_26px_rgba(0,127,255,0.22)]">Sign</span>
                        </h1>
                        <p className="mt-1.5 max-w-[520px] text-[0.93rem] text-muted [@media(max-height:820px)]:hidden">
                            Sign naturally to the camera and convert it into clear spoken sentences.
                        </p>
                    </div>
                    <div className={`hidden rounded-full border px-3 py-1 text-[0.72rem] font-semibold uppercase tracking-[0.16em] md:block ${sessionActive ? 'border-[#bfdbff] bg-[#edf5ff] text-brand' : 'border-[#d9e7fb] bg-white text-muted'}`}>
                        {sessionActive ? 'Live Feed' : 'Camera Ready'}
                    </div>
                </header>

                <div className={`mx-auto w-full max-w-[560px] overflow-hidden rounded-2xl border bg-[#eaf2ff] transition-all duration-400 md:max-w-[620px] xl:max-w-[660px] ${sessionActive ? 'border-[#9fc9ff] shadow-[0_18px_34px_rgba(0,127,255,0.18)]' : 'border-[#c9defd]'}`}>
                    <WebcamPane
                        isActive={sessionActive}
                        onEmotionDetected={setDetectedEmotion}
                        onSignDetected={handleSignDetected}
                        currentEmotion={detectedEmotion}
                        wordLabel={sessionActive ? signLabel : 'No sign detected'}
                        confidence={sessionActive ? confidence : 0}
                    />
                </div>

                <WordBuffer words={wordBuffer} />
            </section>

            <aside>
                <section className="min-h-0 flex flex-col gap-3 rounded-[24px] border border-border-color bg-surface/95 p-4 shadow-[0_12px_30px_rgba(15,34,68,0.1)] lg:p-5 [@media(max-height:820px)]:p-3">
                    <div className="flex items-center justify-between">
                        <h2 className="font-heading text-[0.74rem] font-bold uppercase tracking-[0.18em] text-muted">Control Panel</h2>
                        <div className="flex items-center gap-2 rounded-full border border-[#d2e4ff] bg-[#f3f8ff] px-2.5 py-1">
                            <span className={`h-2 w-2 rounded-full transition-all duration-300 ${sessionActive ? 'bg-brand shadow-[0_0_12px_rgba(0,127,255,0.7)]' : 'bg-[#a9c0df]'}`} />
                            <span className="text-[0.66rem] font-bold uppercase tracking-[0.14em] text-muted">
                                {sessionActive ? 'Live' : 'Idle'}
                            </span>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-border-color bg-[#f9fbff] p-3">
                        <h3 className="mb-2.5 font-heading text-[0.7rem] font-bold uppercase tracking-[0.18em] text-muted">Session Controls</h3>
                        <button
                            onClick={() => {
                                setSessionActive((prev) => {
                                    const next = !prev;
                                    if (next) {
                                        trackingRef.current.holdCounter = 0;
                                        trackingRef.current.currentClass = null;
                                    }
                                    return next;
                                });
                            }}
                            className={`flex h-11 w-full items-center justify-center gap-2 rounded-xl border text-[0.86rem] font-bold tracking-wide transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_14px_26px_rgba(15,34,68,0.14)] ${sessionActive
                                ? 'border-[#ffc4a5] bg-[#fff1e8] text-[#c85a21] hover:bg-[#ffe7d9]'
                                : 'border-[#c8ddff] bg-white text-brand hover:bg-[#f4f9ff]'
                                }`}
                        >
                            {sessionActive ? <><Square size={15} fill="currentColor" /> Stop</> : <><Play size={15} fill="currentColor" /> Initiate</>}
                        </button>

                        <div className="mt-2 flex gap-2">
                            <button
                                onClick={removeLastWord}
                                className="flex h-9 flex-1 items-center justify-center gap-1.5 rounded-xl border border-border-color bg-white text-[0.78rem] font-semibold text-muted transition-all duration-200 hover:-translate-y-0.5 hover:border-brand hover:text-brand hover:shadow-[0_8px_16px_rgba(15,34,68,0.08)]"
                            >
                                <RotateCcw size={14} /> Undo
                            </button>
                            <button
                                onClick={clearWords}
                                className="flex h-9 flex-1 items-center justify-center gap-1.5 rounded-xl border border-border-color bg-white text-[0.78rem] font-semibold text-muted transition-all duration-200 hover:-translate-y-0.5 hover:border-[#b8d4ff] hover:text-text hover:shadow-[0_8px_16px_rgba(15,34,68,0.08)]"
                            >
                                <XCircle size={14} /> Reset
                            </button>
                        </div>

                        <button
                            onClick={isGenerating ? cancelGeneration : handleGenerateAndSpeak}
                            disabled={!isGenerating && !canGenerate}
                            className={`mt-2 flex h-11 w-full items-center justify-center gap-2 rounded-xl border border-transparent text-[0.86rem] font-bold text-white transition-all duration-200 hover:-translate-y-0.5 hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-45 disabled:saturate-50 disabled:transform-none disabled:shadow-none ${isGenerating
                                ? 'bg-gradient-to-r from-[#ff3b30] to-[#ff7a59] hover:shadow-[0_14px_24px_rgba(255,59,48,0.22)]'
                                : 'bg-gradient-to-r from-brand to-brand-end hover:shadow-[0_14px_24px_rgba(0,127,255,0.28)]'
                                }`}
                        >
                            {isGenerating ? (
                                <>
                                    <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-white/40 border-t-white animate-[spin_0.7s_linear_infinite]" />
                                    Cancel
                                    <span className="ml-1 rounded-full bg-white/15 px-2 py-0.5 text-[0.64rem] font-extrabold uppercase tracking-[0.14em] text-white/90">
                                        {generationStage === 'sentence' ? 'SENTENCE' : 'AUDIO'}
                                    </span>
                                </>
                            ) : (
                                <>&#10022; Generate &amp; Speak</>
                            )}
                        </button>
                    </div>

                    <div className="rounded-2xl border border-border-color bg-[#f9fbff] p-3">
                        <div className="mb-3 flex items-center justify-between">
                            <h3 className="font-heading text-[0.7rem] font-bold uppercase tracking-[0.18em] text-muted">Emotion</h3>
                            <button
                                onClick={() => setEmotionOverride(null)}
                                disabled={emotionOverride === null}
                                className="rounded-md border border-border-color bg-white px-2 py-1 text-[0.68rem] font-bold uppercase tracking-[0.12em] text-muted transition-all duration-200 hover:border-brand hover:text-brand disabled:cursor-not-allowed disabled:opacity-45"
                            >
                                Auto
                            </button>
                        </div>
                        <div className="mb-3 text-[0.78rem] text-muted">
                            Auto-detected: <span className="font-semibold capitalize text-text">{detectedEmotion}</span>
                            {emotionOverride && <span className="ml-2 rounded-full bg-[#ecf4ff] px-2 py-0.5 text-[0.66rem] font-bold uppercase tracking-[0.13em] text-brand">Manual override</span>}
                        </div>
                        <EmotionStrip currentEmotion={selectedEmotion} onSelectEmotion={setEmotionOverride} />
                    </div>

                    <div className="rounded-2xl border border-border-color bg-[#f9fbff] p-3">
                        <div className="mb-2.5 flex items-center justify-between">
                            <h3 className="font-heading text-[0.7rem] font-bold uppercase tracking-[0.18em] text-muted">Generated Sentence</h3>
                            <div className="flex items-center gap-2">
                                {emotionOverride && <span className="rounded-full bg-[#ecf4ff] px-2 py-0.5 text-[0.66rem] font-bold uppercase tracking-[0.13em] text-brand">Manual</span>}
                                <span className="rounded-full border border-[#d2e4ff] bg-white px-2 py-0.5 text-[0.66rem] font-bold uppercase tracking-[0.13em] text-muted capitalize">
                                    {selectedEmotion}
                                </span>
                            </div>
                        </div>
                        <SentenceOutput
                            sentence={sentence}
                            audioUrl={audioUrl}
                            audioFilename={audioFilename}
                        />
                    </div>
                </section>
            </aside>
        </div>
    );
};
