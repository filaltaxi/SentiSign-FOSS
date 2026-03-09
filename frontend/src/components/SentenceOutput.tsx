import { useCallback, useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Play, Square } from 'lucide-react';

interface SentenceOutputProps {
    sentence: string | null;
    audioUrl?: string | null;
    audioFilename?: string | null;
    generationStage?: 'idle' | 'sentence' | 'audio';
}

export function SentenceOutput({
    sentence,
    audioUrl,
    audioFilename,
    generationStage = 'idle',
}: SentenceOutputProps) {
    const [typedSentence, setTypedSentence] = useState<string>('');
    const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
    const [autoplayBlocked, setAutoplayBlocked] = useState<boolean>(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const showAudioProcessing = generationStage === 'audio' && !audioUrl;

    useEffect(() => {
        if (!sentence) {
            Promise.resolve().then(() => setTypedSentence(''));
            return;
        }

        Promise.resolve().then(() => setTypedSentence(''));
        let charIndex = 0;
        const typeInterval = setInterval(() => {
            charIndex += 1;
            setTypedSentence(sentence.slice(0, charIndex));

            if (charIndex >= sentence.length) {
                clearInterval(typeInterval);
            }
        }, 18);

        return () => clearInterval(typeInterval);
    }, [sentence]);

    const playAudio = useCallback(async () => {
        const audioEl = audioRef.current;
        if (!audioEl) return;

        try {
            await audioEl.play();
            setAutoplayBlocked(false);
        } catch {
            setAutoplayBlocked(true);
        }
    }, []);

    useEffect(() => {
        const audioEl = audioRef.current;
        if (!audioEl) {
            setAutoplayBlocked(false);
            Promise.resolve().then(() => setIsSpeaking(false));
            return;
        }

        audioEl.pause();
        audioEl.currentTime = 0;
        setAutoplayBlocked(false);
        Promise.resolve().then(() => setIsSpeaking(false));

        if (!audioUrl) {
            return;
        }

        const handleCanPlay = () => {
            void playAudio();
        };

        if (audioEl.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
            void playAudio();
            return;
        }

        audioEl.addEventListener('canplay', handleCanPlay, { once: true });
        return () => {
            audioEl.removeEventListener('canplay', handleCanPlay);
        };
    }, [audioUrl, playAudio]);

    const stopAudio = () => {
        const audioEl = audioRef.current;
        if (!audioEl) return;
        audioEl.pause();
        audioEl.currentTime = 0;
        setIsSpeaking(false);
    };

    return (
        <div className="flex flex-col gap-3">
            <div
                className={twMerge(
                    clsx(
                        "relative overflow-hidden rounded-[20px] border-2 border-[#b9d8ff] bg-[linear-gradient(160deg,#f8fbff_0%,#edf5ff_100%)] px-4 py-4 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.85)] transition-all duration-300",
                        !sentence ? "text-muted" : "sentence-reveal text-text shadow-[0_12px_22px_rgba(15,34,68,0.08)]"
                    )
                )}
            >
                <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-brand/30 to-transparent" />
                {typedSentence ? (
                    <p className="mx-auto min-h-[76px] max-w-[20ch] text-balance text-[clamp(1.35rem,2.6vw,2rem)] font-bold leading-[1.18] text-[#325784]">
                        {typedSentence}
                    </p>
                ) : (
                    <p className="mx-auto min-h-[76px] max-w-[22ch] text-balance text-[1rem] font-semibold leading-[1.35] text-[#5879a4]">
                        Generated sentence appears here
                    </p>
                )}
            </div>

            {showAudioProcessing && (
                <div className="animate-in fade-in zoom-in-95 duration-300">
                    <div className="relative overflow-hidden rounded-[999px] border border-[#cfe2ff] bg-[linear-gradient(180deg,#f8fbff_0%,#eaf3ff_100%)] px-5 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.88),0_14px_24px_rgba(15,34,68,0.08)]">
                        <div className="pointer-events-none absolute inset-y-0 left-[-35%] w-[35%] bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.9),transparent)] animate-[audio-processing-sheen_1.8s_ease-in-out_infinite]" />
                        <div className="relative flex min-h-[46px] items-center justify-center gap-3">
                            <div className="flex items-center gap-1.5" aria-hidden="true">
                                <span className="audio-processing-dot [animation-delay:0ms]" />
                                <span className="audio-processing-dot [animation-delay:180ms]" />
                                <span className="audio-processing-dot [animation-delay:360ms]" />
                            </div>
                            <div className="flex flex-col items-start">
                                <span className="text-[0.72rem] font-extrabold uppercase tracking-[0.18em] text-[#5c7fa9]">
                                    Processing Audio
                                </span>
                                <span className="text-[0.82rem] font-semibold text-[#54749c]">
                                    Preparing the spoken output...
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {audioUrl && (
                <div className="flex flex-col gap-2 animate-in fade-in zoom-in-95 duration-400">
                    {isSpeaking && (
                        <div className="flex items-end justify-center gap-1.5 rounded-xl border border-[#d3e6ff] bg-[#eff6ff] py-2">
                            <span className="wave-bar [animation-delay:0ms]" />
                            <span className="wave-bar [animation-delay:120ms]" />
                            <span className="wave-bar [animation-delay:210ms]" />
                            <span className="wave-bar [animation-delay:300ms]" />
                            <span className="wave-bar [animation-delay:420ms]" />
                        </div>
                    )}
                    {autoplayBlocked && (
                        <div className="rounded-xl border border-[#ffd4a8] bg-[#fff6ea] px-3 py-2 text-[0.76rem] font-semibold text-[#9a5a11]">
                            Browser autoplay was blocked. Press play to start the audio.
                        </div>
                    )}
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
	                        <audio
	                            src={audioUrl}
	                            controls
	                            ref={audioRef}
	                            onPlay={() => setIsSpeaking(true)}
	                            onPause={() => setIsSpeaking(false)}
	                            onEnded={() => setIsSpeaking(false)}
	                            className="sentisign-audio h-11 w-full overflow-hidden rounded-xl border border-border-color bg-[#0b1220] accent-brand shadow-[0_8px_16px_rgba(15,34,68,0.08)] sm:flex-1"
	                        />
                        <div className="flex gap-2 sm:flex-none">
                            <button
                                type="button"
                                onClick={stopAudio}
                                className="flex min-w-[104px] items-center justify-center gap-2 rounded-xl border border-[#ffd0cd] bg-white px-4 py-2.5 text-[0.8rem] font-bold text-[#cc2d2d] transition-all duration-300 hover:bg-[#fff2f2] hover:shadow-[0_10px_20px_rgba(255,59,48,0.14)]"
                            >
                                <Square size={14} fill="currentColor" /> Stop
                            </button>
                            {autoplayBlocked && (
                                <button
                                    type="button"
                                    onClick={() => void playAudio()}
                                    className="flex min-w-[104px] items-center justify-center gap-2 rounded-xl border border-[#c8ddff] bg-white px-4 py-2.5 text-[0.8rem] font-bold text-brand transition-all duration-300 hover:bg-[#f4f9ff] hover:shadow-[0_10px_20px_rgba(0,127,255,0.12)]"
                                >
                                    <Play size={14} fill="currentColor" /> Play
                                </button>
                            )}
                            <a
                                href={audioUrl}
                                download={audioFilename || 'sentisign.wav'}
                                className="flex min-w-[120px] items-center justify-center gap-2 rounded-xl border border-[#c8ddff] bg-white px-4 py-2.5 text-[0.8rem] font-bold text-brand no-underline transition-all duration-300 hover:bg-[#f4f9ff] hover:shadow-[0_10px_20px_rgba(0,127,255,0.12)]"
                            >
                                Download
                            </a>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
