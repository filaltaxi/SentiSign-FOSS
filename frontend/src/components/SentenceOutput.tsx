import { useEffect, useState } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface SentenceOutputProps {
    sentence: string | null;
    audioUrl?: string | null;
    audioFilename?: string | null;
}

export function SentenceOutput({
    sentence,
    audioUrl,
    audioFilename,
}: SentenceOutputProps) {
    const [typedSentence, setTypedSentence] = useState<string>('');
    const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

    useEffect(() => {
        if (!sentence) {
            setTypedSentence('');
            return;
        }

        setTypedSentence('');
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

    useEffect(() => {
        setIsSpeaking(false);
    }, [audioUrl]);

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
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                        <audio
                            src={audioUrl}
                            controls
                            autoPlay
                            onPlay={() => setIsSpeaking(true)}
                            onPause={() => setIsSpeaking(false)}
                            onEnded={() => setIsSpeaking(false)}
                            className="h-11 w-full rounded-xl border border-border-color bg-white accent-brand shadow-[0_8px_16px_rgba(15,34,68,0.08)] sm:flex-1"
                        />
                        <a
                            href={audioUrl}
                            download={audioFilename || 'sentisign.wav'}
                            className="flex min-w-[120px] items-center justify-center gap-2 rounded-xl border border-[#c8ddff] bg-white px-4 py-2.5 text-[0.8rem] font-bold text-brand no-underline transition-all duration-300 hover:bg-[#f4f9ff] hover:shadow-[0_10px_20px_rgba(0,127,255,0.12)]"
                        >
                            Download
                        </a>
                    </div>
                </div>
            )}
        </div>
    );
}
